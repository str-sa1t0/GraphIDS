# utils/dataloaders.py
import os
import pickle
import shutil
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.utils import degree

torch.serialization.add_safe_globals(
    [
        torch_geometric.data.data.DataEdgeAttr,  # type: ignore[attr-defined]
        torch_geometric.data.data.DataTensorAttr,  # type: ignore[attr-defined]
        torch_geometric.data.storage.GlobalStorage,  # type: ignore[attr-defined]
    ]
)


# -----------------------------
# Collate / Sequential Dataset
# -----------------------------
def collate_fn(batch):
    sequences, masks = zip(*batch, strict=False)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return sequences_padded, masks_padded


class SequentialDataset(TorchDataset):
    def __init__(self, data, window, device, step=None):
        self.data = data
        self.window = window
        self.device = device
        self.step = window if step is None else step

    def __getitem__(self, index):
        start_idx = index * self.step
        end_idx = min(start_idx + self.window, len(self.data))
        x = self.data[start_idx:end_idx].to(self.device)
        mask = torch.ones_like(x, dtype=torch.bool).to(self.device)
        return x, mask

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.step + 1)


# -----------------------------
# Utilities
# -----------------------------
def _node_to_display_str(v) -> str:
    """
    node id / ip表現の復元（IPv4整数っぽい場合は dotted に変換）
    """
    s = str(v)
    if "." in s:
        return s
    try:
        iv = int(float(v))
        if (2**24) <= iv <= (2**32 - 1):
            import ipaddress

            return str(ipaddress.IPv4Address(iv))
    except Exception:
        pass
    return s


def _compute_struct_node_features(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Graph structure based node features for "infrastructure-likeness".

    Return: [N, 5]
      [ log1p(deg_in), log1p(deg_out), log1p(uniq_in), log1p(uniq_out), log1p(kcore) ]

    - deg_in/out: multi-edge degree (directed)
    - uniq_in/out: unique neighbor count (directed)
    - kcore: undirected core number (optional, networkx)
    """
    # Ensure CPU
    if edge_index.is_cuda:
        edge_index = edge_index.detach().cpu()

    src = edge_index[0]
    dst = edge_index[1]

    # Multi-edge degree
    deg_out = degree(src, num_nodes=num_nodes).to(torch.float32)
    deg_in = degree(dst, num_nodes=num_nodes).to(torch.float32)

    # Unique neighbors: unique (src,dst)
    pairs = torch.stack([src, dst], dim=1)  # [E,2]
    uniq_pairs = torch.unique(pairs, dim=0)
    uniq_src = uniq_pairs[:, 0]
    uniq_dst = uniq_pairs[:, 1]

    uniq_out = degree(uniq_src, num_nodes=num_nodes).to(torch.float32)
    uniq_in = degree(uniq_dst, num_nodes=num_nodes).to(torch.float32)

    # k-core (undirected)
    kcore = torch.zeros(num_nodes, dtype=torch.float32)
    try:
        import networkx as nx

        s = src.numpy()
        d = dst.numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(zip(s, d))
        core = nx.core_number(G)
        for n, k in core.items():
            if 0 <= int(n) < num_nodes:
                kcore[int(n)] = float(k)
    except Exception:
        # networkx 未導入 or 失敗時はゼロでOK（モデルは他の特徴で動く）
        pass

    # Stabilize scale
    deg_in = torch.log1p(deg_in)
    deg_out = torch.log1p(deg_out)
    uniq_in = torch.log1p(uniq_in)
    uniq_out = torch.log1p(uniq_out)
    kcore = torch.log1p(kcore)

    extra = torch.stack([deg_in, deg_out, uniq_in, uniq_out, kcore], dim=1)  # [N,5]
    return extra


# -----------------------------
# Dataset
# -----------------------------
class NetFlowDataset:
    def __init__(
        self,
        name,
        data_dir,
        force_reload=False,
        fraction=None,
        data_type="benign",
        seed=42,
    ):
        self.name = name
        self.data_dir = data_dir
        self.fraction = fraction
        self.data_type = data_type
        self.seed = seed

        # Setup directories
        graph_dir = os.path.join(data_dir, "pyg_graph_data")
        if fraction is not None:
            assert 0 < fraction < 1
            fraction_str = str(fraction).replace(".", "_")
            self.processed_dir = os.path.join(graph_dir, f"{name}_{fraction_str}")
        else:
            self.processed_dir = os.path.join(graph_dir, name)

        self.raw_dir = os.path.join(data_dir, name)

        # Handle force reload
        if force_reload and os.path.exists(self.processed_dir):
            print(f"Force reload: Removing existing processed data at {self.processed_dir}")
            shutil.rmtree(self.processed_dir)

            # Also remove the scaler for this dataset
            scaler_path = os.path.join("scalers", f"scaler_{self.name}.pkl")
            if os.path.exists(scaler_path):
                print(f"Removing old scaler: {scaler_path}")
                os.remove(scaler_path)

        # Process if needed
        if self._needs_processing():
            self._process()

        # Load processed graphs
        self.train_graph = torch.load(os.path.join(self.processed_dir, "train.pt"))[0]
        self.val_graph = torch.load(os.path.join(self.processed_dir, "val.pt"))[0]
        self.test_graph = torch.load(os.path.join(self.processed_dir, "test.pt"))[0]

        # node_id -> 元IP mapping
        self.mapping: Optional[List[str]] = None
        mapping_path = os.path.join(self.processed_dir, "node_mapping.pkl")
        if os.path.exists(mapping_path):
            with open(mapping_path, "rb") as f:
                self.mapping = pickle.load(f)
            print(f"[OK] loaded node mapping: {mapping_path} (size={len(self.mapping)})")
        else:
            print("[WARN] node_mapping.pkl not found. src_ip/dst_ip will remain as node IDs.")

    def _needs_processing(self):
        if not os.path.exists(self.processed_dir):
            return True
        required_files = ["train.pt", "val.pt", "test.pt", "node_mapping.pkl"]
        for filename in required_files:
            if not os.path.exists(os.path.join(self.processed_dir, filename)):
                return True
        return False

    def _process(self):
        print(f"Processing dataset {self.name}...")
        os.makedirs(self.processed_dir, exist_ok=True)

        df = pd.read_csv(os.path.join(self.raw_dir, f"{self.name}.csv"))

        if self.fraction is not None:
            df = df.groupby(by="Attack").sample(frac=self.fraction, random_state=self.seed)

        # raw meta columns (keep unscaled)
        for col in ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "L4_DST_PORT", "PROTOCOL"]:
            if col in df.columns:
                df[col + "_RAW"] = df[col]

        x = df.drop(columns=["Attack", "Label"])
        y = df[["Attack", "Label"]]

        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0)

        # edge feature selection
        if "v3" in self.name:
            edge_features = [
                col for col in x.columns
                if col
                not in [
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                    "FLOW_END_MILLISECONDS",
                    "FLOW_START_MILLISECONDS",
                ]
            ]
        else:
            edge_features = [col for col in x.columns if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]]

        df = pd.concat([x, y], axis=1)

        df_train, df_val_test = train_test_split(
            df, test_size=0.2, random_state=self.seed, stratify=y["Attack"]
        )

        if self.data_type == "benign":
            df_train = df_train[df_train["Label"] == 0]

        # scaler fit on train only
        scaler_path = os.path.join("scalers", f"scaler_{self.name}.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            except Exception as e:
                print(f"Failed to load scaler: {e}. Creating new one.")
                scaler = MinMaxScaler()
                scaler.fit(df_train[edge_features])
        else:
            scaler = MinMaxScaler()
            scaler.fit(df_train[edge_features])
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        # scale train/val/test
        df_train[edge_features] = scaler.transform(df_train[edge_features])

        df_val_test_scaled = scaler.transform(df_val_test[edge_features])
        df_val_test[edge_features] = np.clip(df_val_test_scaled, -10, 10)

        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=self.seed,
            stratify=df_val_test["Attack"],
        )

        # time sorting for v3
        if "v3" in self.name and "FLOW_START_MILLISECONDS" in df_train.columns:
            df_train = df_train.sort_values(by="FLOW_START_MILLISECONDS")
            df_val = df_val.sort_values(by="FLOW_START_MILLISECONDS")
            df_test = df_test.sort_values(by="FLOW_START_MILLISECONDS")

        # build node set from all splits
        unique_nodes = pd.concat(
            [
                df_train["IPV4_SRC_ADDR"],
                df_train["IPV4_DST_ADDR"],
                df_val["IPV4_SRC_ADDR"],
                df_val["IPV4_DST_ADDR"],
                df_test["IPV4_SRC_ADDR"],
                df_test["IPV4_DST_ADDR"],
            ],
            ignore_index=True,
        ).unique()

        node_map = {node: i for i, node in enumerate(unique_nodes)}
        num_nodes = len(node_map)

        # node_mapping.pkl save
        mapping = [_node_to_display_str(v) for v in unique_nodes]
        mapping_path = os.path.join(self.processed_dir, "node_mapping.pkl")
        with open(mapping_path, "wb") as f:
            pickle.dump(mapping, f)
        print(f"[OK] saved node mapping: {mapping_path} (size={len(mapping)})")

        # -----------------------------
        # ★ Global structural node features (k-core / degree / unique-neighbor)
        #   We compute from ALL flows so that train/val/test node feature is consistent.
        # -----------------------------
        df_all = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

        all_src = np.array([node_map[ip] for ip in df_all["IPV4_SRC_ADDR"]], dtype=np.int64)
        all_dst = np.array([node_map[ip] for ip in df_all["IPV4_DST_ADDR"]], dtype=np.int64)
        edge_index_all = torch.tensor(np.array([all_src, all_dst]), dtype=torch.long)

        extra_struct = _compute_struct_node_features(edge_index_all, num_nodes)  # [N,5]

        # baseline node features (keep original behavior)
        # NOTE: original GraphIDS used x with same dim as edge feature dim
        baseline_dim = len(edge_features)
        x_node_base = torch.ones(num_nodes, baseline_dim, dtype=torch.float32)

        # final node features
        x_node_global = torch.cat([x_node_base, extra_struct], dim=1)  # [N, baseline_dim+5]

        datasets = {"train": df_train, "val": df_val, "test": df_test}

        for split_name, df_split in datasets.items():
            src_nodes = np.array([node_map[ip] for ip in df_split["IPV4_SRC_ADDR"]], dtype=np.int64)
            dst_nodes = np.array([node_map[ip] for ip in df_split["IPV4_DST_ADDR"]], dtype=np.int64)

            edge_index = torch.tensor(np.array([src_nodes, dst_nodes]), dtype=torch.long)
            edge_attr = torch.tensor(df_split[edge_features].values, dtype=torch.float32)
            edge_labels = torch.tensor(df_split["Label"].values, dtype=torch.long)

            # ★ Use global node features (consistent across splits)
            data = Data(
                x=x_node_global.clone(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_labels=edge_labels,
                num_nodes=num_nodes,
            )

            # raw meta attributes
            if "FLOW_START_MILLISECONDS_RAW" in df_split.columns:
                ts_ms = df_split["FLOW_START_MILLISECONDS_RAW"].fillna(0).astype(np.int64).to_numpy()
                # keep same behavior as your main.py expects (int->ns)
                data.TIMESTAMP = torch.tensor(ts_ms * 1_000_000, dtype=torch.long)

            if "L4_DST_PORT_RAW" in df_split.columns:
                dport = df_split["L4_DST_PORT_RAW"].fillna(0).astype(np.int64).to_numpy()
                data.L4_DST_PORT = torch.tensor(dport, dtype=torch.long)

            if "PROTOCOL_RAW" in df_split.columns:
                proto = df_split["PROTOCOL_RAW"].fillna(0).astype(np.int64).to_numpy()
                data.PROTOCOL = torch.tensor(proto, dtype=torch.long)

            torch.save([data], os.path.join(self.processed_dir, f"{split_name}.pt"))

        seed_file = os.path.join(self.processed_dir, ".seed")
        with open(seed_file, "w") as f:
            f.write(str(self.seed))

        print("Done!")

    def __len__(self):
        return 3

    @property
    def num_node_features(self):
        return self.train_graph.x.shape[1]

    @property
    def num_edge_features(self):
        return self.train_graph.edge_attr.shape[1]

    @property
    def num_nodes(self):
        return self.train_graph.num_nodes
