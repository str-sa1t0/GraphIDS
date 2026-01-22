# main.py
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import precision_recall_curve
from torch_geometric.loader import LinkNeighborLoader

from models.graphids import GraphIDS
from utils.dataloaders import NetFlowDataset
from utils.parser import Parser
from utils.trainers import test, train_encoder
import rank_ips
from rank_ips import rank_infra, InfraRankConfig


# Suppress this warning: even if in prototype stage, it works correctly for our use case
warnings.filterwarnings(
    "ignore", message="The PyTorch API of nested tensors is in prototype stage"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_get_edge_attr_numpy(g, attr_name: str, expected_len: int) -> np.ndarray | None:
    """
    PyG Data の属性が存在し、かつ edge数に合うなら numpy で返す。
    例: g.TIMESTAMP, g.L4_DST_PORT, g.PROTOCOL など
    """
    if not hasattr(g, attr_name):
        return None
    x = getattr(g, attr_name)
    if x is None:
        return None
    try:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
    except Exception:
        return None

    # shape補正（(E,1) -> (E,)）
    if x.ndim >= 2 and x.shape[0] == expected_len and x.shape[1] == 1:
        x = x.reshape(-1)

    if x.shape[0] != expected_len:
        return None
    return x


def main(run):
    config = run.config
    set_seed(int(config.seed))

    dataset = NetFlowDataset(
        name=config.dataset,
        data_dir=config.data_dir,
        force_reload=config.reload_dataset,
        fraction=config.fraction,
        data_type=config.data_type,
        seed=config.seed,
    )

    ndim_in = dataset.num_node_features
    edim_in = dataset.num_edge_features
    print("Number of features:", edim_in)
    print("Node feature dim:", ndim_in)

    model = GraphIDS(
        ndim_in=ndim_in,
        edim_in=edim_in,
        edim_out=config.edim_out,
        embed_dim=config.ae_embedding_dim,
        num_heads=4,
        num_layers=config.num_layers,
        window_size=config.window_size,
        dropout=config.dropout,
        ae_dropout=config.ae_dropout,
        positional_encoding=config.positional_encoding,
        agg_type=config.agg_type,
        mask_ratio=config.mask_ratio,
    ).to(device)

    # -----------------------------
    # Optimizer: include extra heads if they exist
    # -----------------------------
    param_groups = [
        {"params": model.encoder.parameters(), "weight_decay": config.weight_decay},
        {"params": model.transformer.parameters(), "weight_decay": config.ae_weight_decay},
    ]
    if hasattr(model, "temporal_head"):
        param_groups.append({"params": model.temporal_head.parameters(), "weight_decay": config.ae_weight_decay})
    if hasattr(model, "proj_head"):
        param_groups.append({"params": model.proj_head.parameters(), "weight_decay": config.ae_weight_decay})

    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate)

    # -----------------------------
    # Checkpoint
    # -----------------------------
    checkpoint = config.checkpoint
    if checkpoint is not None and os.path.exists(checkpoint):
        print("Loading model from checkpoint")
        start_epoch, threshold = model.load_checkpoint(checkpoint, optimizer)
        run.config.epoch = start_epoch
    else:
        checkpoint_id = run.name if config.wandb else config.seed
        checkpoint = f"checkpoints/GraphIDS_{config.dataset}_{checkpoint_id}.ckpt"
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        start_epoch = 0
        threshold = None

    shuffle = config.positional_encoding == "None"
    fanout_list = [config.fanout] if config.fanout != -1 else [-1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cpu_count = os.cpu_count()
    recommended_workers = min(cpu_count, 6) if cpu_count is not None else 0
    persistent = True if recommended_workers and recommended_workers > 0 else False

    # -----------------------------
    # DataLoaders
    # -----------------------------
    if start_epoch >= config.num_epochs or config.test:
        print("Model already trained OR test-only mode")

        test_loader = LinkNeighborLoader(
            data=dataset.test_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.test_graph.edge_index,
            edge_label=dataset.test_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=persistent,
            drop_last=False,
        )
    else:
        train_loader = LinkNeighborLoader(
            data=dataset.train_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.train_graph.edge_index,
            edge_label=dataset.train_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=persistent,
            drop_last=True,
        )
        val_loader = LinkNeighborLoader(
            data=dataset.val_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.val_graph.edge_index,
            edge_label=dataset.val_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=persistent,
            drop_last=True,
        )
        test_loader = LinkNeighborLoader(
            data=dataset.test_graph,
            num_neighbors=fanout_list,
            edge_label_index=dataset.test_graph.edge_index,
            edge_label=dataset.test_graph.edge_labels,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=recommended_workers,
            pin_memory=True,
            persistent_workers=persistent,
            drop_last=False,
        )

        print("Starting training...")
        model, threshold = train_encoder(
            model=model,
            window_size=config.window_size,
            step_percent=config.step_percent,
            ae_batch_size=config.ae_batch_size,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            start_epoch=start_epoch,
            num_epochs=config.num_epochs,
            optimizer=optimizer,
            run=run,
            patience=config.patience,
            checkpoint=checkpoint,
            device=device,
        )

    # -----------------------------
    # Evaluate
    # -----------------------------
    print("Evaluating on test set...")
    test_f1, test_pr_auc, errors, test_labels, prediction_time = test(
        model,
        test_loader,
        config.ae_batch_size,
        config.window_size,
        device,
        threshold=threshold,
    )

    if errors.numel() and test_labels.numel() and errors.numel() == test_labels.numel():
        precision, recall, _ = precision_recall_curve(test_labels.cpu(), errors.cpu())
    else:
        precision, recall = np.array([]), np.array([])

    if config.save_curve and errors.numel() and test_labels.numel():
        run.log(
            {
                "Precision-Recall Curve": wandb.plot.pr_curve(
                    y_true=test_labels.cpu().numpy(),
                    y_probas=errors.cpu().numpy(),
                    title="Precision-Recall Curve",
                ),
            }
        )
        os.makedirs("curves", exist_ok=True)
        np.savez(
            f"curves/precision_recall_{run.name}.npz",
            precision=precision,
            recall=recall,
        )

    test_pred = (errors > float(threshold)).int() if threshold is not None and errors.numel() else torch.zeros_like(test_labels)

    print(f"Test macro F1-score: {test_f1:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")
    print(f"Test prediction time: {prediction_time:.4f} seconds")

    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU memory usage: {peak_memory_mb:.2f} MB")
    else:
        peak_memory_mb = 0.0

    run.log(
        {
            "final_test_f1": test_f1,
            "final_test_pr_auc": test_pr_auc,
            "test_threshold": threshold,
            "test_prediction_time": prediction_time,
            "peak_gpu_memory_mb": peak_memory_mb,
            "Validation Confusion Matrix": wandb.plot.confusion_matrix(
                y_true=test_labels.ravel().tolist(),
                preds=test_pred.ravel().tolist(),
                class_names=["Benign", "Malicious"],
                title="Validation Confusion Matrix",
            ),
        }
    )

    # ============================================================
    # Post-processing: Suspicious IP Ranking
    # ============================================================
    print("Generating suspicious IP ranking...")
    rank_method = getattr(config, "rank_method", "topk_sum")

    # ここで「何のmethodを使うか」を必ず出す
    print(
        "[RANK] "
        f"rank_method={rank_method} "
        f"tau(threshold)={float(threshold) if threshold is not None else None} "
        f"window_sec=3600 key_mode=dst_ip topk=10 topn=20"
    )

    run_id = run.name if getattr(run, "name", None) is not None else (
        run.id if getattr(run, "id", None) is not None else f"seed{config.seed}"
    )

    # node_id -> ip mapping
    node_mapping = getattr(dataset, "mapping", None)

    def idx_to_name(i: int) -> str:
        if node_mapping is None:
            return str(i)
        if isinstance(node_mapping, list):
            if 0 <= i < len(node_mapping):
                return str(node_mapping[i])
            return str(i)
        if isinstance(node_mapping, dict):
            if i in node_mapping:
                return str(node_mapping[i])
            if str(i) in node_mapping:
                return str(node_mapping[str(i)])
            return str(i)
        return str(i)

    try:
        g = dataset.test_graph
        edge_index_t = g.edge_index
        edge_index = edge_index_t.detach().cpu().numpy()

        n_edges = edge_index.shape[1]
        n_scores = int(errors.shape[0])

        if n_edges != n_scores:
            print(f"[WARN] mismatch: edges={n_edges}, scores={n_scores} -> truncate")
        n_use = min(n_edges, n_scores)

        src_ids = edge_index[0, :n_use]
        dst_ids = edge_index[1, :n_use]

        src_ips = [idx_to_name(int(i)) for i in src_ids]
        dst_ips = [idx_to_name(int(i)) for i in dst_ids]

        df_flows = pd.DataFrame(
            {
                "src_ip": src_ips,
                "dst_ip": dst_ips,
                "anomaly_score": errors[:n_use].detach().cpu().numpy(),
            }
        )

        # optional meta
        ts = _maybe_get_edge_attr_numpy(g, "TIMESTAMP", n_use)
        if ts is not None:
            df_flows["ts"] = ts[:n_use]

        dport = _maybe_get_edge_attr_numpy(g, "L4_DST_PORT", n_use)
        if dport is not None:
            df_flows["dst_port"] = dport[:n_use]

        proto = _maybe_get_edge_attr_numpy(g, "PROTOCOL", n_use)
        if proto is not None:
            df_flows["proto"] = proto[:n_use]

        out_dir = "inference_outputs"
        os.makedirs(out_dir, exist_ok=True)

        flow_csv_path = os.path.join(out_dir, f"inference_flows_{run_id}.csv")
        df_flows.to_csv(flow_csv_path, index=False)
        print(f"Per-flow scores saved to: {flow_csv_path}")

        rank_out_dir = os.path.join(out_dir, f"rankings_{run_id}")
        print(f"Running IP ranking logic... Output: {rank_out_dir}")

        # You can switch ranking method here:
        #   - "topk_sum"
        #   - "ppr_anom_pairwise"
        rank_method = getattr(config, "rank_method", "topk_sum")

        if rank_method == "ppr_anom":
            df_rank = rank_ips.rank_ppr_anomaly(
                df_flows,
                threshold=float(threshold) if threshold is not None else None,
                window_sec=3600,      # まずは1時間窓
                alpha=0.15,
                iters=50,
                reverse=False,        # 深さを逆向きに見たければ True も試す
                topn=20,
            )
            os.makedirs(rank_out_dir, exist_ok=True)
            out_path = os.path.join(rank_out_dir, "rank_GLOBAL.csv")
            df_rank.to_csv(out_path, index=False)
            print(f"[OK] PPR rank saved: {out_path}")

            if len(df_rank) > 0:
                run.log({"suspicious_ip_ranking": wandb.Table(dataframe=df_rank)})
        elif rank_method == "infra":
            df_rank = rank_infra(
                df_flows,
                InfraRankConfig(
                    topn=20,
                    hits_iters=50,
                    use_threshold=float(threshold) if threshold is not None else None,  # GraphIDSの閾値を利用
                    min_score=0.0,
                    key_mode="dst_ip",           # "signature" も試せる
                    beacon_min_events=6,
                    max_edges_per_key=5000,
                    include_private_dst=True,    # ここは環境に応じて
                    include_private_src=True,
                    hub_penalty=0.6,
                )
            )

            os.makedirs(rank_out_dir, exist_ok=True)
            out_path = os.path.join(rank_out_dir, "rank_GLOBAL.csv")
            df_rank.to_csv(out_path, index=False)
            print(f"[OK] Infra rank saved: {out_path}")

            if len(df_rank) > 0:
                run.log({"suspicious_ip_ranking": wandb.Table(dataframe=df_rank)})
        else:
            rank_ips.aggregate_from_df(
                df=df_flows,
                ts_col="ts",
                src_ip_col="src_ip",
                dst_ip_col="dst_ip",
                score_col="anomaly_score",
                dst_port_col="dst_port",
                proto_col="proto",
                key_mode="dst_ip",
                window_sec=3600,
                method=rank_method,
                topk=10,
                tau=0.0,
                topn=20,
                out_dir=rank_out_dir,
                track_unique_src=True,
                max_unique_src=10000,
                emit_global=True,
            )

        global_rank_path = os.path.join(rank_out_dir, "rank_GLOBAL.csv")
        if os.path.exists(global_rank_path):
            df_global = pd.read_csv(global_rank_path)
            run.log({"suspicious_ip_ranking": wandb.Table(dataframe=df_global.head(20))})

    except Exception as e:
        print(f"Failed to generate IP ranking: {e}")
        import traceback
        traceback.print_exc()

    run.finish()


if __name__ == "__main__":
    args = Parser().parse_args()

    if args.config is not None:
        config = args.config
    else:
        config = {
            "data_type": args.data_type,
            "dataset": args.dataset,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ae_weight_decay": args.ae_weight_decay,
            "edim_out": args.edim_out,
            "batch_size": args.batch_size,
            "fanout": args.fanout,
            "agg_type": args.agg_type,
            "num_layers": args.num_layers,
            "mask_ratio": args.mask_ratio,
            "patience": args.patience,
            "ae_batch_size": args.ae_batch_size,
            "window_size": args.window_size,
            "step_percent": args.step_percent,
            "ae_embedding_dim": args.ae_embedding_dim,
            "ae_dropout": args.ae_dropout,
            "dropout": args.dropout,
            "positional_encoding": args.positional_encoding,
            "fraction": args.fraction,
        }

    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    run = wandb.init(project="GraphIDS", config=config)

    # Set of parameters that must be passed via command line
    run.config["data_dir"] = args.data_dir
    run.config["checkpoint"] = args.checkpoint
    run.config["reload_dataset"] = args.reload_dataset
    run.config["test"] = args.test
    run.config["save_curve"] = args.save_curve
    run.config["seed"] = args.seed
    run.config["wandb"] = args.wandb

    main(run)
