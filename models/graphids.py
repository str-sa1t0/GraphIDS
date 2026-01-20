# models/graphids.py
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import torch.nn.functional as F


class SAGELayer(MessagePassing):
    def __init__(self, ndim_in, edim_in, edim_out, agg_type="mean", dropout_rate=0.0):
        super().__init__(aggr=agg_type)
        self.fc_neigh = nn.Linear(edim_in, ndim_in)
        # ★ 追加：node feature を注入するための線形層
        self.fc_node = nn.Linear(ndim_in, ndim_in)

        self.fc_edge = nn.Linear(ndim_in * 2, edim_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.agg_type: str = agg_type
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_node.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)

        nn.init.zeros_(self.fc_neigh.bias)
        nn.init.zeros_(self.fc_node.bias)
        nn.init.zeros_(self.fc_edge.bias)

    def message(self, edge_attr):  # type: ignore[override]
        return edge_attr

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")

    # ★ node_attr を受け取れるようにする（後方互換）
    def forward(self, edge_index, edge_attr, edge_couples, num_nodes, node_attr=None):
        """
        node_attr: [num_nodes, ndim_in] (optional)
        """
        # Aggregate edge features to nodes
        node_msg = self.propagate(edge_index, edge_attr=edge_attr, size=(num_nodes, num_nodes))  # [N, edim_in]
        node_msg = self.fc_neigh(node_msg)  # [N, ndim_in]

        # ★ node feature injection
        if node_attr is not None:
            # node_attr is [N, ndim_in]
            node_msg = node_msg + self.fc_node(node_attr)

        node_embeddings = self.relu(node_msg)

        edge_embeddings = self.fc_edge(
            torch.cat(
                [
                    node_embeddings[edge_couples[:, 0]],
                    node_embeddings[edge_couples[:, 1]],
                ],
                dim=1,
            )
        )
        return self.dropout(edge_embeddings)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, embed_dim))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe

    def forward(self, x):
        return x + self.pe[: x.size(1), :]


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        num_layers,
        dropout,
        window_size,
        positional_encoding,
        mask_ratio,
    ):
        super().__init__()
        if positional_encoding == "learnable":
            self.positional_encoder = LearnablePositionalEncoding(embed_dim, window_size)
        elif positional_encoding == "sinusoidal":
            self.positional_encoder = SinusoidalPositionalEncoding(embed_dim, window_size)
        else:
            self.positional_encoder = nn.Identity()

        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.mask_ratio = mask_ratio

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(embed_dim, input_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        for name, param in self.encoder.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.decoder.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, src, padding_mask=None, return_memory: bool = False):
        src = self.input_projection(src)
        src = self.positional_encoder(src)

        src_key_padding_mask = None
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.bool()
            src_key_padding_mask = ~torch.any(padding_mask, dim=-1)

        if self.training and self.mask_ratio > 0:
            seq_len = src.size(1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=src.device, dtype=torch.bool),
                diagonal=1,
            )
            mask = mask & (torch.rand(seq_len, seq_len, device=src.device) < self.mask_ratio)
            attention_mask = mask | mask.T
        else:
            attention_mask = None

        memory = self.encoder(src, mask=attention_mask, src_key_padding_mask=src_key_padding_mask)

        output = self.decoder(
            src,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=src_key_padding_mask,
        )

        output = self.output_projection(output)
        if return_memory:
            return output, memory
        return output


class GraphIDS(nn.Module):
    def __init__(
        self,
        ndim_in,
        edim_in,
        edim_out,
        embed_dim,
        num_heads,
        num_layers,
        window_size=512,
        dropout=0.0,
        ae_dropout=0.1,
        positional_encoding=None,
        agg_type="mean",
        mask_ratio=0.15,
        # ★ SSLの温度（contrastive）
        contrast_tau: float = 0.2,
    ):
        super().__init__()
        self.encoder = SAGELayer(ndim_in, edim_in, edim_out, agg_type, dropout)
        self.transformer = TransformerAutoencoder(
            edim_out,
            embed_dim,
            num_heads,
            num_layers,
            ae_dropout,
            window_size,
            positional_encoding,
            mask_ratio,
        )

        # ★ 追加：Temporal / Contrastive head
        self.temporal_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.contrast_tau = float(contrast_tau)

    def save_checkpoint(self, path, optimizer=None, epoch=0, threshold=None):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "threshold": threshold,
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["threshold"]

    def _pool_memory(self, memory: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is None:
            return memory.mean(dim=1)

        if padding_mask.dim() == 3:
            valid = torch.any(padding_mask.bool(), dim=-1)  # [B, L]
        else:
            valid = padding_mask.bool()

        denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (memory * valid.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def temporal_pred_loss(self, memory: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if memory.size(1) < 2:
            return torch.tensor(0.0, device=memory.device)

        z_t = memory[:, :-1, :]
        z_tp1 = memory[:, 1:, :]
        z_hat = self.temporal_head(z_t)

        if padding_mask is None:
            return F.mse_loss(z_hat, z_tp1)

        if padding_mask.dim() == 3:
            valid = torch.any(padding_mask.bool(), dim=-1)
        else:
            valid = padding_mask.bool()

        valid_pair = valid[:, 1:] & valid[:, :-1]
        if valid_pair.sum() == 0:
            return torch.tensor(0.0, device=memory.device)

        diff = (z_hat - z_tp1).pow(2).mean(dim=-1)
        return (diff * valid_pair.float()).sum() / valid_pair.float().sum().clamp_min(1)

    def contrastive_loss(
        self,
        mem_a: torch.Tensor,
        mem_b: torch.Tensor,
        padding_a: torch.Tensor | None = None,
        padding_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        za = self._pool_memory(mem_a, padding_a)
        zb = self._pool_memory(mem_b, padding_b)

        za = F.normalize(self.proj_head(za), dim=-1)
        zb = F.normalize(self.proj_head(zb), dim=-1)

        logits = (za @ zb.T) / self.contrast_tau
        labels = torch.arange(za.size(0), device=za.device)
        return F.cross_entropy(logits, labels)
