import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataloaders import SequentialDataset, collate_fn


# ============================================================
# Helpers (Self-supervised extensions)
# ============================================================

def _supports_return_memory(transformer) -> bool:
    """
    TransformerAutoencoder.forward が return_memory 引数を受け取れるかを判定。
    例: def forward(self, src, padding_mask=None, return_memory: bool=False)
    """
    try:
        fn = transformer.forward
        code = getattr(fn, "__code__", None)
        if code is None:
            return False
        return "return_memory" in code.co_varnames
    except Exception:
        return False


def _transformer_forward(transformer, x: torch.Tensor, mask: torch.Tensor, return_memory: bool):
    """
    互換性を壊さずに forward するラッパ
      - return_memory 対応なら (outputs, memory)
      - 未対応なら (outputs, None)
    """
    if return_memory and _supports_return_memory(transformer):
        out = transformer(x, mask, return_memory=True)
        # out is expected (outputs, memory)
        return out[0], out[1]
    else:
        outputs = transformer(x, mask)
        return outputs, None


def _valid_token_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: [B, L, D] (0/1) を想定
    return: valid: [B, L] True=valid
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return torch.any(mask, dim=-1)


def _pool_memory(memory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    memory: [B, L, D]
    mask: [B, L, D] (0/1) optional
    return: pooled [B, D]
    """
    if mask is None:
        return memory.mean(dim=1)

    valid = _valid_token_mask(mask)  # [B, L]
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1)  # [B,1]
    pooled = (memory * valid.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


def _feature_mask_aug(x: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    """
    x: [B, L, D]
    feature-wise masking (random elements -> 0)
    """
    if p <= 0:
        return x
    x2 = x.clone()
    m = (torch.rand_like(x2) < p)
    x2[m] = 0.0
    return x2


def _temporal_pred_loss_from_memory(
    memory: torch.Tensor,
    temporal_head: nn.Module,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    memory: [B, L, D]
    Predict memory[t+1] from memory[t] using temporal_head
    """
    if memory.size(1) < 2:
        return torch.tensor(0.0, device=memory.device)

    z_t = memory[:, :-1, :]
    z_tp1 = memory[:, 1:, :]

    z_hat = temporal_head(z_t)  # [B, L-1, D]

    # token-wise MSE
    per_tok = (z_hat - z_tp1).pow(2).mean(dim=-1)  # [B, L-1]

    if mask is None:
        return per_tok.mean()

    valid = _valid_token_mask(mask)  # [B, L]
    valid_pair = valid[:, :-1] & valid[:, 1:]  # [B, L-1]
    denom = valid_pair.float().sum().clamp_min(1)
    return (per_tok * valid_pair.float()).sum() / denom


def _info_nce(za: torch.Tensor, zb: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """
    za, zb: [B, D]
    """
    za = F.normalize(za, dim=-1)
    zb = F.normalize(zb, dim=-1)
    logits = (za @ zb.T) / tau  # [B,B]
    labels = torch.arange(za.size(0), device=za.device)
    return F.cross_entropy(logits, labels)


def _contrastive_loss_from_memory(
    mem_a: torch.Tensor,
    mem_b: torch.Tensor,
    proj_head: nn.Module,
    mask_a: Optional[torch.Tensor] = None,
    mask_b: Optional[torch.Tensor] = None,
    tau: float = 0.2,
) -> torch.Tensor:
    """
    mem_a, mem_b: [B, L, D]
    """
    za = _pool_memory(mem_a, mask_a)  # [B, D]
    zb = _pool_memory(mem_b, mask_b)  # [B, D]
    za = proj_head(za)
    zb = proj_head(zb)
    return _info_nce(za, zb, tau=tau)


def _recon_token_errors(outputs: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    outputs, batch: [B, L, D]
    mask: [B, L, D]
    return: flattened valid token errors (like original calculate_errors)
    """
    squared_errors = ((outputs - batch) ** 2) * mask
    valid = _valid_token_mask(mask)  # [B, L]
    valid_counts = torch.sum(mask, dim=-1)  # [B, L]
    mean_errors = torch.zeros_like(valid_counts, dtype=torch.float32)
    if valid.any():
        mean_errors[valid] = torch.sum(squared_errors, dim=-1)[valid] / torch.clamp(valid_counts[valid], min=1)
    mean_errors = torch.nan_to_num(mean_errors, nan=0.0, posinf=1e6, neginf=-1e6)
    return mean_errors[valid]  # flatten


def _temporal_token_errors(
    memory: torch.Tensor,
    temporal_head: nn.Module,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    memory: [B, L, D]
    returns per-token error aligned to positions [B, L] then flattened valid.
    error is assigned to position t+1 (first token = 0).
    """
    B, L, D = memory.shape
    if L < 2:
        # no temporal errors
        if mask is None:
            return torch.zeros((B * L,), device=memory.device)
        valid = _valid_token_mask(mask)
        return torch.zeros((valid.sum().item(),), device=memory.device)

    z_t = memory[:, :-1, :]
    z_tp1 = memory[:, 1:, :]
    z_hat = temporal_head(z_t)  # [B, L-1, D]
    per_tok = (z_hat - z_tp1).pow(2).mean(dim=-1)  # [B, L-1]

    # align to [B,L] (position 0 has no prediction)
    aligned = torch.zeros((B, L), device=memory.device, dtype=per_tok.dtype)
    aligned[:, 1:] = per_tok

    if mask is None:
        return aligned.reshape(-1)

    valid = _valid_token_mask(mask)  # [B,L]
    return aligned[valid]


# ============================================================
# Training / Eval
# ============================================================

def train_encoder(
    model,
    window_size,
    step_percent,
    ae_batch_size,
    train_loader,
    val_loader,
    test_loader,
    start_epoch,
    num_epochs,
    optimizer,
    run,
    patience,
    checkpoint,
    device="cuda",
):
    """
    Extended self-supervised training:
      - Reconstruction loss (original)
      - + Temporal prediction loss (optional, label-free)
      - + Contrastive loss between augmented views (optional, label-free)

    Enable/weight via wandb config (optional):
      run.config.w_temporal (default 0.5)
      run.config.w_contrast (default 0.5)
      run.config.contrast_feat_mask_p (default 0.15)
      run.config.contrast_tau (default 0.2)
      run.config.score_pred_weight (default = w_temporal)
      run.config.threshold_method ("supervised" or "unsupervised", default "supervised")
      run.config.threshold_multiplier (default 10.0)  # only for unsupervised
    """
    best_pr_auc = 0.0
    cnt_wait = 0

    # recon criterion (same as original)
    criterion = nn.MSELoss(reduction="none")

    # ---- config values (safe defaults) ----
    w_temporal = float(getattr(run.config, "w_temporal", 0.5))
    w_contrast = float(getattr(run.config, "w_contrast", 0.5))
    feat_mask_p = float(getattr(run.config, "contrast_feat_mask_p", 0.15))
    contrast_tau = float(getattr(run.config, "contrast_tau", 0.2))

    threshold_method = str(getattr(run.config, "threshold_method", "supervised"))
    threshold_multiplier = float(getattr(run.config, "threshold_multiplier", 10.0))

    # for scoring (val/test)
    score_pred_weight = float(getattr(run.config, "score_pred_weight", w_temporal))

    for epoch in (pbar := tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs")):
        total_train_loss = 0.0
        model.train()

        for batch in train_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            # (1) edge embeddings via GNN encoder
            train_emb = model.encoder(
                batch.edge_index,
                batch.edge_attr,
                batch.batch_edge_couples,
                batch.num_nodes,
            )

            # (2) sequential windows for transformer
            ae_train_loader = DataLoader(
                SequentialDataset(
                    train_emb,
                    window=window_size,
                    step=int(window_size * step_percent),
                    device=device,
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )

            accumulated_loss = torch.tensor(0.0, device=device)
            seq_count = 0

            for ae_batch, mask in ae_train_loader:
                # --------------------------
                # forward (recon + memory)
                # --------------------------
                outputs, memory = _transformer_forward(
                    model.transformer, ae_batch, mask, return_memory=True
                )

                # (A) Reconstruction loss (original)
                loss_recon = criterion(outputs, ae_batch)
                loss_recon = torch.sum(loss_recon * mask) / torch.sum(mask)

                loss_total = loss_recon

                # (B) Temporal prediction loss (optional)
                loss_temp = torch.tensor(0.0, device=device)
                if (memory is not None) and hasattr(model, "temporal_head") and (w_temporal > 0):
                    loss_temp = _temporal_pred_loss_from_memory(
                        memory, model.temporal_head, mask
                    )
                    loss_total = loss_total + w_temporal * loss_temp

                # (C) Contrastive loss between two views (optional)
                loss_con = torch.tensor(0.0, device=device)
                if (memory is not None) and hasattr(model, "proj_head") and (w_contrast > 0):
                    # View-B augmentation on embedding sequence
                    ae_batch_b = _feature_mask_aug(ae_batch, p=feat_mask_p)
                    _, mem_b = _transformer_forward(
                        model.transformer, ae_batch_b, mask, return_memory=True
                    )
                    if mem_b is not None:
                        loss_con = _contrastive_loss_from_memory(
                            memory, mem_b, model.proj_head, mask, mask, tau=contrast_tau
                        )
                        loss_total = loss_total + w_contrast * loss_con

                accumulated_loss += loss_total
                seq_count += 1

            # Backprop once per GNN mini-batch (same policy as original)
            if seq_count > 0:
                loss = accumulated_loss / seq_count
                total_train_loss += float(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_train_loss /= max(len(train_loader), 1)

        # ---- validation ----
        val_loss, val_errors, val_labels = validate(
            model,
            val_loader,
            ae_batch_size,
            window_size,
            device,
            score_pred_weight=score_pred_weight,
        )
        val_pr_auc = average_precision_score(val_labels.cpu(), val_errors.cpu())

        # threshold selection
        if threshold_method == "unsupervised":
            threshold = find_threshold(
                val_errors, labels=None, method="unsupervised", multiplier=threshold_multiplier
            )
        else:
            threshold = find_threshold(
                val_errors, labels=val_labels, method="supervised"
            )

        # debug test during training (same as original)
        test_f1, test_pr_auc, _, _, _ = test(
            model,
            test_loader,
            ae_batch_size,
            window_size,
            device,
            threshold=threshold,
            score_pred_weight=score_pred_weight,
        )

        # Save checkpoint if equal/better val PR-AUC
        if val_pr_auc >= best_pr_auc:
            model.save_checkpoint(
                checkpoint,
                optimizer=optimizer,
                epoch=epoch,
                threshold=threshold,
            )

        # Early stopping on PR-AUC
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            cnt_wait = 0
        else:
            cnt_wait += 1
            if cnt_wait >= patience:
                print("Early stopping!")
                break

        pbar.set_postfix(
            {
                "train_loss": total_train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "test_f1": test_f1,
                "test_pr_auc": test_pr_auc,
            }
        )
        run.log(
            {
                "train_loss": total_train_loss,
                "val_loss": val_loss,
                "val_pr_auc": val_pr_auc,
                "test_f1": test_f1,
                "test_pr_auc": test_pr_auc,
                "w_temporal": w_temporal,
                "w_contrast": w_contrast,
                "contrast_feat_mask_p": feat_mask_p,
                "contrast_tau": contrast_tau,
                "score_pred_weight": score_pred_weight,
                "threshold_method": threshold_method,
            }
        )

    chk = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    return model, chk["threshold"]


def find_threshold(errors, labels=None, method="unsupervised", multiplier=10.0):
    if method == "unsupervised":
        median = errors.median()
        mad = (errors - median).abs().median() * 1.4826  # normal consistency factor
        best_threshold = median + multiplier * mad
    elif method == "supervised" and labels is not None:
        best_f1 = 0.0
        best_threshold = errors.mean()
        for threshold in torch.linspace(errors.min(), errors.max(), steps=500):
            val_pred = (errors > threshold).int()
            f1 = f1_score(
                labels.cpu(), val_pred.cpu(), average="macro", zero_division=0
            )
            if f1 > best_f1:
                best_threshold = threshold.item()
                best_f1 = f1
    else:
        raise ValueError(
            "Invalid method for threshold finding. Use 'unsupervised' or 'supervised' with labels."
        )
    return best_threshold


def validate(
    model,
    val_loader,
    ae_batch_size,
    window_size,
    device,
    score_pred_weight: float = 0.0,
):
    """
    Returns:
      total_val_loss: scalar
      errors: token-wise flattened anomaly scores
      labels: edge labels (from LinkNeighborLoader)
    """
    criterion = nn.MSELoss(reduction="none")
    model.eval()

    errors = []
    labels = []
    total_val_loss = 0.0

    with torch.inference_mode():
        for batch in val_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            val_emb = model.encoder(
                batch.edge_index,
                batch.edge_attr,
                batch.batch_edge_couples,
                batch.num_nodes,
            )

            labels.append(batch.edge_label.cpu())

            ae_val_loader = DataLoader(
                SequentialDataset(
                    val_emb,
                    window=window_size,
                    step=window_size,  # keep original policy
                    device=device,
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )

            accumulated_loss = torch.tensor(0.0, device=device)
            seq_count = 0

            for ae_batch, mask in ae_val_loader:
                outputs, memory = _transformer_forward(
                    model.transformer, ae_batch, mask, return_memory=True
                )

                # recon loss
                loss = criterion(outputs, ae_batch)
                loss = torch.sum(loss * mask) / torch.sum(mask)
                accumulated_loss += loss
                seq_count += 1

                # recon error per token (flattened)
                recon_err = _recon_token_errors(outputs, ae_batch, mask)

                if (
                    (score_pred_weight > 0.0)
                    and (memory is not None)
                    and hasattr(model, "temporal_head")
                ):
                    pred_err = _temporal_token_errors(memory, model.temporal_head, mask)
                    # align lengths by truncation (safe)
                    n_use = min(recon_err.numel(), pred_err.numel())
                    recon_err = recon_err[:n_use]
                    pred_err = pred_err[:n_use]
                    combined = recon_err + score_pred_weight * pred_err
                    errors.append(combined.cpu())
                else:
                    errors.append(recon_err.cpu())

            if seq_count > 0:
                total_val_loss += float((accumulated_loss / seq_count).item())

    total_val_loss /= max(len(val_loader), 1)
    labels = torch.cat(labels)
    errors = torch.cat(errors) if len(errors) > 0 else torch.tensor([])
    return total_val_loss, errors, labels


def test(
    model,
    test_loader,
    ae_batch_size,
    window_size,
    device,
    threshold,
    score_pred_weight: float = 0.0,
):
    """
    Returns:
      f1, pr_auc, errors, labels, prediction_time
    """
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    model.eval()
    errors = []
    labels = []

    with torch.inference_mode():
        for batch in test_loader:
            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            test_emb = model.encoder(
                batch.edge_index,
                batch.edge_attr,
                batch.batch_edge_couples,
                batch.num_nodes,
            )

            labels.append(batch.edge_label.cpu())

            ae_test_loader = DataLoader(
                SequentialDataset(
                    test_emb, window=window_size, step=window_size, device=device
                ),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )

            for ae_batch, mask in ae_test_loader:
                outputs = model.transformer(ae_batch, mask)

                squared_errors = ((outputs - ae_batch) ** 2) * mask
                valid_mask = mask.sum(dim=-1) > 0
                valid_counts = torch.sum(mask, dim=-1)
                mean_errors = torch.zeros_like(valid_counts, dtype=torch.float32)
                if valid_mask.any():
                    mean_errors[valid_mask] = torch.sum(squared_errors, dim=-1)[
                        valid_mask
                    ] / torch.clamp(valid_counts[valid_mask], min=1)
                mean_errors = torch.nan_to_num(
                    mean_errors, nan=0.0, posinf=1e6, neginf=-1e6
                )
                batch_errors = mean_errors[valid_mask]

                errors.append(batch_errors.cpu())

    labels = torch.cat(labels)
    errors = torch.cat(errors) if len(errors) > 0 else torch.tensor([])

    if errors.numel() > 0:
        if threshold is not None:
            test_pred = (errors > threshold).int()
        else:
            print("No threshold provided, using mean of errors for prediction.")
            test_pred = (errors > errors.mean()).int()
    else:
        test_pred = torch.zeros_like(labels)

    if device == "cuda":
        torch.cuda.synchronize()
    prediction_time = time.perf_counter() - start_time

    f1 = f1_score(labels, test_pred, average="macro", zero_division=0)
    pr_auc = average_precision_score(labels, errors) if errors.numel() > 0 else 0.0

    return f1, pr_auc, errors, labels, prediction_time
