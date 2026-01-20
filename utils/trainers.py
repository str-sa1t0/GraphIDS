# utils/trainers.py
from __future__ import annotations

import time
from typing import Tuple, Optional

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataloaders import SequentialDataset, collate_fn


# ============================================================
# Helpers (compatibility safe)
# ============================================================
def _encode_edges(model, batch) -> torch.Tensor:
    """
    Try to pass node_attr=batch.x if encoder supports it.
    """
    batch.batch_edge_couples = batch.edge_label_index.t()
    try:
        # new signature: encoder(edge_index, edge_attr, edge_couples, num_nodes, node_attr=...)
        return model.encoder(
            batch.edge_index,
            batch.edge_attr,
            batch.batch_edge_couples,
            batch.num_nodes,
            node_attr=batch.x,
        )
    except TypeError:
        # old signature
        return model.encoder(
            batch.edge_index,
            batch.edge_attr,
            batch.batch_edge_couples,
            batch.num_nodes,
        )


def _transformer_forward(model, ae_batch: torch.Tensor, mask: torch.Tensor, return_memory: bool):
    """
    Compatible transformer forward:
      - new: transformer(src, padding_mask, return_memory=True) -> (out, mem)
      - old: transformer(src, padding_mask) -> out
    """
    if not return_memory:
        try:
            return model.transformer(ae_batch, mask, return_memory=False)
        except TypeError:
            return model.transformer(ae_batch, mask)

    # return memory
    try:
        out, mem = model.transformer(ae_batch, mask, return_memory=True)
        return out, mem
    except TypeError:
        out = model.transformer(ae_batch, mask)
        return out, None


# ============================================================
# Threshold
# ============================================================
def find_threshold(errors: torch.Tensor, labels: Optional[torch.Tensor] = None, method="supervised", multiplier=10.0):
    """
    errors: [N]
    labels: [N] (optional)
    """
    if errors.numel() == 0:
        return float("inf")

    if method == "unsupervised":
        median = errors.median()
        mad = (errors - median).abs().median() * 1.4826
        thr = median + multiplier * mad
        return float(thr.item())

    if method == "supervised" and labels is not None:
        best_f1 = 0.0
        best_thr = float(errors.mean().item())
        # robust linspace range
        e_min = float(errors.min().item())
        e_max = float(errors.max().item())
        if abs(e_max - e_min) < 1e-12:
            return best_thr

        for thr in torch.linspace(errors.min(), errors.max(), steps=500, device=errors.device):
            pred = (errors > thr).int()
            f1 = f1_score(labels.cpu(), pred.cpu(), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr.item())
        return best_thr

    raise ValueError("method must be 'unsupervised' or 'supervised' with labels")


# ============================================================
# Error calculation
# ============================================================
def calculate_errors(outputs: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    outputs, batch: [B, L, D]
    mask: [B, L, D] (bool)
    Return per-token MSE (flattened for valid tokens) so that length matches labels (E)
    in val/test when step=window_size (no overlap).
    """
    # squared error per element
    sq = (outputs - batch).pow(2) * mask.float()  # [B,L,D]

    # valid token mask: token has any valid dim
    valid_token = mask.any(dim=-1)  # [B,L]
    denom = mask.float().sum(dim=-1).clamp_min(1.0)  # [B,L]

    token_mse = sq.sum(dim=-1) / denom  # [B,L]
    token_mse = torch.nan_to_num(token_mse, nan=0.0, posinf=1e6, neginf=-1e6)

    # flatten only valid tokens
    return token_mse[valid_token]


# ============================================================
# Validate / Test
# ============================================================
@torch.no_grad()
def validate(model, val_loader, ae_batch_size, window_size, device):
    criterion = nn.MSELoss(reduction="none")
    model.eval()

    errors = []
    labels = []
    total_val_loss = 0.0

    for batch in val_loader:
        batch = batch.to(device)

        val_emb = _encode_edges(model, batch)
        labels.append(batch.edge_label.cpu())

        ae_val_loader = DataLoader(
            SequentialDataset(val_emb, window=window_size, step=window_size, device=device),
            batch_size=ae_batch_size,
            collate_fn=collate_fn,
        )

        accum_loss = 0.0
        seq_count = 0

        for ae_batch, mask in ae_val_loader:
            out = _transformer_forward(model, ae_batch, mask, return_memory=False)
            loss = criterion(out, ae_batch)
            loss = torch.sum(loss * mask.float()) / torch.sum(mask.float()).clamp_min(1.0)

            accum_loss += float(loss.item())
            seq_count += 1

            batch_err = calculate_errors(out, ae_batch, mask)
            errors.append(batch_err.cpu())

        if seq_count > 0:
            total_val_loss += accum_loss / seq_count

    total_val_loss = total_val_loss / max(1, len(val_loader))
    labels = torch.cat(labels) if len(labels) > 0 else torch.empty(0, dtype=torch.long)
    errors = torch.cat(errors) if len(errors) > 0 else torch.empty(0, dtype=torch.float32)

    return total_val_loss, errors, labels


def test(model, test_loader, ae_batch_size, window_size, device, threshold):
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    model.eval()

    errors = []
    labels = []

    with torch.inference_mode():
        for batch in test_loader:
            batch = batch.to(device)

            test_emb = _encode_edges(model, batch)
            labels.append(batch.edge_label.cpu())

            ae_test_loader = DataLoader(
                SequentialDataset(test_emb, window=window_size, step=window_size, device=device),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )

            for ae_batch, mask in ae_test_loader:
                out = _transformer_forward(model, ae_batch, mask, return_memory=False)
                batch_err = calculate_errors(out, ae_batch, mask)
                errors.append(batch_err.cpu())

    labels = torch.cat(labels) if len(labels) > 0 else torch.empty(0, dtype=torch.long)
    errors = torch.cat(errors) if len(errors) > 0 else torch.empty(0, dtype=torch.float32)

    if threshold is None:
        thr = float(errors.mean().item()) if errors.numel() else 0.0
    else:
        thr = float(threshold)

    pred = (errors > thr).int() if errors.numel() else torch.empty(0, dtype=torch.int32)

    if device == "cuda":
        torch.cuda.synchronize()
    prediction_time = time.perf_counter() - start_time

    # metrics
    if labels.numel() and errors.numel() and labels.numel() == errors.numel():
        f1 = f1_score(labels, pred, average="macro", zero_division=0)
        pr_auc = average_precision_score(labels, errors)
    else:
        # safe fallback (shouldn't happen in non-overlap val/test)
        f1 = 0.0
        pr_auc = 0.0

    return f1, pr_auc, errors, labels, prediction_time


# ============================================================
# Train
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
    Train encoder + transformer as reconstruction AE (GraphIDS base).
    Optionally adds:
      - temporal pred loss if model.temporal_pred_loss exists and model.temporal_head exists
      - contrastive loss if model.contrastive_loss exists and model.proj_head exists
    """
    best_pr_auc = 0.0
    cnt_wait = 0

    recon_criterion = nn.MSELoss(reduction="none")

    # Optional weights from config
    cfg = getattr(run, "config", {})
    use_temporal = bool(getattr(cfg, "use_temporal_loss", False))
    use_contrastive = bool(getattr(cfg, "use_contrastive_loss", False))
    lambda_temporal = float(getattr(cfg, "lambda_temporal", 0.1))
    lambda_contrast = float(getattr(cfg, "lambda_contrastive", 0.05))
    threshold_method = getattr(cfg, "threshold_method", "supervised")  # or "unsupervised"
    unsup_multiplier = float(getattr(cfg, "unsup_multiplier", 10.0))

    # step size
    step = int(max(1, int(window_size * float(step_percent))))

    for epoch in (pbar := tqdm(range(start_epoch + 1, num_epochs + 1), desc="Epochs")):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            train_emb = _encode_edges(model, batch)

            ae_train_loader = DataLoader(
                SequentialDataset(train_emb, window=window_size, step=step, device=device),
                batch_size=ae_batch_size,
                collate_fn=collate_fn,
            )

            accum = torch.tensor(0.0, device=device)
            seq_count = 0

            for ae_batch, mask in ae_train_loader:
                # forward (need memory if optional losses enabled)
                need_mem = use_temporal or use_contrastive
                if need_mem:
                    out, mem = _transformer_forward(model, ae_batch, mask, return_memory=True)
                else:
                    out = _transformer_forward(model, ae_batch, mask, return_memory=False)
                    mem = None

                # reconstruction loss
                loss_recon = recon_criterion(out, ae_batch)
                loss_recon = torch.sum(loss_recon * mask.float()) / torch.sum(mask.float()).clamp_min(1.0)

                loss = loss_recon

                # temporal loss
                if use_temporal and mem is not None and hasattr(model, "temporal_pred_loss"):
                    try:
                        loss_t = model.temporal_pred_loss(mem, mask)
                        loss = loss + lambda_temporal * loss_t
                    except Exception:
                        pass

                # contrastive loss (two stochastic views via repeated forward in training)
                if use_contrastive and hasattr(model, "contrastive_loss"):
                    try:
                        # create two views by calling transformer twice (masking/attention randomness)
                        _, mem_a = _transformer_forward(model, ae_batch, mask, return_memory=True)
                        _, mem_b = _transformer_forward(model, ae_batch, mask, return_memory=True)
                        if mem_a is not None and mem_b is not None:
                            loss_c = model.contrastive_loss(mem_a, mem_b, mask, mask)
                            loss = loss + lambda_contrast * loss_c
                    except Exception:
                        pass

                accum = accum + loss
                seq_count += 1

            if seq_count > 0:
                loss_batch = accum / seq_count
                total_train_loss += float(loss_batch.item())

                loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_train_loss = total_train_loss / max(1, len(train_loader))

        # validation
        val_loss, val_errors, val_labels = validate(model, val_loader, ae_batch_size, window_size, device)

        # PR-AUC
        if val_errors.numel() and val_labels.numel() and val_errors.numel() == val_labels.numel():
            val_pr_auc = average_precision_score(val_labels.cpu(), val_errors.cpu())
        else:
            val_pr_auc = 0.0

        # threshold
        if threshold_method == "unsupervised":
            threshold = find_threshold(val_errors, labels=None, method="unsupervised", multiplier=unsup_multiplier)
        else:
            threshold = find_threshold(val_errors, val_labels, method="supervised")

        # debug test (safe)
        try:
            test_f1, test_pr_auc, _, _, _ = test(model, test_loader, ae_batch_size, window_size, device, threshold)
        except Exception:
            test_f1, test_pr_auc = 0.0, 0.0

        # save best checkpoint (>= keeps last best)
        if val_pr_auc >= best_pr_auc:
            model.save_checkpoint(checkpoint, optimizer=optimizer, epoch=epoch, threshold=threshold)

        # early stopping logic
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

        if hasattr(run, "log"):
            run.log(
                {
                    "train_loss": total_train_loss,
                    "val_loss": val_loss,
                    "val_pr_auc": val_pr_auc,
                    "test_f1": test_f1,
                    "test_pr_auc": test_pr_auc,
                    "epoch": epoch,
                }
            )

    # load best
    chk = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(chk["model_state_dict"])
    return model, chk.get("threshold", None)
