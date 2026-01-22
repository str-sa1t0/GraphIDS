# rank_ips.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import ipaddress



# ============================================================
# Utilities
# ============================================================
def _ensure_datetime_series(s: pd.Series) -> pd.Series:
    """
    Convert ts column to pandas datetime.

    Expected:
      - already datetime64
      - int/float epoch time (ns/us/ms/s)

    Your NetFlowDataset currently stores:
      data.TIMESTAMP = FLOW_START_MILLISECONDS_RAW * 1_000_000  (ns)
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    # try numeric epoch -> datetime
    if pd.api.types.is_numeric_dtype(s):
        x = s.fillna(0).astype(np.int64)

        # heuristic based on magnitude
        vmax = int(x.max()) if len(x) > 0 else 0

        # ns epoch ~ 1e18, us epoch ~ 1e15, ms epoch ~ 1e12, sec epoch ~ 1e9
        if vmax >= 10**17:
            unit = "ns"
        elif vmax >= 10**14:
            unit = "us"
        elif vmax >= 10**11:
            unit = "ms"
        else:
            unit = "s"

        return pd.to_datetime(x, unit=unit, errors="coerce")

    # fallback string parse
    return pd.to_datetime(s, errors="coerce")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _maybe_filter_tau(df: pd.DataFrame, score_col: str, tau: float) -> pd.DataFrame:
    if tau is None:
        return df
    try:
        tau = float(tau)
    except Exception:
        return df
    if tau <= 0:
        return df
    return df[df[score_col] >= tau]


def _topk_sum_group(
    g: pd.DataFrame,
    score_col: str,
    topk: int,
    src_ip_col: Optional[str] = None,
    track_unique_src: bool = False,
    max_unique_src: int = 10000,
) -> Dict[str, Any]:
    """
    Score a group (one key e.g., one dst_ip) by summing top-k flow anomaly scores.

    Returns:
      score, n_flows, topk, unique_src(optional)
    """
    if g.empty:
        out = {"score": 0.0, "n_flows": 0}
        if track_unique_src:
            out["unique_src"] = 0
        return out

    # top-k by score
    if topk is None or topk <= 0:
        topk = len(g)

    g2 = g.nlargest(min(topk, len(g)), columns=score_col)

    score = float(g2[score_col].sum())
    out = {
        "score": score,
        "n_flows": int(len(g)),
        "topk_used": int(len(g2)),
    }

    if track_unique_src and src_ip_col is not None and src_ip_col in g2.columns:
        # cap to avoid memory blow-up
        if len(g2) > max_unique_src:
            out["unique_src"] = int(g2[src_ip_col].iloc[:max_unique_src].nunique())
        else:
            out["unique_src"] = int(g2[src_ip_col].nunique())

    return out


# ============================================================
# PPR + anomaly + pairwise (depth proxy via k-core)
# ============================================================
def _build_node_index_from_ips(all_ips: List[str]) -> Tuple[Dict[str, int], List[str]]:
    uniq = sorted(set(map(str, all_ips)))
    ip2i = {ip: i for i, ip in enumerate(uniq)}
    return ip2i, uniq


def _ppr_power_iteration(
    edges: List[Tuple[int, int]],
    num_nodes: int,
    seeds: List[int],
    alpha: float = 0.15,
    iters: int = 30,
) -> np.ndarray:
    """
    Simple undirected PPR via power iteration.
    """
    nbrs = [[] for _ in range(num_nodes)]
    for u, v in edges:
        if u == v:
            continue
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            nbrs[u].append(v)
            nbrs[v].append(u)

    deg = np.array([len(n) for n in nbrs], dtype=np.float32)
    deg[deg == 0] = 1.0

    r = np.zeros(num_nodes, dtype=np.float32)
    p0 = np.zeros(num_nodes, dtype=np.float32)

    if len(seeds) == 0:
        return r

    p0[seeds] = 1.0 / float(len(seeds))
    r[:] = p0

    for _ in range(iters):
        new_r = alpha * p0
        for u in range(num_nodes):
            ru = r[u] / deg[u]
            if ru == 0:
                continue
            for v in nbrs[u]:
                new_r[v] += (1.0 - alpha) * ru
        r = new_r

    return r


def _compute_ip_struct_features(df: pd.DataFrame, src_ip_col: str, dst_ip_col: str) -> pd.DataFrame:
    """
    Compute per-IP structural features using edges from df.
      deg_in/out, uniq_in/out, kcore (optional)
    """
    src = df[src_ip_col].astype(str).to_numpy()
    dst = df[dst_ip_col].astype(str).to_numpy()

    all_ips = np.unique(np.concatenate([src, dst], axis=0))
    ip2i = {ip: i for i, ip in enumerate(all_ips)}
    n = len(all_ips)

    deg_out = np.zeros(n, dtype=np.float32)
    deg_in = np.zeros(n, dtype=np.float32)
    for s, d in zip(src, dst):
        deg_out[ip2i[s]] += 1
        deg_in[ip2i[d]] += 1

    seen_out = [set() for _ in range(n)]
    seen_in = [set() for _ in range(n)]
    for s, d in zip(src, dst):
        si = ip2i[s]
        di = ip2i[d]
        seen_out[si].add(di)
        seen_in[di].add(si)

    uniq_out = np.array([float(len(seen_out[i])) for i in range(n)], dtype=np.float32)
    uniq_in = np.array([float(len(seen_in[i])) for i in range(n)], dtype=np.float32)

    # k-core (undirected)
    kcore = np.zeros(n, dtype=np.float32)
    try:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for s, d in zip(src, dst):
            G.add_edge(ip2i[s], ip2i[d])
        core = nx.core_number(G)
        for k, v in core.items():
            if 0 <= int(k) < n:
                kcore[int(k)] = float(v)
    except Exception:
        pass

    # log1p normalization
    deg_in = np.log1p(deg_in)
    deg_out = np.log1p(deg_out)
    uniq_in = np.log1p(uniq_in)
    uniq_out = np.log1p(uniq_out)
    kcore = np.log1p(kcore)

    feat = pd.DataFrame(
        {
            "ip": all_ips,
            "deg_in": deg_in,
            "deg_out": deg_out,
            "uniq_in": uniq_in,
            "uniq_out": uniq_out,
            "kcore": kcore,
        }
    )
    return feat


def _learn_pairwise_weights(
    df_feat: pd.DataFrame,
    target_col: str = "kcore",
    lr: float = 0.05,
    steps: int = 400,
    margin: float = 0.2,
    seed: int = 0,
) -> Tuple[Dict[str, float], float]:
    """
    Learn linear weights to rank larger target_col higher (pairwise hinge).

    Use as a *refinement* head over base_score (PPR×anom).
    """
    rng = np.random.default_rng(seed)
    feat_cols = [c for c in df_feat.columns if c not in ["ip", target_col]]

    X = df_feat[feat_cols].values.astype(np.float32)
    y = df_feat[target_col].values.astype(np.float32)

    if len(X) == 0 or np.std(y) < 1e-6:
        w = np.zeros((X.shape[1],), dtype=np.float32)
        b = 0.0
        return {feat_cols[i]: float(w[i]) for i in range(len(feat_cols))}, float(b)

    # standardize
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - X_mean) / X_std

    w = np.zeros((Xn.shape[1],), dtype=np.float32)
    idx = np.arange(len(y))

    for _ in range(steps):
        i = rng.choice(idx, size=min(512, len(idx)), replace=True)
        j = rng.choice(idx, size=min(512, len(idx)), replace=True)

        mask = y[i] > y[j]
        if mask.sum() == 0:
            continue

        ii = i[mask]
        jj = j[mask]

        si = Xn[ii] @ w
        sj = Xn[jj] @ w

        diff = si - sj
        vio = (margin - diff) > 0
        if vio.sum() == 0:
            continue

        grad = -(Xn[ii[vio]] - Xn[jj[vio]]).mean(axis=0)
        w -= lr * grad

    # unnormalize back to original space
    w_real = w / X_std[0]
    b_real = -float((X_mean[0] / X_std[0]) @ w)

    weights = {feat_cols[i]: float(w_real[i]) for i in range(len(feat_cols))}
    return weights, float(b_real)


def _rank_ppr_anom_pairwise(
    df: pd.DataFrame,
    src_ip_col: str,
    dst_ip_col: str,
    score_col: str,
    key_mode: str = "dst_ip",
    topn: int = 20,
    # PPR params
    ppr_alpha: float = 0.15,
    ppr_iters: int = 30,
    ppr_beta: float = 3.0,
    seed_topk: int = 2000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Global ranking:
      base_score = anomaly_sum(key) * (1 + beta * PPR(key))
      final_score = base_score * (1 + sigmoid(linear(struct)))
    """
    # --- node index
    all_ips = pd.concat([df[src_ip_col], df[dst_ip_col]]).astype(str).tolist()
    ip2i, i2ip = _build_node_index_from_ips(all_ips)
    num_nodes = len(i2ip)

    # --- edges (undirected)
    edges: List[Tuple[int, int]] = []
    for s, d in zip(df[src_ip_col].astype(str).tolist(), df[dst_ip_col].astype(str).tolist()):
        edges.append((ip2i[s], ip2i[d]))

    # --- seeds: endpoints of top anomalous flows
    df_seed = df.nlargest(min(seed_topk, len(df)), columns=score_col)
    seed_ips = set(df_seed[src_ip_col].astype(str).tolist() + df_seed[dst_ip_col].astype(str).tolist())
    seeds = [ip2i[ip] for ip in seed_ips if ip in ip2i]

    ppr = _ppr_power_iteration(edges, num_nodes, seeds, alpha=ppr_alpha, iters=ppr_iters)
    if ppr.max() > 0:
        ppr = ppr / (ppr.max() + 1e-12)

    # --- anomaly aggregation per key
    if key_mode == "dst_ip":
        keys = df[dst_ip_col].astype(str)
        gdf = df.copy()
        gdf["__key"] = keys
    elif key_mode == "src_ip":
        keys = df[src_ip_col].astype(str)
        gdf = df.copy()
        gdf["__key"] = keys
    else:
        # both
        dfa = df.copy()
        dfb = df.copy()
        dfa["__key"] = dfa[src_ip_col].astype(str)
        dfb["__key"] = dfb[dst_ip_col].astype(str)
        gdf = pd.concat([dfa, dfb], axis=0, ignore_index=True)

    anom_sum = gdf.groupby("__key", sort=False)[score_col].sum().reset_index()
    anom_sum.columns = ["ip", "anomaly_sum"]

    # --- struct features from entire df
    feat_struct = _compute_ip_struct_features(df, src_ip_col, dst_ip_col)

    # --- join
    anom_sum["ppr"] = anom_sum["ip"].map(lambda x: float(ppr[ip2i.get(str(x), 0)]) if str(x) in ip2i else 0.0)
    feat = anom_sum.merge(feat_struct, on="ip", how="left").fillna(0)

    # --- base
    feat["base_score"] = feat["anomaly_sum"] * (1.0 + ppr_beta * feat["ppr"])

    # --- pairwise refinement (depth proxy: kcore)
    learn_cols = ["ip", "anomaly_sum", "ppr", "deg_in", "deg_out", "uniq_in", "uniq_out", "kcore"]
    weights, bias = _learn_pairwise_weights(feat[learn_cols].copy(), target_col="kcore")

    def lin_score(row: pd.Series) -> float:
        s = bias
        for k, w in weights.items():
            s += w * float(row.get(k, 0.0))
        return float(s)

    feat["learned_score"] = feat.apply(lin_score, axis=1)
    feat["final_score"] = feat["base_score"] * (1.0 + 1.0 / (1.0 + np.exp(-feat["learned_score"])))

    feat = feat.sort_values("final_score", ascending=False).head(topn).reset_index(drop=True)

    meta = {
        "weights": weights,
        "bias": bias,
        "ppr_alpha": ppr_alpha,
        "ppr_iters": ppr_iters,
        "ppr_beta": ppr_beta,
        "seed_topk": seed_topk,
    }
    return feat, meta


# ============================================================
# Public API: aggregate_from_df
# ============================================================
def aggregate_from_df(
    df: pd.DataFrame,
    ts_col: str = "ts",
    src_ip_col: str = "src_ip",
    dst_ip_col: str = "dst_ip",
    score_col: str = "anomaly_score",
    dst_port_col: Optional[str] = None,
    proto_col: Optional[str] = None,
    key_mode: str = "dst_ip",
    window_sec: int = 3600,
    method: str = "topk_sum",
    topk: int = 10,
    tau: float = 0.0,
    topn: int = 20,
    out_dir: str = "rankings",
    track_unique_src: bool = True,
    max_unique_src: int = 10000,
    emit_global: bool = True,
) -> pd.DataFrame:
    """
    Aggregate anomaly scores into "suspicious IP ranking".

    Parameters
    ----------
    df : DataFrame with columns [src_ip_col, dst_ip_col, score_col, ts_col(optional)]
    key_mode : "dst_ip" | "src_ip" | "both"
    method :
      - "topk_sum" : sum of top-k anomaly scores per key (baseline)
      - "ppr_anom_pairwise" : (PPR diffusion × anomaly) + pairwise refinement using k-core as depth proxy
    window_sec : per-window ranking, e.g., 3600 for 1 hour
    emit_global : create rank_GLOBAL.csv
    """
    _safe_mkdir(out_dir)

    # basic columns sanity
    need_cols = {src_ip_col, dst_ip_col, score_col}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"aggregate_from_df: missing columns: {missing}")

    df = df.copy()
    df[src_ip_col] = df[src_ip_col].astype(str)
    df[dst_ip_col] = df[dst_ip_col].astype(str)

    # filter tau
    df = _maybe_filter_tau(df, score_col, tau)

    # ensure ts
    if ts_col in df.columns:
        df[ts_col] = _ensure_datetime_series(df[ts_col])
    else:
        # allow global ranking only
        df[ts_col] = pd.NaT

    # create key
    if key_mode == "dst_ip":
        df["__key"] = df[dst_ip_col].astype(str)
    elif key_mode == "src_ip":
        df["__key"] = df[src_ip_col].astype(str)
    elif key_mode == "both":
        # duplicate
        dfa = df.copy()
        dfb = df.copy()
        dfa["__key"] = dfa[src_ip_col].astype(str)
        dfb["__key"] = dfb[dst_ip_col].astype(str)
        df = pd.concat([dfa, dfb], axis=0, ignore_index=True)
    else:
        raise ValueError("key_mode must be one of: dst_ip, src_ip, both")

    # ---------------------------
    # Method: PPR×Anom + Pairwise
    # ---------------------------
    if method in ["ppr_anom_pairwise", "ppr_anom"]:
        # global only (can be extended to window ranking later)
        df_rank, meta = _rank_ppr_anom_pairwise(
            df=df.dropna(subset=["__key"]),
            src_ip_col=src_ip_col,
            dst_ip_col=dst_ip_col,
            score_col=score_col,
            key_mode=key_mode,
            topn=topn,
        )

        out_path = os.path.join(out_dir, "rank_GLOBAL.csv")
        df_rank.to_csv(out_path, index=False)

        # save meta
        try:
            with open(os.path.join(out_dir, "rank_model_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        return df_rank

    # ---------------------------
    # Method: baseline topk_sum
    # ---------------------------
    # per-window ranking
    if df[ts_col].notna().any():
        # lowercase 's' to avoid pandas FutureWarning
        window = f"{int(window_sec)}s"
        df["__ws"] = df[ts_col].dt.floor(window)

        # group by window+key
        # Avoid groupby.apply deprecation by manual loop (small overhead, stable)
        ws_values = df["__ws"].dropna().unique()
        ws_values = sorted(ws_values)

        for ws in ws_values:
            df_ws = df[df["__ws"] == ws]
            if df_ws.empty:
                continue

            rows = []
            for k, g in df_ws.groupby("__key", sort=False):
                out = _topk_sum_group(
                    g,
                    score_col=score_col,
                    topk=topk,
                    src_ip_col=src_ip_col,
                    track_unique_src=track_unique_src,
                    max_unique_src=max_unique_src,
                )
                rows.append(
                    {
                        "ip": str(k),
                        "score": out["score"],
                        "n_flows": out["n_flows"],
                        "topk_used": out["topk_used"],
                        "unique_src": out.get("unique_src", None),
                        "window_start": ws,
                    }
                )

            df_rank_ws = pd.DataFrame(rows)
            df_rank_ws = df_rank_ws.sort_values("score", ascending=False).head(topn)

            # file name
            # safe format
            ws_str = pd.Timestamp(ws).strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"rank_WS_{ws_str}.csv")
            df_rank_ws.to_csv(out_path, index=False)

    # global ranking
    if emit_global:
        rows = []
        for k, g in df.groupby("__key", sort=False):
            out = _topk_sum_group(
                g,
                score_col=score_col,
                topk=topk,
                src_ip_col=src_ip_col,
                track_unique_src=track_unique_src,
                max_unique_src=max_unique_src,
            )
            rows.append(
                {
                    "ip": str(k),
                    "score": out["score"],
                    "n_flows": out["n_flows"],
                    "topk_used": out["topk_used"],
                    "unique_src": out.get("unique_src", None),
                }
            )

        df_rank = pd.DataFrame(rows).sort_values("score", ascending=False).head(topn).reset_index(drop=True)
        out_path = os.path.join(out_dir, "rank_GLOBAL.csv")
        df_rank.to_csv(out_path, index=False)
        return df_rank

    # fallback
    return pd.DataFrame(columns=["ip", "score", "n_flows", "topk_used", "unique_src"])



def rank_ppr_anomaly(
    df: pd.DataFrame,
    src_ip_col="src_ip",
    dst_ip_col="dst_ip",
    score_col="anomaly_score",
    ts_col="ts",
    window_sec: int | None = None,
    threshold: float | None = None,
    alpha: float = 0.15,
    iters: int = 50,
    reverse: bool = False,   # True にすると dst->src 方向に拡散（「深さ」を逆向きに見る用途）
    beta_deg: float = 0.30,  # degreeボーナス
    beta_src: float = 0.50,  # unique_srcボーナス
    topn: int = 20,
) -> pd.DataFrame:
    """
    PPR diffusion over graph × anomaly.
    - base score: incoming anomaly sum (or filtered by threshold)
    - propagate via PageRank
    - optionally add degree/unique_src bonus
    """

    import numpy as np
    import pandas as pd

    if df.empty:
        return pd.DataFrame(columns=["ip", "score", "base", "in_deg", "out_deg", "unique_src"])

    work = df.copy()

    # --- score filter ---
    if threshold is not None:
        work = work[work[score_col] >= float(threshold)]
        if work.empty:
            # fallback: keep top 1% if everything filtered out
            work = df.nlargest(max(1, len(df)//100), score_col).copy()

    # --- time windows (optional) ---
    if window_sec is not None and ts_col in work.columns:
        # try convert to datetime
        ts = pd.to_datetime(work[ts_col], unit="ns", errors="coerce")
        if ts.notna().any():
            work["__ws"] = ts.dt.floor(f"{int(window_sec)}s")
        else:
            work["__ws"] = "GLOBAL"
    else:
        work["__ws"] = "GLOBAL"

    all_scores = {}

    for ws, g in work.groupby("__ws", sort=False):
        # nodes
        nodes = pd.Index(pd.unique(pd.concat([g[src_ip_col], g[dst_ip_col]], ignore_index=True)))
        node2i = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        if n == 0:
            continue

        # edges
        src = g[src_ip_col].map(node2i).to_numpy(dtype=np.int64)
        dst = g[dst_ip_col].map(node2i).to_numpy(dtype=np.int64)
        w = g[score_col].to_numpy(dtype=np.float64)

        if reverse:
            src, dst = dst, src  # reverse direction

        # adjacency matrix (dense: node数が小さい前提。44ノードなら余裕)
        A = np.zeros((n, n), dtype=np.float64)
        np.add.at(A, (src, dst), w)

        row_sum = A.sum(axis=1, keepdims=True)
        P = np.divide(A, row_sum, out=np.zeros_like(A), where=row_sum > 0)  # row-stochastic

        # base score = incoming anomaly
        base = np.zeros(n, dtype=np.float64)
        np.add.at(base, dst, w)
        s = base.sum()
        if s <= 0:
            continue
        base = base / s

        # PPR: r = alpha*base + (1-alpha)*P^T*r
        r = base.copy()
        for _ in range(int(iters)):
            r = alpha * base + (1.0 - alpha) * (P.T @ r)

        # degree features (from A>0, not raw edge count)
        out_deg = (A > 0).sum(axis=1).astype(np.float64)
        in_deg = (A > 0).sum(axis=0).astype(np.float64)

        # unique src per dst (original方向で見るのが自然なので g で作る)
        uniq_src = g.groupby(dst_ip_col)[src_ip_col].nunique()
        uniq_src_vec = np.array([uniq_src.get(ip, 0) for ip in nodes], dtype=np.float64)

        # final score
        score = r.copy()
        score = score * (1.0 + beta_deg * np.log1p(in_deg))
        score = score * (1.0 + beta_src * np.log1p(uniq_src_vec))

        # accumulate
        for i, ip in enumerate(nodes):
            all_scores[ip] = all_scores.get(ip, 0.0) + float(score[i])

    out = pd.DataFrame(
        [{"ip": k, "score": v} for k, v in all_scores.items()]
    ).sort_values("score", ascending=False)

    return out.head(topn).reset_index(drop=True)



def _is_private_ip(ip: str) -> bool:
    """Return True if ip is private/reserved/etc. Safe fallback."""
    try:
        return ipaddress.ip_address(str(ip)).is_private
    except Exception:
        return False


def _infer_datetime_unit(ts: pd.Series) -> str:
    """
    Infer datetime unit from magnitude.
    Accepts seconds, milliseconds, microseconds, nanoseconds.
    """
    s = pd.to_numeric(ts, errors="coerce").dropna()
    if s.empty:
        return "s"

    mx = float(s.max())

    # Rough thresholds for UNIX epoch scale:
    # s  ~ 1e9
    # ms ~ 1e12
    # us ~ 1e15
    # ns ~ 1e18
    if mx > 1e17:
        return "ns"
    if mx > 1e14:
        return "us"
    if mx > 1e11:
        return "ms"
    return "s"


def _to_datetime_safe(ts: pd.Series) -> pd.Series:
    """
    Convert timestamp series to pandas datetime, inferring unit.
    Returns NaT for invalid.
    """
    unit = _infer_datetime_unit(ts)
    out = pd.to_datetime(pd.to_numeric(ts, errors="coerce"), unit=unit, errors="coerce")
    return out


def _normalize_01(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Min-max normalize to [0,1]. If constant/empty -> zeros."""
    if x is None or len(x) == 0:
        return pd.Series([], dtype=float)
    xmin = float(np.nanmin(x.to_numpy())) if x.notna().any() else 0.0
    xmax = float(np.nanmax(x.to_numpy())) if x.notna().any() else 0.0
    if abs(xmax - xmin) < eps:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return (x - xmin) / (xmax - xmin + eps)


def _beacon_score_collective(ts_dt: np.ndarray, min_events: int = 6) -> float:
    """
    Beaconing score using *collective* inter-arrival stability for a target.
    Input: ts_dt as datetime64[ns] array, already sorted or unsorted.
    Output: score in [0,1], higher = more periodic/regular.
    """
    if ts_dt is None or len(ts_dt) < min_events:
        return 0.0

    # Sort, compute deltas in seconds
    ts_dt = np.sort(ts_dt.astype("datetime64[ns]"))
    delta_ns = np.diff(ts_dt).astype("timedelta64[ns]").astype(np.int64)
    if len(delta_ns) < (min_events - 1):
        return 0.0

    delta_sec = delta_ns / 1e9
    delta_sec = delta_sec[np.isfinite(delta_sec)]
    delta_sec = delta_sec[delta_sec > 0]

    if len(delta_sec) < (min_events - 1):
        return 0.0

    mean = float(np.mean(delta_sec))
    std = float(np.std(delta_sec))
    if mean <= 0:
        return 0.0

    # (1) Regularity via CV: lower CV => more periodic
    cv = std / (mean + 1e-12)
    regularity = 1.0 / (1.0 + cv)  # in (0,1]

    # (2) Concentration: many deltas near the same value => periodic
    # Robust binning: log-scale bins help wide distributions
    # For very small/large deltas, keep it stable.
    d = delta_sec
    d = np.clip(d, 1e-3, 1e6)
    logd = np.log10(d)
    # 12 bins across observed range
    bins = np.linspace(logd.min(), logd.max(), num=13)
    hist, _ = np.histogram(logd, bins=bins)
    peak_ratio = float(hist.max()) / float(len(d) + 1e-12)  # [0,1]

    # (3) Support factor: need enough events
    support = min(1.0, len(ts_dt) / 20.0)

    # Combine
    score = regularity * peak_ratio * support
    return float(np.clip(score, 0.0, 1.0))


def _weighted_hits(
    src_nodes: np.ndarray,
    dst_nodes: np.ndarray,
    weights: np.ndarray,
    num_nodes: int,
    iters: int = 50,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted HITS on directed edges src->dst.
    Returns hub, authority vectors in R^num_nodes.

    hub(u)  = sum_{u->v} w(u,v) * auth(v)
    auth(v) = sum_{u->v} w(u,v) * hub(u)
    """
    hub = np.ones(num_nodes, dtype=np.float64)
    auth = np.ones(num_nodes, dtype=np.float64)

    w = weights.astype(np.float64)
    w = np.clip(w, 0.0, np.percentile(w, 99.5) if len(w) > 1000 else w.max() + eps) + eps

    for _ in range(int(iters)):
        # hub update
        hub_new = np.zeros_like(hub)
        np.add.at(hub_new, src_nodes, w * auth[dst_nodes])

        # auth update
        auth_new = np.zeros_like(auth)
        np.add.at(auth_new, dst_nodes, w * hub_new[src_nodes])

        # normalize
        hn = np.linalg.norm(hub_new) + eps
        an = np.linalg.norm(auth_new) + eps
        hub = hub_new / hn
        auth = auth_new / an

    return hub, auth


@dataclass
class InfraRankConfig:
    # Core columns
    src_col: str = "src_ip"
    dst_col: str = "dst_ip"
    score_col: str = "anomaly_score"
    ts_col: str = "ts"

    # Optional meta
    dport_col: str = "dst_port"
    proto_col: str = "proto"

    # Ranking behavior
    topn: int = 20
    hits_iters: int = 50

    # Filtering
    min_score: float = 0.0          # keep edges with score > min_score
    use_threshold: Optional[float] = None  # if set, use max(min_score, threshold)
    max_edges_per_key: Optional[int] = 5000  # cap edges per target key (for huge datasets)

    # Target key definition (for CDN/rotating IP mitigation)
    key_mode: Literal["dst_ip", "signature"] = "dst_ip"
    # signature means: dst_ip:dst_port:proto

    # Infra focus
    include_private_dst: bool = True
    include_private_src: bool = True

    # Beacon
    beacon_min_events: int = 6

    # Weighting (tunable)
    w_authority: float = 1.0
    w_unique_src: float = 0.9
    w_active_days: float = 0.9
    w_beacon: float = 0.8
    w_anom: float = 0.5  # keep mild, since HITS already uses anomaly

    # Anti-hub penalty (reduce common infra like DNS/proxy)
    hub_penalty: float = 0.6  # stronger => penalize high indegree targets more


def rank_infra(df_flows: pd.DataFrame, cfg: InfraRankConfig = InfraRankConfig()) -> pd.DataFrame:
    """
    Infer attacker-side infra candidates (C2/management/aggregation) using:

    - Weighted HITS Authority (edge weights = anomaly_score)
    - unique_src (fan-in diversity)
    - active_days (persistence / re-appearance)
    - collective beacon score (periodicity stability)
    - mild anomaly aggregation

    Returns a DataFrame with top infra candidates.
    """
    need_cols = [cfg.src_col, cfg.dst_col, cfg.score_col]
    for c in need_cols:
        if c not in df_flows.columns:
            raise ValueError(f"rank_infra: missing required column: {c}")

    df = df_flows.copy()

    # sanitize
    df[cfg.src_col] = df[cfg.src_col].astype(str)
    df[cfg.dst_col] = df[cfg.dst_col].astype(str)
    df[cfg.score_col] = pd.to_numeric(df[cfg.score_col], errors="coerce").fillna(0.0)

    # filter private ip if desired
    if not cfg.include_private_src:
        df = df[~df[cfg.src_col].map(_is_private_ip)]
    if not cfg.include_private_dst:
        df = df[~df[cfg.dst_col].map(_is_private_ip)]

    # score filter
    thr = cfg.min_score
    if cfg.use_threshold is not None:
        thr = max(thr, float(cfg.use_threshold))
    df = df[df[cfg.score_col] > float(thr)].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "key", "dst_ip", "infra_score",
            "authority", "unique_src", "active_days", "beacon_score",
            "anom_topk_sum", "anom_mean", "anom_max",
            "deg_in", "deg_out",
        ])

    # build key
    if cfg.key_mode == "signature":
        # safer even if missing meta columns
        dport = df[cfg.dport_col].astype(str) if cfg.dport_col in df.columns else "NA"
        proto = df[cfg.proto_col].astype(str) if cfg.proto_col in df.columns else "NA"
        df["__key"] = df[cfg.dst_col] + ":" + dport + ":" + proto
    else:
        df["__key"] = df[cfg.dst_col]

    # optional cap per key (avoid huge keys dominating)
    if cfg.max_edges_per_key is not None and cfg.max_edges_per_key > 0:
        k = int(cfg.max_edges_per_key)
        # Keep top-k edges by anomaly within each key
        df = df.sort_values(cfg.score_col, ascending=False)
        df = df.groupby("__key", sort=False).head(k).copy()

    # timestamp -> datetime (for active_days/beacon)
    if cfg.ts_col in df.columns:
        df["__dt"] = _to_datetime_safe(df[cfg.ts_col])
        df["__day"] = df["__dt"].dt.floor("D")
    else:
        df["__dt"] = pd.NaT
        df["__day"] = pd.NaT

    # -------------------------
    # Weighted HITS (Authority)
    # Nodes are src_ip and key nodes in one namespace
    # -------------------------
    nodes = pd.Index(pd.unique(pd.concat([df[cfg.src_col], df["__key"]], ignore_index=True)))
    nid = {n: i for i, n in enumerate(nodes)}

    src_id = df[cfg.src_col].map(nid).to_numpy(dtype=np.int64)
    dst_id = df["__key"].map(nid).to_numpy(dtype=np.int64)
    w = df[cfg.score_col].to_numpy(dtype=np.float64)

    hub, auth = _weighted_hits(
        src_nodes=src_id,
        dst_nodes=dst_id,
        weights=w,
        num_nodes=len(nodes),
        iters=cfg.hits_iters,
    )

    # Extract authority for target keys only
    key_index = df["__key"].drop_duplicates()
    key_ids = key_index.map(nid).to_numpy(dtype=np.int64)
    authority = pd.Series(auth[key_ids], index=key_index.to_numpy())

    # degrees (in/out on the bipartite projection)
    deg_in = pd.Series(0.0, index=key_index.to_numpy())
    deg_out = pd.Series(0.0, index=key_index.to_numpy())
    # indegree: count of edges into key
    tmp_in = df.groupby("__key", sort=False)[cfg.src_col].size().astype(float)
    deg_in.loc[tmp_in.index] = tmp_in
    # outdegree: for src nodes, but we can report per key a proxy (same as deg_in) and skip deg_out
    # For completeness: deg_out of key itself in this bipartite is 0. Keep as 0.
    # If you later extend to key->key edges, you can populate it.

    # -------------------------
    # Aggregate features per key
    # -------------------------
    g = df.groupby("__key", sort=False)

    unique_src = g[cfg.src_col].nunique().astype(float)

    # persistence: number of active days (NaT-safe)
    if df["__day"].notna().any():
        active_days = g["__day"].nunique().astype(float)
    else:
        active_days = pd.Series(0.0, index=unique_src.index)

    # anomaly aggregates
    anom_mean = g[cfg.score_col].mean().astype(float)
    anom_max = g[cfg.score_col].max().astype(float)

    # Top-k sum (robust)
    def _topk_sum(s: pd.Series, k: int = 10) -> float:
        if s.empty:
            return 0.0
        return float(np.sum(np.sort(s.to_numpy())[-k:]))

    anom_topk_sum = g[cfg.score_col].apply(_topk_sum).astype(float)

    # beacon collective score
    beacon_scores = {}
    if df["__dt"].notna().any():
        for key, sub in g:
            ts_dt = sub["__dt"].dropna().to_numpy(dtype="datetime64[ns]")
            beacon_scores[key] = _beacon_score_collective(ts_dt, min_events=cfg.beacon_min_events)
    beacon_score = pd.Series(beacon_scores, dtype=float)
    beacon_score = beacon_score.reindex(unique_src.index).fillna(0.0)

    # -------------------------
    # Combine into InfraScore
    #   - normalize main axes
    #   - penalty for common infra (high deg_in)
    # -------------------------
    auth_n = _normalize_01(authority.reindex(unique_src.index).fillna(0.0))
    u_n = _normalize_01(np.log1p(unique_src))
    d_n = _normalize_01(np.log1p(active_days))
    b_n = beacon_score.clip(0.0, 1.0)  # already [0,1]
    a_n = _normalize_01(np.log1p(anom_topk_sum))

    # anti-hub penalty: higher indegree -> more likely generic infra (DNS/proxy)
    hub_pen = 1.0 / (1.0 + np.log1p(deg_in.reindex(unique_src.index).fillna(0.0)))
    hub_pen = hub_pen ** float(cfg.hub_penalty)

    # score = product of components (with small floors)
    score = (
        (0.10 + cfg.w_authority * auth_n) *
        (0.10 + cfg.w_unique_src * u_n) *
        (0.10 + cfg.w_active_days * d_n) *
        (0.05 + cfg.w_beacon * b_n) *
        (0.20 + cfg.w_anom * a_n) *
        hub_pen
    )

    out = pd.DataFrame({
        "key": unique_src.index.astype(str),
        "dst_ip": unique_src.index.astype(str),  # will be overwritten for signature mode
        "infra_score": score.astype(float),
        "authority": authority.reindex(unique_src.index).fillna(0.0).astype(float),
        "unique_src": unique_src.astype(float),
        "active_days": active_days.reindex(unique_src.index).fillna(0.0).astype(float),
        "beacon_score": b_n.astype(float),
        "anom_topk_sum": anom_topk_sum.astype(float),
        "anom_mean": anom_mean.astype(float),
        "anom_max": anom_max.astype(float),
        "deg_in": deg_in.reindex(unique_src.index).fillna(0.0).astype(float),
        "deg_out": deg_out.reindex(unique_src.index).fillna(0.0).astype(float),
    })

    if cfg.key_mode == "signature":
        # key = "dst_ip:dst_port:proto"
        # keep dst_ip only for display
        out["dst_ip"] = out["key"].str.split(":", n=1).str[0]

    out = out.sort_values("infra_score", ascending=False).head(int(cfg.topn)).reset_index(drop=True)
    return out
