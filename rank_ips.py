#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank suspicious IPs/endpoints from per-flow anomaly scores.

Typical pipeline:
  1) GraphIDS inference -> per-flow CSV with columns:
       ts, src_ip, dst_ip, anomaly_score, (optional) dst_port, proto
  2) This script aggregates anomaly_score per key (dst_ip or endpoint) per time window,
     then outputs top-N suspicious keys.

Assumptions for streaming mode:
  - Input is sorted by ts ascending (recommended). If not, use --no_stream (loads all).

Scoring methods:
  - topk_sum: sum of top-k anomaly scores per key in a window
  - exceed_sum: sum(max(0, score - tau)) per key in a window
  - mean: mean(score) per key in a window
  - max: max(score) per key in a window
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any
import numpy as np
try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. pip install pandas") from e


def parse_ts(x: Any) -> datetime:
    """
    Parse timestamp field to timezone-aware datetime (UTC).
    Accepts ISO8601 strings, unix seconds, pandas Timestamp, datetime.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("ts is null")

    if isinstance(x, datetime):
        dt = x
    else:
        dt = pd.to_datetime(x, utc=True, errors="raise").to_pydatetime()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def floor_time(dt: datetime, window: timedelta) -> datetime:
    """
    Floor dt to window boundary (UTC).
    """
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = int((dt - epoch).total_seconds())
    w = int(window.total_seconds())
    floored = (seconds // w) * w
    return epoch + timedelta(seconds=floored)


# -----------------------------
# Score accumulators
# -----------------------------
class Accumulator:
    def update(self, score: float, src_ip: Optional[str] = None) -> None:
        raise NotImplementedError

    def value(self) -> float:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def src_nunique(self) -> Optional[int]:
        return None


class TopKSum(Accumulator):
    def __init__(self, k: int, track_unique_src: bool = False, max_unique_src: int = 50000) -> None:
        self.k = max(1, k)
        self.heap: List[float] = []  # min-heap size<=k
        self._count = 0
        self._sum_topk = 0.0
        self._track_unique_src = track_unique_src
        self._src_set = set() if track_unique_src else None
        self._max_unique_src = max_unique_src

    def update(self, score: float, src_ip: Optional[str] = None) -> None:
        self._count += 1

        # update top-k heap + sum in O(log k)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, score)
            self._sum_topk += score
        else:
            if score > self.heap[0]:
                smallest = heapq.heapreplace(self.heap, score)
                self._sum_topk += (score - smallest)

        if self._track_unique_src and src_ip is not None and self._src_set is not None:
            # cap to avoid unbounded memory
            if len(self._src_set) < self._max_unique_src:
                self._src_set.add(src_ip)

    def value(self) -> float:
        return float(self._sum_topk)

    def count(self) -> int:
        return self._count

    def src_nunique(self) -> Optional[int]:
        if self._src_set is None:
            return None
        return len(self._src_set)


class ExceedSum(Accumulator):
    def __init__(self, tau: float, track_unique_src: bool = False, max_unique_src: int = 50000) -> None:
        self.tau = float(tau)
        self._sum = 0.0
        self._count = 0
        self._track_unique_src = track_unique_src
        self._src_set = set() if track_unique_src else None
        self._max_unique_src = max_unique_src

    def update(self, score: float, src_ip: Optional[str] = None) -> None:
        self._count += 1
        if score > self.tau:
            self._sum += (score - self.tau)

        if self._track_unique_src and src_ip is not None and self._src_set is not None:
            if len(self._src_set) < self._max_unique_src:
                self._src_set.add(src_ip)

    def value(self) -> float:
        return float(self._sum)

    def count(self) -> int:
        return self._count

    def src_nunique(self) -> Optional[int]:
        if self._src_set is None:
            return None
        return len(self._src_set)


class Mean(Accumulator):
    def __init__(self, track_unique_src: bool = False, max_unique_src: int = 50000) -> None:
        self._sum = 0.0
        self._count = 0
        self._track_unique_src = track_unique_src
        self._src_set = set() if track_unique_src else None
        self._max_unique_src = max_unique_src

    def update(self, score: float, src_ip: Optional[str] = None) -> None:
        self._count += 1
        self._sum += score

        if self._track_unique_src and src_ip is not None and self._src_set is not None:
            if len(self._src_set) < self._max_unique_src:
                self._src_set.add(src_ip)

    def value(self) -> float:
        return float(self._sum / self._count) if self._count else 0.0

    def count(self) -> int:
        return self._count

    def src_nunique(self) -> Optional[int]:
        if self._src_set is None:
            return None
        return len(self._src_set)


class Max(Accumulator):
    def __init__(self, track_unique_src: bool = False, max_unique_src: int = 50000) -> None:
        self._max = float("-inf")
        self._count = 0
        self._track_unique_src = track_unique_src
        self._src_set = set() if track_unique_src else None
        self._max_unique_src = max_unique_src

    def update(self, score: float, src_ip: Optional[str] = None) -> None:
        self._count += 1
        if score > self._max:
            self._max = score

        if self._track_unique_src and src_ip is not None and self._src_set is not None:
            if len(self._src_set) < self._max_unique_src:
                self._src_set.add(src_ip)

    def value(self) -> float:
        return float(self._max) if self._count else 0.0

    def count(self) -> int:
        return self._count

    def src_nunique(self) -> Optional[int]:
        if self._src_set is None:
            return None
        return len(self._src_set)


# -----------------------------
# Aggregation core
# -----------------------------
@dataclass
class WindowResult:
    window_start: datetime
    window_end: datetime
    rows: List[Dict[str, Any]]


def make_key(
    row: Dict[str, Any],
    mode: str,
    dst_ip_col: str,
    dst_port_col: str,
    proto_col: str,
) -> str:
    if mode == "dst_ip":
        return str(row[dst_ip_col])
    if mode == "endpoint":
        dip = str(row[dst_ip_col])
        dport = str(row.get(dst_port_col, ""))
        proto = str(row.get(proto_col, ""))
        return f"{dip}:{dport}:{proto}"
    raise ValueError(f"Unknown key_mode={mode}")


def create_accumulator(
    method: str,
    topk: int,
    tau: float,
    track_unique_src: bool,
    max_unique_src: int,
) -> Accumulator:
    if method == "topk_sum":
        return TopKSum(k=topk, track_unique_src=track_unique_src, max_unique_src=max_unique_src)
    if method == "exceed_sum":
        return ExceedSum(tau=tau, track_unique_src=track_unique_src, max_unique_src=max_unique_src)
    if method == "mean":
        return Mean(track_unique_src=track_unique_src, max_unique_src=max_unique_src)
    if method == "max":
        return Max(track_unique_src=track_unique_src, max_unique_src=max_unique_src)
    raise ValueError(f"Unknown method={method}")


def dump_window_csv(
    out_path: str,
    window_start: datetime,
    window_end: datetime,
    key_name: str,
    top_rows: List[Tuple[str, Accumulator]],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_start_utc", "window_end_utc", key_name, "score", "flow_count", "src_ip_nunique"])
        for key, acc in top_rows:
            w.writerow([
                window_start.isoformat(),
                window_end.isoformat(),
                key,
                f"{acc.value():.10g}",
                acc.count(),
                "" if acc.src_nunique() is None else acc.src_nunique(),
            ])


def top_n_from_map(
    m: Dict[str, Accumulator],
    n: int
) -> List[Tuple[str, Accumulator]]:
    # sort descending by score
    return sorted(m.items(), key=lambda kv: kv[1].value(), reverse=True)[:n]


# -----------------------------
# Readers
# -----------------------------
def iter_rows_csv(
    path: str,
    usecols: List[str],
    chunksize: int,
) -> Iterator[Dict[str, Any]]:
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        # ensure dict per row
        for r in chunk.to_dict(orient="records"):
            yield r


def iter_rows_parquet(
    path: str,
    usecols: List[str],
    chunksize: int,
) -> Iterator[Dict[str, Any]]:
    # pandas read_parquet has no chunksize; fallback: load all
    df = pd.read_parquet(path, columns=usecols)
    for r in df.to_dict(orient="records"):
        yield r


# -----------------------------
# Main aggregation
# -----------------------------
def aggregate_streaming(
    input_path: str,
    input_format: str,
    ts_col: str,
    src_ip_col: str,
    dst_ip_col: str,
    score_col: str,
    dst_port_col: str,
    proto_col: str,
    key_mode: str,
    window_sec: int,
    method: str,
    topk: int,
    tau: float,
    topn: int,
    chunksize: int,
    out_dir: str,
    track_unique_src: bool,
    max_unique_src: int,
    emit_global: bool,
) -> None:
    window = timedelta(seconds=window_sec)

    # choose columns
    usecols = [ts_col, src_ip_col, dst_ip_col, score_col]
    if key_mode == "endpoint":
        # may not exist; row.get handles missing
        usecols += [dst_port_col, proto_col]
    usecols = list(dict.fromkeys(usecols))  # de-dup

    # row iterator
    if input_format == "csv":
        it = iter_rows_csv(input_path, usecols=usecols, chunksize=chunksize)
    elif input_format == "parquet":
        it = iter_rows_parquet(input_path, usecols=usecols, chunksize=chunksize)
    else:
        raise ValueError("input_format must be csv or parquet")

    current_ws: Optional[datetime] = None
    current_we: Optional[datetime] = None
    acc_map: Dict[str, Accumulator] = {}
    global_map: Dict[str, Accumulator] = {} if emit_global else {}

    def flush_window(ws: datetime, we: datetime) -> None:
        if not acc_map:
            return
        top_rows = top_n_from_map(acc_map, topn)
        fname = f"rank_{ws.strftime('%Y%m%dT%H%M%SZ')}_{we.strftime('%Y%m%dT%H%M%SZ')}.csv"
        out_path = os.path.join(out_dir, fname)
        key_name = "dst_ip" if key_mode == "dst_ip" else "endpoint"
        dump_window_csv(out_path, ws, we, key_name, top_rows)
        acc_map.clear()

    for row in it:
        dt = parse_ts(row[ts_col])
        ws = floor_time(dt, window)
        we = ws + window

        if current_ws is None:
            current_ws, current_we = ws, we

        # If window advanced, flush until we catch up.
        # Assumes sorted by ts; if gaps, it will just flush once and move.
        if ws != current_ws:
            flush_window(current_ws, current_we)  # type: ignore[arg-type]
            current_ws, current_we = ws, we

        key = make_key(row, key_mode, dst_ip_col, dst_port_col, proto_col)
        try:
            score = float(row[score_col])
        except Exception:
            continue

        src_ip = str(row.get(src_ip_col, "")) if track_unique_src else None

        if key not in acc_map:
            acc_map[key] = create_accumulator(method, topk, tau, track_unique_src, max_unique_src)
        acc_map[key].update(score, src_ip=src_ip)

        if emit_global:
            if key not in global_map:
                global_map[key] = create_accumulator(method, topk, tau, track_unique_src, max_unique_src)
            global_map[key].update(score, src_ip=src_ip)

    # final flush
    if current_ws is not None and current_we is not None:
        flush_window(current_ws, current_we)

    if emit_global and global_map:
        top_rows = top_n_from_map(global_map, topn)
        out_path = os.path.join(out_dir, "rank_GLOBAL.csv")
        key_name = "dst_ip" if key_mode == "dst_ip" else "endpoint"
        dump_window_csv(out_path, datetime(1970,1,1,tzinfo=timezone.utc), datetime(1970,1,1,tzinfo=timezone.utc), key_name, top_rows)


def aggregate_non_stream(
    input_path: str,
    input_format: str,
    ts_col: str,
    src_ip_col: str,
    dst_ip_col: str,
    score_col: str,
    dst_port_col: str,
    proto_col: str,
    key_mode: str,
    window_sec: int,
    method: str,
    topk: int,
    tau: float,
    topn: int,
    out_dir: str,
    track_unique_src: bool,
    max_unique_src: int,
    emit_global: bool,
) -> None:
    # Load all then groupby. Safer if input is not sorted, but needs RAM.
    if input_format == "csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    window = f"{window_sec}S"
    df["__ws"] = df[ts_col].dt.floor(window)

    if key_mode == "dst_ip":
        df["__key"] = df[dst_ip_col].astype(str)
        key_name = "dst_ip"
    else:
        # endpoint mode
        df["__key"] = (
            df[dst_ip_col].astype(str)
            + ":"
            + df.get(dst_port_col, "").astype(str)
            + ":"
            + df.get(proto_col, "").astype(str)
        )
        key_name = "endpoint"

    os.makedirs(out_dir, exist_ok=True)

    def score_group(g: pd.DataFrame) -> float:
        s = g[score_col].astype(float).to_numpy()
        if method == "topk_sum":
            k = max(1, topk)
            if len(s) <= k:
                return float(s.sum())
            # partial sort
            import numpy as np
            idx = np.argpartition(s, -k)[-k:]
            return float(s[idx].sum())
        if method == "exceed_sum":
            return float(((s - tau) * (s > tau)).sum())
        if method == "mean":
            return float(s.mean()) if len(s) else 0.0
        if method == "max":
            return float(s.max()) if len(s) else 0.0
        raise ValueError(method)

    # per-window ranking
    for ws, gws in df.groupby("__ws"):
        scores = gws.groupby("__key", sort=False).apply(score_group, include_groups=False)
        counts = gws.groupby("__key", sort=False).size()
        if track_unique_src:
            nunique = gws.groupby("__key", sort=False)[src_ip_col].nunique()
        else:
            nunique = None

        top = scores.sort_values(ascending=False).head(topn)
        out_path = os.path.join(out_dir, f"rank_{ws.strftime('%Y%m%dT%H%M%SZ')}_{(ws + pd.Timedelta(seconds=window_sec)).strftime('%Y%m%dT%H%M%SZ')}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["window_start_utc", "window_end_utc", key_name, "score", "flow_count", "src_ip_nunique"])
            for key, sc in top.items():
                w.writerow([
                    ws.to_pydatetime().isoformat(),
                    (ws + pd.Timedelta(seconds=window_sec)).to_pydatetime().isoformat(),
                    key,
                    f"{float(sc):.10g}",
                    int(counts.loc[key]),
                    "" if nunique is None else int(nunique.loc[key]),
                ])

    if emit_global:
        scores = df.groupby("__key", sort=False).apply(score_group, include_groups=False).sort_values(ascending=False).head(topn)
        counts = df.groupby("__key", sort=False).size()
        if track_unique_src:
            nunique = df.groupby("__key", sort=False)[src_ip_col].nunique()
        else:
            nunique = None
        out_path = os.path.join(out_dir, "rank_GLOBAL.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["window_start_utc", "window_end_utc", key_name, "score", "flow_count", "src_ip_nunique"])
            for key, sc in scores.items():
                w.writerow([
                    datetime(1970,1,1,tzinfo=timezone.utc).isoformat(),
                    datetime(1970,1,1,tzinfo=timezone.utc).isoformat(),
                    key,
                    f"{float(sc):.10g}",
                    int(counts.loc[key]),
                    "" if nunique is None else int(nunique.loc[key]),
                ])


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Per-flow score file (CSV or Parquet)")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--ts_col", default="ts")
    ap.add_argument("--src_ip_col", default="src_ip")
    ap.add_argument("--dst_ip_col", default="dst_ip")
    ap.add_argument("--score_col", default="anomaly_score")
    ap.add_argument("--dst_port_col", default="dst_port")
    ap.add_argument("--proto_col", default="proto")

    ap.add_argument("--key_mode", choices=["dst_ip", "endpoint"], default="dst_ip",
                    help="dst_ip: rank by dst_ip, endpoint: rank by dst_ip:dst_port:proto")

    ap.add_argument("--window_sec", type=int, default=3600, help="Time window seconds (default 3600=1h)")
    ap.add_argument("--method", choices=["topk_sum", "exceed_sum", "mean", "max"], default="topk_sum")
    ap.add_argument("--topk", type=int, default=20, help="k for topk_sum")
    ap.add_argument("--tau", type=float, default=0.0, help="tau for exceed_sum")
    ap.add_argument("--topn", type=int, default=50, help="top N keys output per window")

    ap.add_argument("--chunksize", type=int, default=500000, help="CSV chunksize for streaming")
    ap.add_argument("--no_stream", action="store_true", help="Disable streaming (loads all, works if ts unsorted)")
    ap.add_argument("--track_unique_src", action="store_true",
                    help="Also compute src_ip distinct count per key (memory heavier)")
    ap.add_argument("--max_unique_src", type=int, default=50000,
                    help="Cap unique src_ip stored per key in streaming mode")
    ap.add_argument("--emit_global", action="store_true", help="Also output rank_GLOBAL.csv")

    args = ap.parse_args()

    if args.method == "exceed_sum" and args.tau <= 0.0:
        raise SystemExit("--method exceed_sum requires --tau > 0")

    if args.no_stream:
        aggregate_non_stream(
            input_path=args.input,
            input_format=args.format,
            ts_col=args.ts_col,
            src_ip_col=args.src_ip_col,
            dst_ip_col=args.dst_ip_col,
            score_col=args.score_col,
            dst_port_col=args.dst_port_col,
            proto_col=args.proto_col,
            key_mode=args.key_mode,
            window_sec=args.window_sec,
            method=args.method,
            topk=args.topk,
            tau=args.tau,
            topn=args.topn,
            out_dir=args.out_dir,
            track_unique_src=args.track_unique_src,
            max_unique_src=args.max_unique_src,
            emit_global=args.emit_global,
        )
    else:
        aggregate_streaming(
            input_path=args.input,
            input_format=args.format,
            ts_col=args.ts_col,
            src_ip_col=args.src_ip_col,
            dst_ip_col=args.dst_ip_col,
            score_col=args.score_col,
            dst_port_col=args.dst_port_col,
            proto_col=args.proto_col,
            key_mode=args.key_mode,
            window_sec=args.window_sec,
            method=args.method,
            topk=args.topk,
            tau=args.tau,
            topn=args.topn,
            chunksize=args.chunksize,
            out_dir=args.out_dir,
            track_unique_src=args.track_unique_src,
            max_unique_src=args.max_unique_src,
            emit_global=args.emit_global,
        )

def aggregate_from_df(
    df: "pd.DataFrame",
    ts_col: str = "ts",
    src_ip_col: str = "src_ip",
    dst_ip_col: str = "dst_ip",
    score_col: str = "anomaly_score",
    dst_port_col: str = "dst_port",
    proto_col: str = "proto",
    key_mode: str = "dst_ip",
    window_sec: int = 3600,
    method: str = "topk_sum",
    topk: int = 20,
    tau: float = 0.0,
    topn: int = 50,
    out_dir: str = "rankings",
    track_unique_src: bool = False,
    max_unique_src: int = 50000,
    emit_global: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ts が無い場合：GLOBAL のみ出す（時間窓は作らない）
    if ts_col not in df.columns:
        if not emit_global:
            return

        # key 作成
        if key_mode == "dst_ip":
            df_key = df[dst_ip_col].astype(str)
            key_name = "dst_ip"
        elif key_mode == "endpoint":
            df_key = (
                df[dst_ip_col].astype(str)
                + ":" + df.get(dst_port_col, "").astype(str)
                + ":" + df.get(proto_col, "").astype(str)
            )
            key_name = "endpoint"
        else:
            raise ValueError(f"Unknown key_mode={key_mode}")

        s = df[score_col].astype(float).to_numpy()

        def score_group(g: "pd.DataFrame") -> float:
            arr = g[score_col].astype(float).to_numpy()
            if method == "topk_sum":
                k = max(1, topk)
                if len(arr) <= k:
                    return float(arr.sum())
                idx = np.argpartition(arr, -k)[-k:]
                return float(arr[idx].sum())
            if method == "exceed_sum":
                return float(((arr - tau) * (arr > tau)).sum())
            if method == "mean":
                return float(arr.mean()) if len(arr) else 0.0
            if method == "max":
                return float(arr.max()) if len(arr) else 0.0
            raise ValueError(method)

        tmp = df.copy()
        tmp["__key"] = df_key
        scores = tmp.groupby("__key", sort=False).apply(score_group).sort_values(ascending=False).head(topn)
        counts = tmp.groupby("__key", sort=False).size()
        nunique = tmp.groupby("__key", sort=False)[src_ip_col].nunique() if track_unique_src else None

        out_path = os.path.join(out_dir, "rank_GLOBAL.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["window_start_utc", "window_end_utc", key_name, "score", "flow_count", "src_ip_nunique"])
            for key, sc in scores.items():
                w.writerow([
                    datetime(1970,1,1,tzinfo=timezone.utc).isoformat(),
                    datetime(1970,1,1,tzinfo=timezone.utc).isoformat(),
                    key,
                    f"{float(sc):.10g}",
                    int(counts.loc[key]),
                    "" if nunique is None else int(nunique.loc[key]),
                ])
        return

    # ts がある場合：既存の non-stream 相当で時間窓＋GLOBAL
    tmp_path = os.path.join(out_dir, "_tmp_flows_for_ranking.csv")
    df.to_csv(tmp_path, index=False)

    aggregate_non_stream(
        input_path=tmp_path,
        input_format="csv",
        ts_col=ts_col,
        src_ip_col=src_ip_col,
        dst_ip_col=dst_ip_col,
        score_col=score_col,
        dst_port_col=dst_port_col,
        proto_col=proto_col,
        key_mode=key_mode,
        window_sec=window_sec,
        method=method,
        topk=topk,
        tau=tau,
        topn=topn,
        out_dir=out_dir,
        track_unique_src=track_unique_src,
        max_unique_src=max_unique_src,
        emit_global=emit_global,
    )


if __name__ == "__main__":
    main()