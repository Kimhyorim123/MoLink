#!/usr/bin/env python3

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ConcurrencyStats:
    n_rows: int
    n_ok: int
    max_concurrency: int
    p50_concurrency: float
    p90_concurrency: float
    mean_concurrency: float


def _percentile(sorted_vals: List[int], p: float) -> float:
    if not sorted_vals:
        return float('nan')
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def load_intervals(csv_path: Path) -> Tuple[List[Tuple[int, int]], int, int]:
    intervals: List[Tuple[int, int]] = []
    n_rows = 0
    n_ok = 0
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            status = (row.get('status') or '').strip()
            if status != 'ok':
                continue
            try:
                s = int(row['start_time_ns'])
                e = int(row['end_time_ns'])
            except Exception:
                continue
            if e <= s:
                continue
            intervals.append((s, e))
            n_ok += 1
    return intervals, n_rows, n_ok


def compute_concurrency(intervals: List[Tuple[int, int]]) -> List[int]:
    # Sweep line over start/end events.
    events: List[Tuple[int, int]] = []
    for s, e in intervals:
        events.append((s, +1))
        events.append((e, -1))
    # End before start at same timestamp to avoid overcounting zero-length overlaps.
    events.sort(key=lambda x: (x[0], x[1]))

    cur = 0
    samples: List[int] = []
    for _, delta in events:
        cur += delta
        samples.append(cur)
    return samples


def stats_for_csv(csv_path: Path) -> Optional[ConcurrencyStats]:
    intervals, n_rows, n_ok = load_intervals(csv_path)
    if not intervals:
        return None
    samples = compute_concurrency(intervals)
    if not samples:
        return None
    sorted_samples = sorted(samples)
    max_c = max(samples)
    mean_c = sum(samples) / len(samples)
    return ConcurrencyStats(
        n_rows=n_rows,
        n_ok=n_ok,
        max_concurrency=max_c,
        p50_concurrency=_percentile(sorted_samples, 50),
        p90_concurrency=_percentile(sorted_samples, 90),
        mean_concurrency=mean_c,
    )


def iter_csv_files(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob('**/*.csv')))
        elif p.is_file() and p.suffix == '.csv':
            out.append(p)
    # Heuristic: ignore summary CSVs.
    filtered: List[Path] = []
    for p in out:
        if '/_summary/' in p.as_posix():
            continue
        if p.name in {'summary.csv', 'table2.csv'}:
            continue
        filtered.append(p)
    return filtered


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute effective microbatch via in-flight concurrency from loadgen CSVs.')
    ap.add_argument('paths', nargs='+', help='CSV file(s) or directories containing CSVs')
    ap.add_argument('--out', default=None, help='Optional output CSV path')
    args = ap.parse_args()

    csv_files = iter_csv_files([Path(x) for x in args.paths])
    rows: List[Dict[str, str]] = []

    for csv_path in csv_files:
        st = stats_for_csv(csv_path)
        if st is None:
            continue
        rows.append({
            'csv': str(csv_path),
            'n_rows': str(st.n_rows),
            'n_ok': str(st.n_ok),
            'max_concurrency': str(st.max_concurrency),
            'p50_concurrency': f'{st.p50_concurrency:.2f}',
            'p90_concurrency': f'{st.p90_concurrency:.2f}',
            'mean_concurrency': f'{st.mean_concurrency:.2f}',
        })

    rows.sort(key=lambda r: r['csv'])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['csv'])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(str(out_path))
    else:
        for r in rows:
            print(
                r['csv'],
                'max=', r['max_concurrency'],
                'p50=', r['p50_concurrency'],
                'p90=', r['p90_concurrency'],
                'mean=', r['mean_concurrency'],
                f"(ok={r['n_ok']}/{r['n_rows']})",
            )


if __name__ == '__main__':
    main()
