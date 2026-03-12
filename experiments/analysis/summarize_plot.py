#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["ttft_ms", "e2e_ms", "tpot_ms"]
METRIC_LABELS = {
    "ttft_ms": "TTFT (s)",
    "tpot_ms": "TPOT (s)",
    "e2e_ms": "End-to-end Latency (s)",
}


def _iter_csv_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.csv")))
        else:
            paths.append(p)

    uniq: List[Path] = []
    seen = set()
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def p95(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    if len(x) == 0:
        return float("nan")
    return float(np.percentile(x, 95))


def _fmt_value_seconds(metric: str, value_s: float) -> str:
    if np.isnan(value_s):
        return "nan"
    if metric == "e2e_ms":
        return f"{value_s:.0f}"
    return f"{value_s:.2f}"


def _fmt_value_with_pct(metric: str, value_s: float, baseline_s: float) -> str:
    if np.isnan(value_s):
        return "nan"
    if np.isnan(baseline_s) or baseline_s == 0:
        return _fmt_value_seconds(metric, value_s)
    pct = 100.0 * (value_s / baseline_s)
    return f"{_fmt_value_seconds(metric, value_s)} ({pct:.0f}%)"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # only successful requests
    df = df[df["status"] == "ok"].copy()

    group_cols = ["system", "rate_rps", "bandwidth_mbps", "rtt_ms"]

    agg = {}
    for m in METRICS:
        agg[m] = ["mean", "median", p95]

    out = df.groupby(group_cols).agg(agg)
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    out = out.reset_index().sort_values(["system", "rate_rps"])
    return out


def _pick_stat_col(metric: str, stat: str) -> str:
    if stat not in {"mean", "median", "p95"}:
        raise ValueError(f"unknown stat: {stat}")
    return f"{metric}_{stat}"


def make_table2(
    summary: pd.DataFrame,
    *,
    stat: str,
    baseline_system: str,
    systems: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a wide Table-2-style dataframe.

    Values are in seconds. Non-baseline systems are annotated with (X%)
    relative to the baseline_system for the same rate.
    """

    if systems is None:
        systems = sorted(summary["system"].unique().tolist())
    else:
        systems = list(systems)

    rates = sorted(summary["rate_rps"].unique().tolist())

    # Build lookup: (system, rate) -> metric value in seconds
    lookup = {}
    for _, row in summary.iterrows():
        key = (str(row["system"]), float(row["rate_rps"]))
        lookup[key] = row

    rows = []
    for rate in rates:
        base_row = lookup.get((baseline_system, float(rate)))
        base_vals_s = {}
        for metric in ["ttft_ms", "tpot_ms", "e2e_ms"]:
            col = _pick_stat_col(metric, stat)
            if base_row is None or col not in base_row:
                base_vals_s[metric] = float("nan")
            else:
                base_vals_s[metric] = float(base_row[col]) / 1000.0

        out_row = {"Rate (req/s)": rate}

        for metric in ["ttft_ms", "tpot_ms", "e2e_ms"]:
            for sysname in systems:
                row = lookup.get((sysname, float(rate)))
                col = _pick_stat_col(metric, stat)
                value_s = float("nan")
                if row is not None and col in row and pd.notna(row[col]):
                    value_s = float(row[col]) / 1000.0

                if sysname == baseline_system:
                    cell = _fmt_value_seconds(metric, value_s)
                else:
                    cell = _fmt_value_with_pct(metric, value_s, base_vals_s[metric])

                out_row[f"{METRIC_LABELS[metric]} | {sysname}"] = cell

        rows.append(out_row)

    df = pd.DataFrame(rows)

    # Order columns: Rate then metrics grouped by metric order and system order
    cols = ["Rate (req/s)"]
    for metric in ["ttft_ms", "tpot_ms", "e2e_ms"]:
        for sysname in systems:
            cols.append(f"{METRIC_LABELS[metric]} | {sysname}")
    return df[cols]


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a minimal GitHub-flavored markdown table without extra deps."""

    def esc(v: object) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            s = ""
        else:
            s = str(v)
        return s.replace("|", "\\|").replace("\n", " ")

    headers = [esc(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        cells = [esc(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_lines(summary: pd.DataFrame, out_dir: Path) -> None:
    systems: List[str] = sorted(summary["system"].unique().tolist())

    for metric in METRICS:
        ycol = f"{metric}_p95"
        if ycol not in summary.columns:
            continue

        plt.figure(figsize=(6.5, 4.0))
        for sysname in systems:
            sub = summary[summary["system"] == sysname].sort_values("rate_rps")
            plt.plot(sub["rate_rps"], sub[ycol], marker="o", label=sysname)

        plt.xlabel("rate (req/s)")
        plt.ylabel(f"{metric} p95 (ms)")
        plt.title(f"{metric} p95 vs rate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_p95.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Summarize request-level CSVs and plot metric curves. "
            "Optionally emit a Table-2-style markdown table with baseline-relative percentages."
        )
    )
    p.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        help="Input CSV file(s) or directory(ies). Directories are searched recursively for *.csv",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--stat",
        default="median",
        choices=["mean", "median", "p95"],
        help="Which statistic to use for Table2 output",
    )
    p.add_argument(
        "--baseline-system",
        default="vllm",
        help="System name used as baseline for percentage comparisons",
    )
    p.add_argument(
        "--emit-table2",
        action="store_true",
        help="Write Table2-style outputs (table2.csv, table2.md) into out-dir",
    )
    p.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Optional explicit system ordering (e.g., vllm vllm_chunked molink)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = _iter_csv_paths(args.inputs)
    if len(csv_paths) == 0:
        raise SystemExit("no CSV inputs found")

    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    summary = summarize(df)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_lines(summary, out_dir)

    if args.emit_table2:
        table2 = make_table2(
            summary,
            stat=args.stat,
            baseline_system=args.baseline_system,
            systems=args.systems,
        )
        table2_csv = out_dir / "table2.csv"
        table2_md = out_dir / "table2.md"
        table2.to_csv(table2_csv, index=False)
        table2_md.write_text(df_to_markdown_table(table2), encoding="utf-8")

    extra = ""
    if args.emit_table2:
        extra = ", table2.csv/table2.md"
    print(f"[done] wrote {summary_path}{extra} and plots to {out_dir}")


if __name__ == "__main__":
    main()
