#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["ttft", "tpot", "e2e"]
STATS = ["p50", "p95"]


def summarize_request_csvs(jit_dir: Path, scenario: str) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(jit_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df = df[df["status"] == "ok"].copy()
        if df.empty:
            continue
        system = str(df["system"].iloc[0])
        rate = float(df["rate_rps"].iloc[0])
        rows.append(
            {
                "scenario": scenario,
                "system": system,
                "rate": rate,
                "ttft_p50": float(np.percentile(df["ttft_ms"].dropna().astype(float), 50)),
                "ttft_p95": float(np.percentile(df["ttft_ms"].dropna().astype(float), 95)),
                "tpot_p50": float(np.percentile(df["tpot_ms"].dropna().astype(float), 50)),
                "tpot_p95": float(np.percentile(df["tpot_ms"].dropna().astype(float), 95)),
                "e2e_p50": float(np.percentile(df["e2e_ms"].dropna().astype(float), 50)),
                "e2e_p95": float(np.percentile(df["e2e_ms"].dropna().astype(float), 95)),
            }
        )
    if not rows:
        raise RuntimeError(f"no request CSVs found in {jit_dir}")
    return pd.DataFrame(rows).sort_values(["system", "rate"])


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    systems = list(dict.fromkeys(summary["system"].tolist()))
    for metric in METRICS:
        for stat in STATS:
            col = f"{metric}_{stat}"
            plt.figure(figsize=(6.5, 4.0))
            for system in systems:
                sub = summary[summary["system"] == system].sort_values("rate")
                plt.plot(sub["rate"], sub[col], marker="o", label=system)
            plt.xlabel("rate (req/s)")
            plt.ylabel(f"{metric.upper()} {stat} (ms)")
            plt.title(f"{metric.upper()} {stat} vs rate")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{metric}_{stat}.png", dpi=200)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge existing baseline/fixed summary with new JIT raw CSVs.")
    parser.add_argument("--existing-summary", required=True, help="Existing summary.csv with molink_baseline and molink_modified rows")
    parser.add_argument("--jit-dir", required=True, help="Directory containing raw molink_jit_rate*.csv files")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.read_csv(args.existing_summary)
    existing = existing[existing["scenario"] == args.scenario].copy()
    existing["system"] = existing["system"].replace({"molink_modified": "molink_fixed2mb"})

    jit = summarize_request_csvs(Path(args.jit_dir), args.scenario)
    jit["system"] = jit["system"].replace({"molink_jit": "molink_jit"})

    combined = pd.concat([existing, jit], ignore_index=True)
    combined = combined.sort_values(["rate", "system"]).reset_index(drop=True)
    combined.to_csv(out_dir / "summary.csv", index=False)

    plot_summary(combined, out_dir)


if __name__ == "__main__":
    main()
