#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

import ray


@dataclass(frozen=True)
class PipResult:
    node_ip: str
    ok: bool
    action: str
    stdout_tail: str
    stderr_tail: str
    returncode: int


def _tail(s: str, n: int = 30) -> str:
    lines = s.splitlines()
    if len(lines) <= n:
        return s
    return "\n".join(lines[-n:])


def _node_ips() -> list[str]:
    ips: list[str] = []
    for n in ray.nodes():
        if not n.get("Alive", False):
            continue
        ip = n.get("NodeManagerAddress")
        if ip:
            ips.append(str(ip))
    return sorted(set(ips))


@ray.remote
def _run(cmd: list[str], action: str) -> PipResult:
    node_ip = ray.util.get_node_ip_address()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return PipResult(
        node_ip=node_ip,
        ok=p.returncode == 0,
        action=action,
        stdout_tail=_tail(p.stdout),
        stderr_tail=_tail(p.stderr),
        returncode=p.returncode,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pin vLLM-compatible deps on all Ray nodes (no SSH).")
    ap.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "192.168.79.4:6379"))
    ap.add_argument("--numpy", default="1.26.4")
    ap.add_argument("--pillow", default="10.3.0")
    ap.add_argument("--fsspec", default="2024.9.0")
    ap.add_argument("--transformers", default="4.45.2")
    ap.add_argument("--uninstall-tensorrt-llm", action="store_true")
    ap.add_argument("--pip-check", action="store_true")
    args = ap.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    ips = _node_ips()
    print("[ray] nodes:", ", ".join(ips))

    if not ips:
        raise SystemExit("No Ray nodes found")

    install_pkgs = [
        f"numpy=={args.numpy}",
        f"pillow=={args.pillow}",
        f"fsspec=={args.fsspec}",
        f"transformers=={args.transformers}",
        "cffi",
    ]

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "--upgrade",
        "--force-reinstall",
        *install_pkgs,
    ]

    tasks = []
    for ip in ips:
        tasks.append(
            _run.options(resources={f"node:{ip}": 0.001}).remote(install_cmd, action="pin")
        )
    results: list[PipResult] = ray.get(tasks)

    for r in sorted(results, key=lambda x: x.node_ip):
        print(f"\n[{r.node_ip}] action={r.action} ok={r.ok} rc={r.returncode}")
        if r.stdout_tail:
            print("--- stdout (tail) ---")
            print(r.stdout_tail)
        if r.stderr_tail:
            print("--- stderr (tail) ---")
            print(r.stderr_tail)

    if args.uninstall_tensorrt_llm:
        uninstall_cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "tensorrt-llm"]
        tasks = [
            _run.options(resources={f"node:{ip}": 0.001}).remote(uninstall_cmd, action="uninstall")
            for ip in ips
        ]
        results = ray.get(tasks)
        for r in sorted(results, key=lambda x: x.node_ip):
            print(f"\n[{r.node_ip}] action={r.action} ok={r.ok} rc={r.returncode}")
            if r.stdout_tail:
                print("--- stdout (tail) ---")
                print(r.stdout_tail)
            if r.stderr_tail:
                print("--- stderr (tail) ---")
                print(r.stderr_tail)

    if args.pip_check:
        check_cmd = [sys.executable, "-m", "pip", "check"]
        tasks = [
            _run.options(resources={f"node:{ip}": 0.001}).remote(check_cmd, action="pip_check")
            for ip in ips
        ]
        results = ray.get(tasks)
        for r in sorted(results, key=lambda x: x.node_ip):
            print(f"\n[{r.node_ip}] action={r.action} ok={r.ok} rc={r.returncode}")
            if r.stdout_tail:
                print("--- stdout (tail) ---")
                print(r.stdout_tail)
            if r.stderr_tail:
                print("--- stderr (tail) ---")
                print(r.stderr_tail)


if __name__ == "__main__":
    main()
