#!/usr/bin/env python3

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional

import ray


@dataclass(frozen=True)
class ExecResult:
    node_ip: str
    returncode: int
    stdout: str
    stderr: str


@ray.remote
def _exec_on_node(cmd: str, timeout_s: int) -> ExecResult:
    import ray as _ray

    node_ip = _ray.util.get_node_ip_address()
    try:
        # Use bash -lc to honor PATH and allow compound commands.
        p = subprocess.run(
            ["/bin/bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
        return ExecResult(
            node_ip=node_ip,
            returncode=int(p.returncode),
            stdout=p.stdout[-6000:],
            stderr=p.stderr[-6000:],
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            node_ip=node_ip,
            returncode=124,
            stdout=(e.stdout or "")[-6000:],
            stderr=(e.stderr or f"timeout after {timeout_s}s")[-6000:],
        )
    except Exception as e:
        return ExecResult(node_ip=node_ip, returncode=1, stdout="", stderr=repr(e))


def _discover_node_ips() -> list[str]:
    ips: list[str] = []
    try:
        for n in ray.nodes():
            if not n.get("Alive", False):
                continue
            ip = n.get("NodeManagerAddress")
            if ip:
                ips.append(str(ip))
    except Exception:
        ips = []

    if not ips:
        resources = ray.cluster_resources()
        for k in resources.keys():
            if isinstance(k, str) and k.startswith("node:") and k != "node:__internal_head__":
                ips.append(k.split(":", 1)[1])

    return sorted(set(ips))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute a shell command on all Ray nodes (no SSH).")
    ap.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "192.168.79.4:6379"),
        help="Ray head address, e.g. 192.168.79.4:6379",
    )
    ap.add_argument(
        "--cmd",
        required=True,
        help="Command to run on each node. Wrap in quotes. Example: --cmd 'hostname; nvidia-smi -L'",
    )
    ap.add_argument(
        "--nodes",
        default=None,
        help="Optional comma-separated node IP allowlist. Example: 192.168.79.4,192.168.79.9",
    )
    ap.add_argument("--timeout-s", type=int, default=600, help="Per-node timeout")
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit non-zero if any node returns non-zero",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    if args.nodes:
        node_ips = [x.strip() for x in args.nodes.split(",") if x.strip()]
    else:
        node_ips = _discover_node_ips()

    if not node_ips:
        raise SystemExit("no alive Ray nodes found")

    print("[ray] nodes:", ", ".join(node_ips))
    print("[exec] cmd:", shlex.quote(args.cmd))

    tasks = []
    for ip in node_ips:
        tasks.append(_exec_on_node.options(resources={f"node:{ip}": 0.001}).remote(args.cmd, args.timeout_s))

    results: list[ExecResult] = ray.get(tasks)

    any_fail = False
    for r in sorted(results, key=lambda x: x.node_ip):
        ok = r.returncode == 0
        any_fail |= not ok
        print(f"--- node {r.node_ip} rc={r.returncode} ---")
        if r.stdout.strip():
            print(r.stdout.rstrip())
        if r.stderr.strip():
            print("[stderr]")
            print(r.stderr.rstrip())

    if args.fail_fast and any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
