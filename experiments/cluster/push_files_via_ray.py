#!/usr/bin/env python3

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import ray


@dataclass(frozen=True)
class PushResult:
    node_ip: str
    wrote_path: str
    bytes_written: int
    sha256: str
    ok: bool
    error: str | None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@ray.remote
def _write_file_on_node(dst_path: str, data: bytes, mode: int | None) -> PushResult:
    try:
        import ray as _ray
        node_ip = _ray.util.get_node_ip_address()
        p = Path(dst_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        if mode is not None:
            os.chmod(p, mode)
        sha = _sha256_bytes(data)
        return PushResult(
            node_ip=node_ip,
            wrote_path=str(p),
            bytes_written=len(data),
            sha256=sha,
            ok=True,
            error=None,
        )
    except Exception as e:
        node_ip = "<unknown>"
        try:
            import ray as _ray

            node_ip = _ray.util.get_node_ip_address()
        except Exception:
            pass
        return PushResult(
            node_ip=node_ip,
            wrote_path=dst_path,
            bytes_written=0,
            sha256="",
            ok=False,
            error=repr(e),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Push files to all Ray nodes (no SSH).")
    ap.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "192.168.79.4:6379"),
        help="Ray head address, e.g. 192.168.79.4:6379",
    )
    ap.add_argument("--src", action="append", required=True, help="Source file path (repeatable)")
    ap.add_argument(
        "--dst",
        action="append",
        required=True,
        help="Destination file path on every node (repeatable, must match --src count)",
    )
    ap.add_argument(
        "--mode",
        default="0755",
        help="chmod mode for pushed files (octal as string, default 0755). Use 'keep' to keep src mode.",
    )
    args = ap.parse_args()

    if len(args.src) != len(args.dst):
        raise SystemExit("--src and --dst must have the same count")

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    resources = ray.cluster_resources()
    print("[ray] cluster_resources:", resources)

    # Discover alive nodes and their IPs.
    node_ips: list[str] = []
    try:
        for n in ray.nodes():
            if not n.get("Alive", False):
                continue
            ip = n.get("NodeManagerAddress")
            if ip:
                node_ips.append(str(ip))
    except Exception:
        node_ips = []

    # Fallback: parse cluster_resources keys like "node:192.168.79.9".
    if not node_ips:
        for k in resources.keys():
            if isinstance(k, str) and k.startswith("node:") and k != "node:__internal_head__":
                node_ips.append(k.split(":", 1)[1])

    # Last resort: at least push to current node.
    node_ips = sorted(set(node_ips))
    if not node_ips:
        node_ips = [ray.util.get_node_ip_address()]

    print("[ray] alive nodes:", ", ".join(node_ips))

    for src, dst in zip(args.src, args.dst, strict=True):
        src_path = Path(src)
        data = src_path.read_bytes()
        src_sha = _sha256_bytes(data)

        if args.mode == "keep":
            mode = src_path.stat().st_mode & 0o777
        else:
            mode = int(args.mode, 8)

        print(f"[push] {src} -> {dst} (sha256={src_sha}, mode={oct(mode)})")
        tasks = []
        for ip in node_ips:
            # Force execution on each specific node.
            tasks.append(
                _write_file_on_node.options(resources={f"node:{ip}": 0.001}).remote(dst, data, mode)
            )
        results: list[PushResult] = ray.get(tasks)

        # Deduplicate by (node_ip, wrote_path)
        uniq: dict[tuple[str, str], PushResult] = {(r.node_ip, r.wrote_path): r for r in results}
        for r in sorted(uniq.values(), key=lambda x: x.node_ip):
            match = r.sha256 == src_sha
            print(
                f"  - node={r.node_ip} ok={r.ok} match={match} bytes={r.bytes_written} path={r.wrote_path}"
                + (f" err={r.error}" if r.error else "")
            )


if __name__ == "__main__":
    main()
