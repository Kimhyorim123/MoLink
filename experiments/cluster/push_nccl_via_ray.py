#!/usr/bin/env python3

import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable

import ray


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@ray.remote
def write_bytes(dst_path: str, payload: bytes) -> dict:
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(payload)
    os.replace(tmp, dst)
    st = dst.stat()
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    return {
        "dst": str(dst),
        "size": st.st_size,
        "sha256": h,
        "hostname": os.uname().nodename,
    }


def iter_alive_node_ips() -> Iterable[str]:
    for n in ray.nodes():
        if n.get("Alive"):
            yield n["NodeManagerAddress"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Distribute libnccl.so.2 to Ray nodes without SSH.")
    ap.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "192.168.79.4:6379"))
    ap.add_argument("--src", required=True, help="Source file on the driver (node5).")
    ap.add_argument("--dst", required=True, help="Destination path on every target node.")
    ap.add_argument("--nodes", nargs="*", default=None, help="Target node IPs. Default: all alive nodes except driver.")
    args = ap.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"src not found: {src}")

    payload = src.read_bytes()
    src_sha = sha256_file(src)
    print(f"src={src} size={len(payload)} sha256={src_sha}")

    driver_ip = ray.util.get_node_ip_address()

    target_ips = args.nodes
    if not target_ips:
        target_ips = [ip for ip in iter_alive_node_ips() if ip != driver_ip]

    if not target_ips:
        raise SystemExit("no target nodes found")

    futures = []
    for ip in target_ips:
        futures.append(
            write_bytes.options(resources={f"node:{ip}": 0.001}).remote(args.dst, payload)
        )

    results = ray.get(futures)
    ok = True
    for r in results:
        same = (r.get("sha256") == src_sha)
        ok = ok and same
        print(f"{r.get('hostname')} dst={r.get('dst')} size={r.get('size')} sha256={r.get('sha256')} match={same}")

    if not ok:
        raise SystemExit("sha256 mismatch on at least one node")


if __name__ == "__main__":
    main()
