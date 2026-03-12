#!/usr/bin/env python3

import argparse
import os
import socket
import time
from dataclasses import dataclass

import ray


@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    node_ip: str
    hostname: str
    torch_version: str
    torch_cuda: str
    nccl_pkg: str


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


@ray.remote
def _node_probe() -> NodeInfo:
    import importlib

    import torch

    hostname = socket.gethostname()
    node_ip = ray.util.get_node_ip_address()
    node_id = ray.get_runtime_context().get_node_id()

    nccl_pkg_ver = "<not-installed>"
    try:
        md = importlib.metadata  # type: ignore[attr-defined]
    except Exception:
        import importlib_metadata as md  # type: ignore[no-redef]
    try:
        nccl_pkg_ver = md.version("nvidia-nccl-cu12")
    except Exception:
        pass

    return NodeInfo(
        node_id=node_id,
        node_ip=node_ip,
        hostname=hostname,
        torch_version=torch.__version__,
        torch_cuda=str(torch.version.cuda),
        nccl_pkg=nccl_pkg_ver,
    )


@ray.remote(num_gpus=1)
def _nccl_worker(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    timeout_s: int,
) -> str:
    import datetime
    import os
    import socket

    import torch
    import torch.distributed as dist

    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_NET", "Socket")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", os.environ.get("NCCL_SOCKET_IFNAME", "enp7s0"))

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout_s),
    )

    x = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    dist.barrier()
    result = float(x.item())

    dist.destroy_process_group()

    host = socket.gethostname()
    return f"rank={rank} host={host} all_reduce={result}"


def _format_node_infos(infos: list[NodeInfo]) -> str:
    lines = []
    for info in sorted(infos, key=lambda x: (x.node_ip, x.hostname)):
        lines.append(
            f"- {info.hostname} ({info.node_ip}) torch={info.torch_version} cuda={info.torch_cuda} nvidia-nccl-cu12={info.nccl_pkg}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "192.168.79.4:6379"))
    ap.add_argument("--world-size", type=int, default=3)
    ap.add_argument("--timeout-s", type=int, default=180)
    ap.add_argument("--master-addr", default="192.168.79.4")
    ap.add_argument("--master-port", type=int, default=0)
    args = ap.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    resources = ray.cluster_resources()
    print("[ray] cluster_resources:", resources)

    # Probe all nodes we can schedule on.
    num_nodes = int(resources.get("node", 1)) if "node" in resources else None
    probe_tasks = []
    if num_nodes is None:
        probe_tasks = [_node_probe.remote()]
    else:
        for _ in range(num_nodes):
            probe_tasks.append(_node_probe.remote())
    infos = ray.get(probe_tasks)
    # Deduplicate by node_id
    uniq: dict[str, NodeInfo] = {i.node_id: i for i in infos}
    infos = list(uniq.values())
    print("[probe] nodes:\n" + _format_node_infos(infos))

    if int(resources.get("GPU", 0)) < args.world_size:
        raise SystemExit(f"Need >= {args.world_size} GPUs in Ray, but got {resources.get('GPU', 0)}")

    master_port = args.master_port or _pick_free_port()
    print(f"[nccl] starting all-reduce world_size={args.world_size} master={args.master_addr}:{master_port}")

    # Launch N workers; Ray should place them across GPUs/nodes.
    tasks = [
        _nccl_worker.remote(
            rank=r,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=master_port,
            timeout_s=args.timeout_s,
        )
        for r in range(args.world_size)
    ]

    started = time.time()
    results = ray.get(tasks)
    elapsed = time.time() - started
    print("[nccl] results:")
    for line in results:
        print(" ", line)
    expected = (args.world_size * (args.world_size + 1)) / 2.0
    print(f"[nccl] expected sum per rank = {expected}")
    print(f"[nccl] OK (elapsed {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
