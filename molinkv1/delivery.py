"""
Async tensor delivery for MoLink cross-node pipeline parallelism.

This module provides async delivery of intermediate tensors and sampler outputs
across nodes using gRPC. The delivery is done in a separate process to overlap
computation with communication.
"""

import asyncio
import io
import multiprocessing as mp
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from queue import Empty
from typing import Any, Deque, Dict, Optional, Set, Tuple

import cloudpickle
import grpc.aio as aio
import torch
from vllm.logger import init_logger

from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.utils import get_grpc_options, serialize_metadata

logger = init_logger(__name__)


def _now_monotonic() -> float:
    return time.monotonic()


def _serialize_prefill_envelope(
    intermediate_tensors_cpu: Dict[str, torch.Tensor],
    scheduler_output_bytes: bytes,
    grpc_metadata: Dict[str, Any],
    virtual_engine: int,
) -> bytes:
    tensor_bytes = {}
    for key, tensor in intermediate_tensors_cpu.items():
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        tensor_bytes[key] = buffer.getvalue()

    envelope = {
        "scheduler_output": scheduler_output_bytes,
        "intermediate_tensors": tensor_bytes,
        "grpc_metadata": grpc_metadata,
        "virtual_engine": virtual_engine,
    }
    return cloudpickle.dumps(envelope)


@dataclass
class DeliveryItem:
    push_type: str
    phase: str
    virtual_engine: int
    target_server: str
    enqueue_ts: float
    intermediate_tensors_cpu: Optional[Dict[str, torch.Tensor]] = None
    scheduler_output_bytes: Optional[bytes] = None
    grpc_metadata: Optional[Dict[str, Any]] = None
    output_bytes: Optional[bytes] = None
    transfer_id: Optional[str] = None
    serialized_payload: Optional[bytes] = None
    total_bytes: int = 0
    offset: int = 0
    left_bytes: int = 0
    is_chunked: bool = False
    scheduled_ts: Optional[float] = None
    launch_ts: Optional[float] = None


class TensorDeliveryProcess(mp.Process):
    """Background process for async tensor delivery.

    This process handles serialization and transmission of intermediate
    tensors and sampler outputs to other nodes in the pipeline. Running
    in a separate process allows overlapping of communication with
    computation on the main process.
    """

    def __init__(
        self,
        max_message_size_mb: int = 200,
        max_waiting_weight: int = 30,
        decode_inflight_limit: int = 4,
        prefill_inflight_limit: int = 1,
        head_inflight_limit: int = 1,
        chunk_size_bytes: int = 2 * 1024 * 1024,
        min_chunk_size_bytes: int = 256 * 1024,
        max_chunk_size_bytes: int = 8 * 1024 * 1024,
    ):
        """Initialize the delivery process.

        Args:
            max_message_size_mb: Maximum gRPC message size in MB.
            max_waiting_weight: Number of decode-priority selections to allow
                before forcing one prefill transmission.
            decode_inflight_limit: Maximum number of concurrent decode send tasks.
            prefill_inflight_limit: Maximum number of concurrent prefill send tasks.
            head_inflight_limit: Maximum number of concurrent head-output send tasks.
            chunk_size_bytes: Fixed fallback chunk size for prefill transmission.
            min_chunk_size_bytes: Minimum adaptive chunk size.
            max_chunk_size_bytes: Maximum adaptive chunk size.
        """
        super().__init__(daemon=True, name="MolinkTensorDelivery")

        self.max_message_size_mb = max_message_size_mb
        self.max_waiting_weight = max_waiting_weight
        self.decode_inflight_limit = decode_inflight_limit
        self.prefill_inflight_limit = prefill_inflight_limit
        self.head_inflight_limit = head_inflight_limit
        self.chunk_size_bytes = chunk_size_bytes
        self.min_chunk_size_bytes = min_chunk_size_bytes
        self.max_chunk_size_bytes = max_chunk_size_bytes

        # Queue for pending deliveries from the main process.
        self.delivery_queue: mp.Queue = mp.Queue(maxsize=100)

        # Shutdown event
        self._shutdown = mp.Event()

    def run(self):
        """Main loop for the delivery process."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Cache for gRPC channels and stubs
        channel_cache: Dict[str, aio.Channel] = {}
        stub_cache: Dict[str, molink_pb2_grpc.MolinkServiceStub] = {}

        decode_queue: Deque[DeliveryItem] = deque()
        prefill_queue: Deque[DeliveryItem] = deque()
        head_queue: Deque[DeliveryItem] = deque()
        waiting_weight = 0
        head_tasks: Set[asyncio.Task] = set()
        decode_tasks: Set[asyncio.Task] = set()
        prefill_tasks: Set[asyncio.Task] = set()

        runtime_stats: Dict[str, Optional[float]] = {
            "decode_bytes_per_ms_ema": None,
            "prefill_bytes_per_ms_ema": None,
            "decode_send_ms_ema": None,
            "prefill_send_ms_ema": None,
            "decode_arrival_gap_ms_ema": None,
            "last_decode_arrival_ts": None,
        }

        def get_stub(address: str) -> molink_pb2_grpc.MolinkServiceStub:
            if address not in stub_cache:
                channel = aio.insecure_channel(
                    address, options=get_grpc_options(self.max_message_size_mb)
                )
                channel_cache[address] = channel
                stub_cache[address] = molink_pb2_grpc.MolinkServiceStub(channel)
            return stub_cache[address]

        def _ema_update(current: Optional[float], value: float, alpha: float = 0.2) -> float:
            if current is None:
                return value
            return (1.0 - alpha) * current + alpha * value

        def _update_send_stats(phase: str, sent_bytes: int, send_ms: float) -> None:
            if send_ms <= 0:
                return
            bytes_per_ms = sent_bytes / send_ms
            if phase == "prefill":
                runtime_stats["prefill_send_ms_ema"] = _ema_update(
                    runtime_stats["prefill_send_ms_ema"], send_ms
                )
                runtime_stats["prefill_bytes_per_ms_ema"] = _ema_update(
                    runtime_stats["prefill_bytes_per_ms_ema"], bytes_per_ms
                )
            else:
                runtime_stats["decode_send_ms_ema"] = _ema_update(
                    runtime_stats["decode_send_ms_ema"], send_ms
                )
                runtime_stats["decode_bytes_per_ms_ema"] = _ema_update(
                    runtime_stats["decode_bytes_per_ms_ema"], bytes_per_ms
                )

        def _record_decode_arrival(enqueue_ts: float) -> None:
            last_arrival = runtime_stats.get("last_decode_arrival_ts")
            if last_arrival is not None:
                gap_ms = max((enqueue_ts - last_arrival) * 1000.0, 0.0)
                runtime_stats["decode_arrival_gap_ms_ema"] = _ema_update(
                    runtime_stats["decode_arrival_gap_ms_ema"], gap_ms
                )
            runtime_stats["last_decode_arrival_ts"] = enqueue_ts

        def _estimate_available_time_ms(item: DeliveryItem, tc: float) -> tuple[float, str, Dict[str, Any]]:
            trace = (item.grpc_metadata or {}).get("jit_runtime_trace", {})
            ts = trace.get("decode_window_start_ts") or trace.get("last_decode_start_ts")
            tf = trace.get("decode_window_finish_ts") or trace.get("last_decode_finish_ts")
            last_decode_finish_ts = trace.get("last_decode_finish_ts")
            decode_duration_ms = trace.get("last_decode_duration_ms")
            if decode_duration_ms is None:
                decode_duration_ms = trace.get("decode_window_duration_ms")
            if decode_duration_ms is None:
                decode_duration_ms = runtime_stats.get("decode_send_ms_ema")
            if decode_duration_ms is None:
                decode_duration_ms = 5.0

            bandwidth_bytes_per_ms = (
                runtime_stats.get("prefill_bytes_per_ms_ema")
                or runtime_stats.get("decode_bytes_per_ms_ema")
                or max(self.chunk_size_bytes / max(decode_duration_ms, 1.0), 1.0)
            )
            decode_tokens = trace.get("decode_token_count") or trace.get("last_decode_tokens") or 1
            token_bytes = 2 * int(decode_tokens)
            tm_ms = runtime_stats.get("decode_send_ms_ema")
            if tm_ms is None:
                tm_ms = max(token_bytes / max(bandwidth_bytes_per_ms, 1.0), 0.1)
            to_ms = decode_duration_ms
            info: Dict[str, Any] = {
                "tc": tc,
                "ts": ts,
                "tf": tf,
                "tp": last_decode_finish_ts,
                "Td_ms": decode_duration_ms,
                "Tm_ms": tm_ms,
                "To_ms": to_ms,
                "decode_tokens": decode_tokens,
                "bandwidth_bytes_per_ms": bandwidth_bytes_per_ms,
                "decode_q_len": len(decode_queue),
                "prefill_q_len": len(prefill_queue),
                "head_q_len": len(head_queue),
                "current_exec_start_ts": trace.get("current_exec_start_ts"),
                "current_exec_finish_ts": trace.get("current_exec_finish_ts"),
                "last_decode_finish_ts": last_decode_finish_ts,
                "trace_seq": trace.get("trace_seq"),
            }

            epsilon_s = 0.002
            if ts is not None and abs(tc - ts) <= epsilon_s:
                ta_ms = max(decode_duration_ms, 0.1)
                info["Ta_ms"] = ta_ms
                info["case_reason"] = "tc_almost_equals_ts"
                return ta_ms, "case1", info

            if ts is not None and tf is not None and tc > ts and tc < tf:
                ta_ms = max((tf - tc) * 1000.0, 0.1)
                info["Ta_ms"] = ta_ms
                info["case_reason"] = "tc_inside_current_decode_window"
                return ta_ms, "case2", info

            tp = last_decode_finish_ts
            if tp is None:
                tp = tf if tf is not None else tc
            decode_gap_ms = runtime_stats.get("decode_arrival_gap_ms_ema") or 0.0
            predicted_ts = tp + decode_gap_ms / 1000.0
            predicted_tf = predicted_ts + (to_ms + tm_ms) / 1000.0
            info["predicted_ts"] = predicted_ts
            ta_ms = max((predicted_tf - tc) * 1000.0, 0.1)
            info["predicted_tf"] = predicted_tf
            info["Ta_ms"] = ta_ms
            if ts is None:
                info["case_reason"] = "no_decode_window_available"
            elif tf is None:
                info["case_reason"] = "decode_start_without_finish"
            elif tc <= ts:
                info["case_reason"] = "tc_before_decode_window"
            else:
                info["case_reason"] = "tc_after_decode_window"
            return ta_ms, "case3", info

        def _determine_chunk_size(item: DeliveryItem) -> tuple[int, str, Dict[str, Any]]:
            tc = _now_monotonic()
            ta_ms, ta_case, info = _estimate_available_time_ms(item, tc)
            if info.get("case_reason") == "no_decode_window_available":
                chunk_size = min(self.chunk_size_bytes, item.left_bytes or item.total_bytes)
                info["fallback"] = "fixed_chunk_no_decode_window"
                info["selected_chunk_size"] = chunk_size
                return chunk_size, ta_case, info

            bandwidth_bytes_per_ms = info["bandwidth_bytes_per_ms"]
            dynamic_size = int(max(bandwidth_bytes_per_ms * ta_ms * 0.9, 1.0))
            chunk_size = min(item.left_bytes or item.total_bytes, dynamic_size)
            chunk_size = max(self.min_chunk_size_bytes, chunk_size)
            chunk_size = min(self.max_chunk_size_bytes, chunk_size)
            chunk_size = min(chunk_size, item.left_bytes or item.total_bytes)
            if chunk_size <= 0:
                chunk_size = min(self.chunk_size_bytes, item.left_bytes or item.total_bytes)
                info["fallback"] = "fixed_chunk"
            info["selected_chunk_size"] = chunk_size
            return chunk_size, ta_case, info

        async def deliver_intermediate_tensors(item: DeliveryItem):
            """Deliver intermediate tensors to the next pipeline stage."""
            try:
                assert item.intermediate_tensors_cpu is not None
                assert item.scheduler_output_bytes is not None
                assert item.grpc_metadata is not None

                send_start_ts = _now_monotonic()

                grpc_tensors = molink_pb2.IntermediateTensors()
                payload_bytes = len(item.scheduler_output_bytes)
                for key, tensor in item.intermediate_tensors_cpu.items():
                    buffer = io.BytesIO()
                    torch.save(tensor, buffer)
                    tensor_bytes = buffer.getvalue()
                    payload_bytes += len(tensor_bytes)
                    grpc_tensors.tensors.append(
                        molink_pb2.TensorEntry(key=key, tensor_data=tensor_bytes)
                    )

                request = molink_pb2.GrpcRequestData(
                    scheduler_output=item.scheduler_output_bytes,
                    intermediate_tensors=grpc_tensors,
                    grpc_metadata=serialize_metadata(item.grpc_metadata),
                    virtual_engine=item.virtual_engine,
                )

                stub = get_stub(item.target_server)
                response = await stub.PushIntermediateTensors(request)
                send_end_ts = _now_monotonic()
                _update_send_stats(item.phase, payload_bytes, (send_end_ts - send_start_ts) * 1000.0)
                logger.info(
                    "[MoLink][VE%s][DELIVERY] phase=%s target=%s queue_wait_ms=%.3f "
                    "schedule_wait_ms=%.3f launch_wait_ms=%.3f send_ms=%.3f "
                    "payload_bytes=%d tensors=%d response=%s",
                    item.virtual_engine,
                    item.phase,
                    item.target_server,
                    (send_start_ts - item.enqueue_ts) * 1000.0,
                    ((item.scheduled_ts or send_start_ts) - item.enqueue_ts) * 1000.0,
                    (send_start_ts - (item.launch_ts or send_start_ts)) * 1000.0,
                    (send_end_ts - send_start_ts) * 1000.0,
                    payload_bytes,
                    len(item.intermediate_tensors_cpu),
                    response.res,
                )

            except Exception as e:
                logger.error(
                    f"[MoLink][DELIVERY] Error delivering intermediate tensors: {e}"
                )
                traceback.print_exc()

        async def deliver_sampler_output(item: DeliveryItem):
            """Deliver sampler output to the head node."""
            try:
                assert item.output_bytes is not None
                request = molink_pb2.SamplerOutput(
                    output_data=item.output_bytes,
                    virtual_engine=item.virtual_engine,
                )

                stub = get_stub(item.target_server)
                await stub.PushSamplerOutput(request)

            except Exception as e:
                logger.error(f"[MoLink][DELIVERY] Error delivering sampler output: {e}")
                traceback.print_exc()

        def _build_chunked_prefill_payload(item: DeliveryItem) -> None:
            assert item.intermediate_tensors_cpu is not None
            assert item.scheduler_output_bytes is not None
            assert item.grpc_metadata is not None

            payload = _serialize_prefill_envelope(
                item.intermediate_tensors_cpu,
                item.scheduler_output_bytes,
                item.grpc_metadata,
                item.virtual_engine,
            )
            item.transfer_id = uuid.uuid4().hex
            item.serialized_payload = payload
            item.total_bytes = len(payload)
            item.offset = 0
            item.left_bytes = len(payload)
            item.is_chunked = True
            trace = (item.grpc_metadata or {}).get("jit_runtime_trace", {})
            logger.info(
                "[MoLink][VE%s][DELIVERY] chunk_prepare phase=%s transfer_id=%s total_bytes=%d fallback_chunk_size=%d target=%s trace_seq=%s decode_tokens=%s",
                item.virtual_engine,
                item.phase,
                item.transfer_id,
                item.total_bytes,
                self.chunk_size_bytes,
                item.target_server,
                trace.get("trace_seq"),
                trace.get("decode_token_count"),
            )

        def _take_next_chunk(item: DeliveryItem, chunk_size: int) -> tuple[bytes, int, bool]:
            assert item.serialized_payload is not None
            start = item.offset
            end = min(start + chunk_size, item.total_bytes)
            chunk = item.serialized_payload[start:end]
            item.offset = end
            item.left_bytes = item.total_bytes - item.offset
            is_last_chunk = item.left_bytes == 0
            return chunk, start, is_last_chunk

        async def deliver_prefill_chunk(item: DeliveryItem):
            try:
                assert item.grpc_metadata is not None
                prepare_ms = 0.0
                if item.serialized_payload is None:
                    prepare_start_ts = _now_monotonic()
                    _build_chunked_prefill_payload(item)
                    prepare_ms = (_now_monotonic() - prepare_start_ts) * 1000.0

                chunk_size, ta_case, ta_info = _determine_chunk_size(item)
                chunk_data, chunk_offset, is_last_chunk = _take_next_chunk(item, chunk_size)
                send_start_ts = _now_monotonic()
                request = molink_pb2.GrpcRequestData(
                    grpc_metadata=serialize_metadata(item.grpc_metadata),
                    virtual_engine=item.virtual_engine,
                    transfer_id=item.transfer_id or "",
                    chunk_data=chunk_data,
                    chunk_offset=chunk_offset,
                    total_bytes=item.total_bytes,
                    is_chunked=True,
                    is_last_chunk=is_last_chunk,
                )
                stub = get_stub(item.target_server)
                response = await stub.PushIntermediateTensors(request)
                send_end_ts = _now_monotonic()
                send_ms = (send_end_ts - send_start_ts) * 1000.0
                _update_send_stats(item.phase, len(chunk_data), send_ms)
                logger.info(
                    "[MoLink][VE%s][DELIVERY] chunk_send phase=%s transfer_id=%s target=%s offset=%d sent=%d left=%d queue_wait_ms=%.3f schedule_wait_ms=%.3f launch_wait_ms=%.3f prepare_ms=%.3f send_ms=%.3f response=%s ta_case=%s fallback=%s Ta_ms=%.3f Td_ms=%.3f Tm_ms=%.3f To_ms=%.3f chunk_size=%d",
                    item.virtual_engine,
                    item.phase,
                    item.transfer_id,
                    item.target_server,
                    chunk_offset,
                    len(chunk_data),
                    item.left_bytes,
                    (send_start_ts - item.enqueue_ts) * 1000.0,
                    ((item.scheduled_ts or send_start_ts) - item.enqueue_ts) * 1000.0,
                    (send_start_ts - (item.launch_ts or send_start_ts)) * 1000.0,
                    prepare_ms,
                    send_ms,
                    response.res,
                    ta_case,
                    ta_info.get("fallback"),
                    float(ta_info.get("Ta_ms", 0.0)),
                    float(ta_info.get("Td_ms", 0.0)),
                    float(ta_info.get("Tm_ms", 0.0)),
                    float(ta_info.get("To_ms", 0.0)),
                    chunk_size,
                )
                logger.info(
                    "[MoLink][VE%s][DELIVERY][JIT] ta_case=%s reason=%s fallback=%s tc=%.6f ts=%s tf=%s predicted_tf=%s "
                    "predicted_ts=%s trace_seq=%s current_exec_start_ts=%s current_exec_finish_ts=%s last_decode_finish_ts=%s "
                    "decode_q=%s prefill_q=%s head_q=%s decode_tokens=%s bw_bytes_per_ms=%.3f",
                    item.virtual_engine,
                    ta_case,
                    ta_info.get("case_reason"),
                    ta_info.get("fallback"),
                    float(ta_info.get("tc", 0.0)),
                    ta_info.get("ts"),
                    ta_info.get("tf"),
                    ta_info.get("predicted_tf"),
                    ta_info.get("predicted_ts"),
                    ta_info.get("trace_seq"),
                    ta_info.get("current_exec_start_ts"),
                    ta_info.get("current_exec_finish_ts"),
                    ta_info.get("last_decode_finish_ts"),
                    ta_info.get("decode_q_len"),
                    ta_info.get("prefill_q_len"),
                    ta_info.get("head_q_len"),
                    ta_info.get("decode_tokens"),
                    float(ta_info.get("bandwidth_bytes_per_ms", 0.0)),
                )
                if item.left_bytes > 0:
                    item.enqueue_ts = _now_monotonic()
                    prefill_queue.append(item)
                    logger.info(
                        "[MoLink][VE%s][DELIVERY] chunk_requeue phase=%s transfer_id=%s left=%d prefill_q=%d",
                        item.virtual_engine,
                        item.phase,
                        item.transfer_id,
                        item.left_bytes,
                        len(prefill_queue),
                    )
                else:
                    logger.info(
                        "[MoLink][VE%s][DELIVERY] chunk_done phase=%s transfer_id=%s total_bytes=%d",
                        item.virtual_engine,
                        item.phase,
                        item.transfer_id,
                        item.total_bytes,
                    )
            except Exception as e:
                logger.error(
                    f"[MoLink][DELIVERY] Error delivering prefill chunk: {e}"
                )
                traceback.print_exc()

        def ingest_item(item: DeliveryItem) -> None:
            if item.push_type == "head":
                head_queue.append(item)
                logger.info(
                    "[MoLink][VE%s][DELIVERY] classify push=head target=%s head_q=%d",
                    item.virtual_engine,
                    item.target_server,
                    len(head_queue),
                )
                return

            if item.phase in {"decode", "mixed"}:
                _record_decode_arrival(item.enqueue_ts)
                decode_queue.append(item)
                queue_name = "decode"
                queue_len = len(decode_queue)
            else:
                prefill_queue.append(item)
                queue_name = "prefill"
                queue_len = len(prefill_queue)

            trace = (item.grpc_metadata or {}).get("jit_runtime_trace", {})
            logger.info(
                "[MoLink][VE%s][DELIVERY] classify phase=%s route=%s target=%s decode_q=%d prefill_q=%d selected_q_len=%d trace_seq=%s last_decode_finish_ts=%s",
                item.virtual_engine,
                item.phase,
                queue_name,
                item.target_server,
                len(decode_queue),
                len(prefill_queue),
                queue_len,
                trace.get("trace_seq"),
                trace.get("last_decode_finish_ts"),
            )

        async def fetch_item(timeout: float) -> Optional[DeliveryItem]:
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: self.delivery_queue.get(timeout=timeout),
                )
            except Empty:
                return None
            except Exception:
                return None

        def drain_nowait() -> None:
            while True:
                try:
                    item = self.delivery_queue.get_nowait()
                except Empty:
                    break
                except Exception:
                    break
                ingest_item(item)

        def choose_next_item() -> Tuple[Optional[DeliveryItem], Optional[str], int]:
            nonlocal waiting_weight

            if head_queue:
                return head_queue.popleft(), "head_immediate", waiting_weight

            if decode_queue and prefill_queue:
                if waiting_weight < self.max_waiting_weight:
                    waiting_weight += 1
                    return decode_queue.popleft(), "decode_priority", waiting_weight
                item = prefill_queue.popleft()
                old_weight = waiting_weight
                waiting_weight = 0
                return item, "prefill_starvation_relief", old_weight

            if decode_queue:
                return decode_queue.popleft(), "decode_only", waiting_weight

            if prefill_queue:
                old_weight = waiting_weight
                waiting_weight = 0
                return prefill_queue.popleft(), "prefill_only", old_weight

            return None, None, waiting_weight

        def task_phase(task: asyncio.Task) -> str:
            return getattr(task, "_molink_phase", "unknown")

        def task_reason(task: asyncio.Task) -> str:
            return getattr(task, "_molink_reason", "unknown")

        def task_virtual_engine(task: asyncio.Task) -> int:
            return getattr(task, "_molink_virtual_engine", -1)

        def task_target(task: asyncio.Task) -> str:
            return getattr(task, "_molink_target", "unknown")

        def task_class(task: asyncio.Task) -> str:
            return getattr(task, "_molink_class", "unknown")

        def inflight_total() -> int:
            return len(head_tasks) + len(decode_tasks) + len(prefill_tasks)

        def launch_item(item: DeliveryItem, reason: str) -> None:
            item.launch_ts = _now_monotonic()
            if item.push_type == "head":
                coro = deliver_sampler_output(item)
                task_set = head_tasks
                traffic_class = "head"
            elif item.phase == "prefill":
                coro = deliver_prefill_chunk(item)
                task_set = prefill_tasks
                traffic_class = "prefill"
            else:
                coro = deliver_intermediate_tensors(item)
                task_set = decode_tasks
                traffic_class = "decode"

            task = loop.create_task(coro)
            task._molink_phase = item.phase
            task._molink_reason = reason
            task._molink_virtual_engine = item.virtual_engine
            task._molink_target = item.target_server
            task._molink_class = traffic_class
            task_set.add(task)
            logger.info(
                "[MoLink][VE%s][DELIVERY] launch reason=%s phase=%s target=%s inflight_total=%d decode_inflight=%d prefill_inflight=%d head_inflight=%d decode_q=%d prefill_q=%d head_q=%d",
                item.virtual_engine,
                reason,
                item.phase,
                item.target_server,
                inflight_total(),
                len(decode_tasks),
                len(prefill_tasks),
                len(head_tasks),
                len(decode_queue),
                len(prefill_queue),
                len(head_queue),
            )

        def reap_finished_tasks() -> None:
            done = [
                task
                for task in (head_tasks | decode_tasks | prefill_tasks)
                if task.done()
            ]
            for task in done:
                cls = task_class(task)
                if cls == "head":
                    head_tasks.discard(task)
                elif cls == "decode":
                    decode_tasks.discard(task)
                elif cls == "prefill":
                    prefill_tasks.discard(task)
                try:
                    task.result()
                    logger.info(
                        "[MoLink][VE%s][DELIVERY] complete reason=%s phase=%s target=%s inflight_total=%d decode_inflight=%d prefill_inflight=%d head_inflight=%d",
                        task_virtual_engine(task),
                        task_reason(task),
                        task_phase(task),
                        task_target(task),
                        inflight_total(),
                        len(decode_tasks),
                        len(prefill_tasks),
                        len(head_tasks),
                    )
                except Exception as e:
                    logger.error(
                        "[MoLink][VE%s][DELIVERY] task_failed reason=%s phase=%s target=%s inflight_total=%d decode_inflight=%d prefill_inflight=%d head_inflight=%d error=%s",
                        task_virtual_engine(task),
                        task_reason(task),
                        task_phase(task),
                        task_target(task),
                        inflight_total(),
                        len(decode_tasks),
                        len(prefill_tasks),
                        len(head_tasks),
                        e,
                    )

        def can_launch(item: DeliveryItem) -> bool:
            if item.push_type == "head":
                return len(head_tasks) < self.head_inflight_limit
            if item.phase == "prefill":
                return len(prefill_tasks) < self.prefill_inflight_limit
            return len(decode_tasks) < self.decode_inflight_limit

        async def consumer_loop():
            nonlocal waiting_weight

            def schedule_log(item: DeliveryItem, reason: str, observed_weight: int) -> None:
                now = _now_monotonic()
                item.scheduled_ts = now
                logger.info(
                    "[MoLink][VE%s][DELIVERY] schedule reason=%s phase=%s target=%s W=%d inflight_total=%d decode_inflight=%d prefill_inflight=%d head_inflight=%d decode_q=%d prefill_q=%d head_q=%d enqueue_to_schedule_ms=%.3f",
                    item.virtual_engine,
                    reason,
                    item.phase,
                    item.target_server,
                    observed_weight,
                    inflight_total(),
                    len(decode_tasks),
                    len(prefill_tasks),
                    len(head_tasks),
                    len(decode_queue),
                    len(prefill_queue),
                    len(head_queue),
                    (now - item.enqueue_ts) * 1000.0,
                )

            while not self._shutdown.is_set():
                reap_finished_tasks()

                if not head_queue and not decode_queue and not prefill_queue:
                    item = await fetch_item(0.1)
                    if item is None:
                        if inflight_total():
                            await asyncio.sleep(0.01)
                        continue
                    ingest_item(item)

                drain_nowait()

                while head_queue and len(head_tasks) < self.head_inflight_limit:
                    item, reason, observed_weight = choose_next_item()
                    if item is None:
                        break
                    schedule_log(item, reason, observed_weight)
                    launch_item(item, reason)

                while decode_queue and len(decode_tasks) < self.decode_inflight_limit:
                    item, reason, observed_weight = choose_next_item()
                    if item is None:
                        break
                    if not can_launch(item):
                        if item.push_type == "head":
                            head_queue.appendleft(item)
                        elif item.phase == "prefill":
                            prefill_queue.appendleft(item)
                        else:
                            decode_queue.appendleft(item)
                        break
                    schedule_log(item, reason, observed_weight)
                    launch_item(item, reason)

                while prefill_queue and len(prefill_tasks) < self.prefill_inflight_limit:
                    item, reason, observed_weight = choose_next_item()
                    if item is None:
                        break
                    if not can_launch(item):
                        if item.push_type == "head":
                            head_queue.appendleft(item)
                        elif item.phase == "prefill":
                            prefill_queue.appendleft(item)
                        else:
                            decode_queue.appendleft(item)
                        break
                    schedule_log(item, reason, observed_weight)
                    launch_item(item, reason)

                await asyncio.sleep(0.01)

            remaining = list(head_tasks | decode_tasks | prefill_tasks)
            if remaining:
                await asyncio.gather(*remaining, return_exceptions=True)

        async def main():
            await consumer_loop()
            for channel in channel_cache.values():
                await channel.close()

        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    def stop(self):
        """Stop the delivery process."""
        self._shutdown.set()


class TensorDeliveryManager:
    """Manager for async tensor delivery.

    This class provides a high-level interface for delivering tensors
    and outputs to other nodes in the pipeline.
    """

    def __init__(
        self,
        max_message_size_mb: int = 200,
        max_waiting_weight: int = 30,
        decode_inflight_limit: int = 4,
        prefill_inflight_limit: int = 1,
        head_inflight_limit: int = 1,
        chunk_size_bytes: int = 2 * 1024 * 1024,
        min_chunk_size_bytes: int = 256 * 1024,
        max_chunk_size_bytes: int = 8 * 1024 * 1024,
    ):
        """Initialize the delivery manager.

        Args:
            max_message_size_mb: Maximum gRPC message size in MB.
            max_waiting_weight: Number of decode-priority selections to allow
                before forcing one prefill transmission.
            decode_inflight_limit: Maximum number of concurrent decode send tasks.
            prefill_inflight_limit: Maximum number of concurrent prefill send tasks.
            head_inflight_limit: Maximum number of concurrent head-output send tasks.
            chunk_size_bytes: Fixed fallback chunk size for prefill transmission.
            min_chunk_size_bytes: Minimum adaptive chunk size.
            max_chunk_size_bytes: Maximum adaptive chunk size.
        """
        self.max_message_size_mb = max_message_size_mb
        self.max_waiting_weight = max_waiting_weight
        self.decode_inflight_limit = decode_inflight_limit
        self.prefill_inflight_limit = prefill_inflight_limit
        self.head_inflight_limit = head_inflight_limit
        self.chunk_size_bytes = chunk_size_bytes
        self.min_chunk_size_bytes = min_chunk_size_bytes
        self.max_chunk_size_bytes = max_chunk_size_bytes
        self._process: Optional[TensorDeliveryProcess] = None

    def start(self):
        """Start the delivery process."""
        if self._process is None or not self._process.is_alive():
            self._process = TensorDeliveryProcess(
                self.max_message_size_mb,
                max_waiting_weight=self.max_waiting_weight,
                decode_inflight_limit=self.decode_inflight_limit,
                prefill_inflight_limit=self.prefill_inflight_limit,
                head_inflight_limit=self.head_inflight_limit,
                chunk_size_bytes=self.chunk_size_bytes,
                min_chunk_size_bytes=self.min_chunk_size_bytes,
                max_chunk_size_bytes=self.max_chunk_size_bytes,
            )
            self._process.start()
            logger.info("Tensor delivery process started")

    def stop(self):
        """Stop the delivery process."""
        if self._process is not None:
            self._process.stop()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None
            logger.info("Tensor delivery process stopped")

    def deliver_to_next(
        self,
        intermediate_tensors: Dict[str, torch.Tensor],
        scheduler_output_bytes: bytes,
        grpc_metadata: Dict[str, Any],
        virtual_engine: int,
        next_server: str,
    ):
        """Deliver intermediate tensors to the next pipeline stage.

        This method copies tensors to CPU and queues them for async delivery.

        Args:
            intermediate_tensors: Dict of tensor name to GPU tensor.
            scheduler_output_bytes: Serialized scheduler output.
            grpc_metadata: Pipeline metadata.
            virtual_engine: The virtual engine ID.
            next_server: The address of the next server.
        """
        if self._process is None:
            raise RuntimeError("Delivery process not started")

        tensors_cpu = {k: v.to("cpu") for k, v in intermediate_tensors.items()}
        enqueue_ts = _now_monotonic()
        phase = grpc_metadata.get("transmission_phase", "unknown")
        serialized_payload = None
        transfer_id = None
        total_bytes = 0
        left_bytes = 0
        is_chunked = False
        pre_serialize_ms = 0.0
        if phase == "prefill":
            prepare_start_ts = _now_monotonic()
            serialized_payload = _serialize_prefill_envelope(
                tensors_cpu,
                scheduler_output_bytes,
                grpc_metadata,
                virtual_engine,
            )
            pre_serialize_ms = (_now_monotonic() - prepare_start_ts) * 1000.0
            transfer_id = uuid.uuid4().hex
            total_bytes = len(serialized_payload)
            left_bytes = total_bytes
            is_chunked = True
        logger.info(
            "[MoLink][VE%s][DELIVERY] enqueue phase=%s target=%s tensors=%d pre_serialized=%s pre_serialize_ms=%.3f",
            virtual_engine,
            phase,
            next_server,
            len(tensors_cpu),
            phase == "prefill",
            pre_serialize_ms,
        )

        item = DeliveryItem(
            push_type="next",
            phase=phase,
            virtual_engine=virtual_engine,
            target_server=next_server,
            enqueue_ts=enqueue_ts,
            intermediate_tensors_cpu=tensors_cpu,
            scheduler_output_bytes=scheduler_output_bytes,
            grpc_metadata=grpc_metadata,
            transfer_id=transfer_id,
            serialized_payload=serialized_payload,
            total_bytes=total_bytes,
            left_bytes=left_bytes,
            is_chunked=is_chunked,
        )
        self._process.delivery_queue.put_nowait(item)

    def deliver_to_head(
        self, output_bytes: bytes, virtual_engine: int, head_server: str
    ):
        """Deliver sampler output to the head node.

        Args:
            output_bytes: Serialized ModelRunnerOutput.
            virtual_engine: The virtual engine ID.
            head_server: The address of the head server.
        """
        if self._process is None:
            raise RuntimeError("Delivery process not started")

        item = DeliveryItem(
            push_type="head",
            phase="head",
            virtual_engine=virtual_engine,
            target_server=head_server,
            enqueue_ts=_now_monotonic(),
            output_bytes=output_bytes,
        )
        self._process.delivery_queue.put_nowait(item)
