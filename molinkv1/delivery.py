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
from collections import deque
from dataclasses import dataclass
from queue import Empty
from typing import Any, Deque, Dict, Optional, Set, Tuple

import grpc.aio as aio
import torch
from vllm.logger import init_logger

from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.utils import get_grpc_options, serialize_metadata

logger = init_logger(__name__)


def _now_monotonic() -> float:
    return time.monotonic()


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
        max_inflight_sends: int = 2,
    ):
        """Initialize the delivery process.

        Args:
            max_message_size_mb: Maximum gRPC message size in MB.
            max_waiting_weight: Number of decode-priority selections to allow
                before forcing one prefill transmission.
            max_inflight_sends: Maximum number of concurrent send tasks.
        """
        super().__init__(daemon=True, name="MolinkTensorDelivery")

        self.max_message_size_mb = max_message_size_mb
        self.max_waiting_weight = max_waiting_weight
        self.max_inflight_sends = max_inflight_sends

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
        inflight_tasks: Set[asyncio.Task] = set()

        def get_stub(address: str) -> molink_pb2_grpc.MolinkServiceStub:
            if address not in stub_cache:
                channel = aio.insecure_channel(
                    address, options=get_grpc_options(self.max_message_size_mb)
                )
                channel_cache[address] = channel
                stub_cache[address] = molink_pb2_grpc.MolinkServiceStub(channel)
            return stub_cache[address]

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
                logger.info(
                    "[MoLink][VE%s][DELIVERY] phase=%s target=%s queue_wait_ms=%.3f "
                    "send_ms=%.3f payload_bytes=%d tensors=%d response=%s",
                    item.virtual_engine,
                    item.phase,
                    item.target_server,
                    (send_start_ts - item.enqueue_ts) * 1000.0,
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
                decode_queue.append(item)
                queue_name = "decode"
                queue_len = len(decode_queue)
            else:
                prefill_queue.append(item)
                queue_name = "prefill"
                queue_len = len(prefill_queue)

            logger.info(
                "[MoLink][VE%s][DELIVERY] classify phase=%s route=%s target=%s decode_q=%d prefill_q=%d selected_q_len=%d",
                item.virtual_engine,
                item.phase,
                queue_name,
                item.target_server,
                len(decode_queue),
                len(prefill_queue),
                queue_len,
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

        def launch_item(item: DeliveryItem, reason: str) -> None:
            if item.push_type == "head":
                coro = deliver_sampler_output(item)
            else:
                coro = deliver_intermediate_tensors(item)

            task = loop.create_task(coro)
            task._molink_phase = item.phase
            task._molink_reason = reason
            task._molink_virtual_engine = item.virtual_engine
            task._molink_target = item.target_server
            inflight_tasks.add(task)
            logger.info(
                "[MoLink][VE%s][DELIVERY] launch reason=%s phase=%s target=%s inflight=%d decode_q=%d prefill_q=%d head_q=%d",
                item.virtual_engine,
                reason,
                item.phase,
                item.target_server,
                len(inflight_tasks),
                len(decode_queue),
                len(prefill_queue),
                len(head_queue),
            )

        def reap_finished_tasks() -> None:
            done = [task for task in inflight_tasks if task.done()]
            for task in done:
                inflight_tasks.remove(task)
                try:
                    task.result()
                    logger.info(
                        "[MoLink][VE%s][DELIVERY] complete reason=%s phase=%s target=%s inflight=%d",
                        task_virtual_engine(task),
                        task_reason(task),
                        task_phase(task),
                        task_target(task),
                        len(inflight_tasks),
                    )
                except Exception as e:
                    logger.error(
                        "[MoLink][VE%s][DELIVERY] task_failed reason=%s phase=%s target=%s inflight=%d error=%s",
                        task_virtual_engine(task),
                        task_reason(task),
                        task_phase(task),
                        task_target(task),
                        len(inflight_tasks),
                        e,
                    )

        async def consumer_loop():
            nonlocal waiting_weight

            while not self._shutdown.is_set():
                reap_finished_tasks()

                if not head_queue and not decode_queue and not prefill_queue:
                    item = await fetch_item(0.1)
                    if item is None:
                        if inflight_tasks:
                            await asyncio.sleep(0.01)
                        continue
                    ingest_item(item)

                drain_nowait()

                while len(inflight_tasks) < self.max_inflight_sends:
                    item, reason, observed_weight = choose_next_item()
                    if item is None:
                        break

                    logger.info(
                        "[MoLink][VE%s][DELIVERY] schedule reason=%s phase=%s target=%s W=%d inflight=%d decode_q=%d prefill_q=%d head_q=%d",
                        item.virtual_engine,
                        reason,
                        item.phase,
                        item.target_server,
                        observed_weight,
                        len(inflight_tasks),
                        len(decode_queue),
                        len(prefill_queue),
                        len(head_queue),
                    )
                    launch_item(item, reason)

                await asyncio.sleep(0.01)

            if inflight_tasks:
                await asyncio.gather(*inflight_tasks, return_exceptions=True)

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
        max_inflight_sends: int = 2,
    ):
        """Initialize the delivery manager.

        Args:
            max_message_size_mb: Maximum gRPC message size in MB.
            max_waiting_weight: Number of decode-priority selections to allow
                before forcing one prefill transmission.
            max_inflight_sends: Maximum number of concurrent send tasks.
        """
        self.max_message_size_mb = max_message_size_mb
        self.max_waiting_weight = max_waiting_weight
        self.max_inflight_sends = max_inflight_sends
        self._process: Optional[TensorDeliveryProcess] = None

    def start(self):
        """Start the delivery process."""
        if self._process is None or not self._process.is_alive():
            self._process = TensorDeliveryProcess(
                self.max_message_size_mb,
                max_waiting_weight=self.max_waiting_weight,
                max_inflight_sends=self.max_inflight_sends,
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
        logger.info(
            "[MoLink][VE%s][DELIVERY] enqueue phase=%s target=%s tensors=%d",
            virtual_engine,
            phase,
            next_server,
            len(tensors_cpu),
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
