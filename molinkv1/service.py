"""
MoLink gRPC service implementation for cross-node pipeline parallelism.
"""

import asyncio
import io
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import cloudpickle
import torch
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.utils import PipelineTopology, deserialize_metadata

if TYPE_CHECKING:
    from .executor import MolinkExecutor

logger = init_logger(__name__)


# Chunked prefill can keep the next stage waiting for a full activation payload
# noticeably longer than the original full-send path, especially across multiple hops.
WORKER_INPUT_TIMEOUT_S = 60.0


@dataclass
class PartialTransferState:
    virtual_engine: int
    total_bytes: int
    received_bytes: int
    grpc_metadata: Dict[str, Any]
    buffer: bytearray


class MolinkService(molink_pb2_grpc.MolinkServiceServicer):
    """gRPC service implementation for MoLink cross-node pipeline parallelism.

    This service handles:
    - Pipeline topology management (joining nodes)
    - Intermediate tensor transfer between pipeline stages
    - Sampler output collection at head node
    - Worker step execution triggers
    """

    def __init__(
        self,
        pipeline_size: int,
        executor: "MolinkExecutor",
        head_ip: str,
        start_layer: int,
        end_layer: int,
    ):
        """Initialize the MoLink service.

        Args:
            pipeline_size: Maximum number of concurrent batches/virtual engines.
            executor: The executor that owns this service.
            head_ip: The IP:port of this node.
            start_layer: First layer this node handles.
            end_layer: Last layer this node handles.
        """
        self.executor = executor
        self.pipeline_size = pipeline_size

        # Queues for inter-stage communication
        # input_queue: receives (scheduler_output, intermediate_tensors, grpc_metadata)
        # output_queue: receives final ModelRunnerOutput
        self.input_queue = [asyncio.Queue() for _ in range(pipeline_size)]
        self.output_queue = [asyncio.Queue() for _ in range(pipeline_size)]

        # Lock for pipeline execution
        self.pp_lock = asyncio.Lock()

        # Chunked prefill reassembly state
        self.partial_transfers: Dict[str, PartialTransferState] = {}
        self.partial_transfer_lock = asyncio.Lock()

        # Pipeline topology
        self.topology = PipelineTopology(head_ip, start_layer, end_layer)

        logger.info(f"MoLink service initialized for node {head_ip}")

    async def JoinPipeline(
        self, request: molink_pb2.NodeInfo, context
    ) -> molink_pb2.GrpcResponseData:
        """Handle a new node joining the pipeline."""
        try:
            node_ip = request.ip
            start_layer = request.start_layer
            end_layer = request.end_layer

            self.topology.add_node(node_ip, start_layer, end_layer)

            logger.info(
                f"Node {node_ip} joined pipeline " f"(layers {start_layer}-{end_layer})"
            )

            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"Error in JoinPipeline: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def GetTopology(
        self, request: molink_pb2.HealthCheckRequest, context
    ) -> molink_pb2.PipelineTopology:
        """Get the current pipeline topology."""
        nodes = []
        for node in self.topology.node_pool:
            nodes.append(
                molink_pb2.NodeInfo(
                    ip=node["ip"],
                    start_layer=node["start_layer"],
                    end_layer=node["end_layer"],
                )
            )

        return molink_pb2.PipelineTopology(nodes=nodes)

    def _store_chunk(
        self,
        request: molink_pb2.GrpcRequestData,
        grpc_metadata: Dict[str, Any],
    ) -> PartialTransferState:
        if not request.transfer_id:
            raise ValueError("Missing transfer_id for chunked transfer")
        if request.total_bytes <= 0:
            raise ValueError("total_bytes must be > 0 for chunked transfer")
        if not request.chunk_data:
            raise ValueError("chunk_data is empty for chunked transfer")

        start = int(request.chunk_offset)
        chunk_len = len(request.chunk_data)
        end = start + chunk_len
        total_bytes = int(request.total_bytes)

        if start < 0 or end > total_bytes:
            raise ValueError(
                f"Chunk offset out of range: offset={start} end={end} total={total_bytes}"
            )

        state = self.partial_transfers.get(request.transfer_id)
        if state is None:
            state = PartialTransferState(
                virtual_engine=request.virtual_engine,
                total_bytes=total_bytes,
                received_bytes=0,
                grpc_metadata=grpc_metadata,
                buffer=bytearray(total_bytes),
            )
            self.partial_transfers[request.transfer_id] = state
        elif state.total_bytes != total_bytes:
            raise ValueError(
                f"Mismatched total_bytes for transfer {request.transfer_id}: "
                f"{state.total_bytes} != {total_bytes}"
            )

        state.buffer[start:end] = request.chunk_data
        state.received_bytes += chunk_len
        return state

    def _is_chunk_complete(self, state: PartialTransferState) -> bool:
        return state.received_bytes == state.total_bytes

    def _decode_chunked_payload(
        self,
        payload: bytes,
    ) -> tuple[bytes, Dict[str, bytes], Dict[str, Any], int]:
        envelope = cloudpickle.loads(payload)
        scheduler_output_bytes = envelope["scheduler_output"]
        intermediate_tensors_bytes = envelope["intermediate_tensors"]
        grpc_metadata = envelope["grpc_metadata"]
        virtual_engine = envelope["virtual_engine"]
        return (
            scheduler_output_bytes,
            intermediate_tensors_bytes,
            grpc_metadata,
            virtual_engine,
        )

    async def _handle_full_intermediate_tensors(
        self,
        request: molink_pb2.GrpcRequestData,
    ) -> molink_pb2.GrpcResponseData:
        virtual_engine = request.virtual_engine
        scheduler_output_bytes = request.scheduler_output

        intermediate_tensors_bytes = {}
        for entry in request.intermediate_tensors.tensors:
            intermediate_tensors_bytes[entry.key] = entry.tensor_data

        grpc_metadata = deserialize_metadata(request.grpc_metadata)
        phase = grpc_metadata.get("transmission_phase", "unknown")

        trace = grpc_metadata.get("jit_runtime_trace", {})
        logger.info(
            "[MoLink][VE%s][SERVICE] received phase=%s scheduler_bytes=%d tensors=%d trace_seq=%s decode_tokens=%s",
            virtual_engine,
            phase,
            len(scheduler_output_bytes),
            len(intermediate_tensors_bytes),
            trace.get("trace_seq"),
            trace.get("decode_token_count"),
        )

        await self.input_queue[virtual_engine].put(
            (
                scheduler_output_bytes,
                intermediate_tensors_bytes,
                grpc_metadata,
            )
        )
        return molink_pb2.GrpcResponseData(res=1)

    async def _handle_chunked_intermediate_tensors(
        self,
        request: molink_pb2.GrpcRequestData,
    ) -> molink_pb2.GrpcResponseData:
        grpc_metadata = deserialize_metadata(request.grpc_metadata)
        phase = grpc_metadata.get("transmission_phase", "unknown")

        async with self.partial_transfer_lock:
            state = self._store_chunk(request, grpc_metadata)
            trace = grpc_metadata.get("jit_runtime_trace", {})
            logger.info(
                "[MoLink][VE%s][SERVICE] chunk_recv phase=%s transfer_id=%s offset=%d chunk_bytes=%d received=%d total=%d trace_seq=%s",
                request.virtual_engine,
                phase,
                request.transfer_id,
                request.chunk_offset,
                len(request.chunk_data),
                state.received_bytes,
                state.total_bytes,
                trace.get("trace_seq"),
            )

            if not self._is_chunk_complete(state):
                return molink_pb2.GrpcResponseData(res=1)

            full_payload = bytes(state.buffer)
            del self.partial_transfers[request.transfer_id]

        (
            scheduler_output_bytes,
            intermediate_tensors_bytes,
            grpc_metadata,
            virtual_engine,
        ) = self._decode_chunked_payload(full_payload)

        logger.info(
            "[MoLink][VE%s][SERVICE] chunk_complete phase=%s transfer_id=%s total_bytes=%d",
            virtual_engine,
            grpc_metadata.get("transmission_phase", "unknown"),
            request.transfer_id,
            len(full_payload),
        )

        await self.input_queue[virtual_engine].put(
            (
                scheduler_output_bytes,
                intermediate_tensors_bytes,
                grpc_metadata,
            )
        )
        return molink_pb2.GrpcResponseData(res=1)

    async def PushIntermediateTensors(
        self, request: molink_pb2.GrpcRequestData, context
    ) -> molink_pb2.GrpcResponseData:
        """Receive intermediate tensors from the previous pipeline stage."""
        try:
            if request.is_chunked:
                return await self._handle_chunked_intermediate_tensors(request)
            return await self._handle_full_intermediate_tensors(request)

        except Exception as e:
            logger.error(f"[MoLink][SERVICE] Error in PushIntermediateTensors: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def PushSamplerOutput(
        self, request: molink_pb2.SamplerOutput, context
    ) -> molink_pb2.GrpcResponseData:
        """Receive sampler output from the last pipeline stage."""
        try:
            virtual_engine = request.virtual_engine
            output_bytes = request.output_data
            await self.output_queue[virtual_engine].put(output_bytes)
            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"[MoLink][SERVICE] Error in PushSamplerOutput: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def ExecuteWorkerStep(
        self, request: molink_pb2.GrpcTriggerRequest, context
    ) -> molink_pb2.GrpcResponseData:
        """Execute a forward step on this worker node."""
        try:
            virtual_engine = request.virtual_engine
            try:
                scheduler_output_bytes, intermediate_tensors_bytes, grpc_metadata = (
                    await asyncio.wait_for(
                        self.input_queue[virtual_engine].get(),
                        timeout=WORKER_INPUT_TIMEOUT_S
                    )
                )
                phase = grpc_metadata.get("transmission_phase", "unknown")
                logger.info(
                    "[MoLink][VE%s][WORKER] dequeued phase=%s scheduler_bytes=%d tensors=%d",
                    virtual_engine,
                    phase,
                    len(scheduler_output_bytes),
                    len(intermediate_tensors_bytes),
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[MoLink][VE{virtual_engine}][WORKER] TIMEOUT waiting for input queue! Queue size: {self.input_queue[virtual_engine].qsize()}"
                )
                raise

            def deserialize_tensors(
                tensor_bytes: Dict[str, bytes],
            ) -> IntermediateTensors:
                tensors = {}
                for key, byte_data in tensor_bytes.items():
                    tensor = torch.load(io.BytesIO(byte_data), map_location="cuda")
                    tensors[key] = tensor
                return IntermediateTensors(tensors=tensors)

            intermediate_tensors = await asyncio.to_thread(
                deserialize_tensors, intermediate_tensors_bytes
            )

            async with self.pp_lock:
                await self.executor.execute_worker_step(
                    scheduler_output_bytes,
                    intermediate_tensors,
                    grpc_metadata,
                    virtual_engine,
                )

            return molink_pb2.GrpcResponseData(res=1)

        except Exception as e:
            logger.error(f"[MoLink][WORKER] Error in ExecuteWorkerStep: {e}")
            traceback.print_exc()
            return molink_pb2.GrpcResponseData(res=0, error_message=str(e))

    async def HealthCheck(
        self, request: molink_pb2.HealthCheckRequest, context
    ) -> molink_pb2.HealthCheckResponse:
        """Health check endpoint."""
        return molink_pb2.HealthCheckResponse(status="healthy")
