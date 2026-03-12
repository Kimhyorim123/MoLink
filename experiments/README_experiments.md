# MoLink / vLLM 비교 실험 가이드 (MVP)

이 폴더는 다음 3개 시스템을 **서로 순차 실행**하면서 동일 workload로 TTFT/TPOT/E2E를 측정하기 위한 스크립트 묶음입니다.

- vLLM (chunked prefill OFF)
- vLLM (chunked prefill ON)
- MoLink (멀티 노드 pipeline)

## 0) 공통 전제

- node4/node5/node6 각각 RTX 4070 1장
- loadgen은 node5에서 실행
- 네트워크 조건: **BW 100Mbps 고정 + RTT 30ms 고정**

## 1) 네트워크 shaping (BW=100Mbps, RTT=30ms)

각 노드(node4/node5/node6)에서 peer 노드 IP로 가는 트래픽에만 shaping을 겁니다.

### 1-1) peer IP 확인

각 노드에서:

- `hostname -I | awk '{print $1}'`

예: node4=10.0.0.4, node5=10.0.0.5, node6=10.0.0.6

### 1-2) shaping 적용 (각 노드에서 실행)

RTT 30ms를 목표로 **one-way delay=15ms**를 적용합니다(양방향 합쳐 RTT≈30ms).

예) node5에서 node4/node6로만 shaping:

- `cd MoLink/experiments/net`
- `sudo ./shape_peers.sh --peers 10.0.0.4,10.0.0.6 --rate-mbps 100 --delay-ms 15`

node4에서도 node5/node6로, node6에서도 node4/node5로 같은 방식으로 실행합니다.

검증:

- `tc -s qdisc show dev $(ip route show default | awk '{print $5; exit}')`
- `ping -c 5 <peer_ip>`

### 1-3) shaping 제거

- `sudo ./clear_shape.sh`

## 2) 서버 실행 커맨드 (예시)

### 2-1) vLLM (일반) — 3 GPU 병렬 (node4/node5/node6)

논문 실험처럼 **vLLM도 총 3개 GPU를 병렬로 사용**하도록, Ray 기반 분산 실행 + vLLM의 pipeline parallel(PP=3)을 사용합니다.

#### (1) Ray 클러스터 기동

사전 점검(권장): vLLM가 사용하는 시스템 Python 환경에서 numpy/pillow/fsspec 버전이 vLLM 요구사항을 만족하도록 맞춥니다.

- node4/node5/node6 각각에서:
  - `cd MoLink`
  - `bash experiments/cluster/pin_vllm_deps.sh`
  - `python3 -m pip check`

node5(HEAD, 192.168.79.4):

- `cd MoLink/experiments/cluster`
- `./ray_head.sh --node-ip 192.168.79.4 --port 6379`

node4(WORKER, 192.168.79.9):

- `cd MoLink/experiments/cluster`
- `./ray_worker.sh --node-ip 192.168.79.9 --head 192.168.79.4:6379`

node6(WORKER, 192.168.79.22):

- `cd MoLink/experiments/cluster`
- `./ray_worker.sh --node-ip 192.168.79.22 --head 192.168.79.4:6379`

주의:

- node4/node6에서 `ray`를 실행할 때, 이전에 `.venv-molink` 같은 가상환경을 켜둔 상태면 Ray 버전이 달라질 수 있습니다. **HEAD/WORKER 모두 동일한 Ray 버전**을 쓰도록 맞춰주세요.
- multi-node PP에서 `NCCL error: internal error`가 나면, 노드별 `libnccl.so`가 달라서 생길 수 있습니다. 가장 단순한 해결은 **세 노드에 동일한 `libnccl.so.2`를 같은 경로로 배치**하는 것입니다.
  - 권장 경로: `/home/sslab/nccl/lib/libnccl.so.2`
  - 예시(node4/node6에서 실행):
    - `mkdir -p /home/sslab/nccl/lib`
    - `scp sslab@192.168.79.4:/home/sslab/anaconda3/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 /home/sslab/nccl/lib/`
  - 그 뒤 **Ray를 stop 후 재기동**하세요.

#### (2) vLLM 서버 기동 (node5에서만)

- `cd MoLink/experiments/run`
- `./vllm_pp3.sh`

서버는 node5의 `http://192.168.79.4:8000`에서 OpenAI API를 제공합니다.

loadgen은 prompt 기반 비교를 위해 기본적으로 OpenAI의 `/v1/completions`를 사용합니다.

### 2-2) vLLM (chunked prefill) — 3 GPU 병렬 (node4/node5/node6)

Ray 클러스터는 위와 동일하게 켜둔 상태에서, node5에서:

- `cd MoLink/experiments/run`
- `./vllm_pp3_chunked.sh`

기본 포트는 `8001`입니다(필요 시 `PORT=... ./vllm_pp3_chunked.sh`).

### 2-3) MoLink 분산 (node4/node5/node6)

현재 레포 기준으로 “분산 파이프라인 + README에 나온 `--molink-start-layer/--molink-end-layer`” 플래그는 `molinkv1` 쪽이 일관됩니다.

MoLink(v1) 서버 엔트리(Streaming `/generate`):

- `python -m molinkv1.entrypoints.api_server ...`

레이어 분할 예시(Qwen2.5-7B가 32 layer라고 가정, end는 **exclusive**):

- stage0(node4): `--molink-start-layer 0  --molink-end-layer 11`
- stage1(node5): `--molink-start-layer 11 --molink-end-layer 22`
- stage2(node6): `--molink-start-layer 22 --molink-end-layer -1`

또한 각 stage는 gRPC 포트를 명시하고(stage0의 gRPC 주소를 stage1/2에 전달) 파이프라인에 조인합니다.

예시(HTTP 포트는 stage0만 외부에서 쓰는 것을 추천):

- stage0(node4):
  - `python -m molinkv1.entrypoints.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000 --molink-enabled --molink-grpc-port 50061 --molink-start-layer 0 --molink-end-layer 11`
- stage1(node5):
  - `python -m molinkv1.entrypoints.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8001 --molink-enabled --molink-grpc-port 50062 --molink-start-layer 11 --molink-end-layer 22 --molink-initial-peer 192.168.79.9:50061`
- stage2(node6):
  - `python -m molinkv1.entrypoints.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8002 --molink-enabled --molink-grpc-port 50063 --molink-start-layer 22 --molink-end-layer -1 --molink-initial-peer 192.168.79.9:50061`

loadgen은 stage0(node4)의 `/generate`로만 요청을 보냅니다.

## 3) loadgen 실행 (node5)

스크립트: `experiments/loadgen/openai_stream_loadgen.py` (OpenAI completions/chat + `/generate` 지원)

### 3-1) 단일 rate 실행 예시

- `python experiments/loadgen/openai_stream_loadgen.py \
    --system vllm \
  --mode openai-completions \
  --base-url http://127.0.0.1:8000 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --rate-rps 0.2 --num-requests 50 --poisson \
    --max-tokens 128 --temperature 0 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --out-csv results_vllm_rate0p2.csv`

MoLink(v1) 분산(stage0)의 `/generate`로 쏘는 예:

- `python experiments/loadgen/openai_stream_loadgen.py \
  --system molink \
  --mode generate \
  --base-url http://192.168.79.9:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --rate-rps 0.2 --num-requests 50 --poisson \
  --max-tokens 128 --temperature 0 \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --out-csv results_molink_rate0p2.csv`

### 3-2) rate sweep(수동)

- 0.1 / 0.2 / 0.3 / 0.7 req/s 각각 CSV를 따로 만들고,
- 시스템(vllm / vllm_chunked / molink)별로 동일하게 반복합니다.

자동 sweep 스크립트:

- `chmod +x experiments/loadgen/run_sweep.sh`
- 예) vLLM sweep:
  - `./experiments/loadgen/run_sweep.sh --system vllm --mode openai-completions --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct --rates 0.1,0.2,0.3,0.7 --num-requests 50 --poisson --max-tokens 128 --temperature 0 --tokenizer Qwen/Qwen2.5-7B-Instruct --out-dir results/vllm`

- 예) MoLink sweep(stage0):
  - `./experiments/loadgen/run_sweep.sh --system molink --mode generate --base-url http://192.168.79.9:8000 --model Qwen/Qwen2.5-7B-Instruct --rates 0.1,0.2,0.3,0.7 --num-requests 50 --poisson --max-tokens 128 --temperature 0 --tokenizer Qwen/Qwen2.5-7B-Instruct --out-dir results/molink`

## 4) CSV 집계/그래프

- `python experiments/analysis/summarize_plot.py --in results_*.csv --out-dir plots/`

출력:

- `plots/summary.csv`
- `plots/ttft_ms_p95.png`, `plots/e2e_ms_p95.png`, `plots/tpot_ms_p95.png`

## 5) 의존성

- loadgen: `httpx`
- 분석: `pandas`, `numpy`, `matplotlib`
- 토큰 카운팅(선택): `transformers`
