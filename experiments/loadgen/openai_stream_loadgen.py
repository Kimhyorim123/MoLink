#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: Optional[int]) -> Optional[float]:
    if ns is None:
        return None
    return ns / 1e6


@dataclass
class RequestResult:
    system: str
    mode: str
    base_url: str
    model: str
    bandwidth_mbps: int
    rtt_ms: int
    rate_rps: float
    req_id: int
    prompt: str
    prompt_len_chars: int
    prompt_tokens: Optional[int]
    max_tokens: int
    temperature: float
    start_time_ns: Optional[int]
    first_token_time_ns: Optional[int]
    end_time_ns: Optional[int]
    ttft_ms: Optional[float]
    e2e_ms: Optional[float]
    tpot_ms: Optional[float]
    generated_text_len_chars: int
    generated_tokens: Optional[int]
    status: str
    error: str


class OptionalTokenizer:
    def __init__(self, tokenizer_name_or_path: Optional[str]):
        self._tok = None
        self._name_or_path = tokenizer_name_or_path
        if not tokenizer_name_or_path:
            return
        try:
            from transformers import AutoTokenizer  # type: ignore

            self._tok = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, trust_remote_code=True
            )
        except Exception as e:
            print(
                f"[warn] tokenizer disabled (failed to load {tokenizer_name_or_path}): {e}",
                file=sys.stderr,
            )
            self._tok = None

    def enabled(self) -> bool:
        return self._tok is not None

    def encode(self, text: str) -> Optional[List[int]]:
        if self._tok is None:
            return None
        try:
            # Most HF tokenizers support add_special_tokens kwarg.
            return list(self._tok.encode(text, add_special_tokens=False))
        except TypeError:
            try:
                return list(self._tok.encode(text))
            except Exception:
                return None
        except Exception:
            return None

    def decode(self, token_ids: List[int]) -> Optional[str]:
        if self._tok is None:
            return None
        try:
            return str(self._tok.decode(token_ids, skip_special_tokens=True))
        except TypeError:
            try:
                return str(self._tok.decode(token_ids))
            except Exception:
                return None
        except Exception:
            return None

    def count_tokens(self, text: str) -> Optional[int]:
        if self._tok is None:
            return None
        try:
            return len(self._tok.encode(text))
        except Exception:
            return None


def make_prompt_with_target_tokens(
    tokenizer: OptionalTokenizer,
    target_tokens: int,
    seed_text: str,
) -> str:
    if target_tokens <= 0:
        raise ValueError("prompt_tokens must be > 0")
    if not tokenizer.enabled():
        raise ValueError("--prompt-tokens requires --tokenizer")

    base_ids = tokenizer.encode(seed_text) or tokenizer.encode("hello")
    if not base_ids:
        raise ValueError("failed to encode seed_text with tokenizer")

    # Start from an exact-length token-id sequence, then iteratively correct
    # any tokenization drift caused by decode->encode normalization.
    repeated = (base_ids * (target_tokens // len(base_ids) + 1))[:target_tokens]
    prompt = tokenizer.decode(repeated)
    if prompt is None:
        raise ValueError("failed to decode token ids into prompt text")

    for _ in range(20):
        n = tokenizer.count_tokens(prompt)
        if n is None:
            return prompt
        if n == target_tokens:
            return prompt
        if n > target_tokens:
            ids = tokenizer.encode(prompt)
            if not ids:
                return prompt
            prompt2 = tokenizer.decode(ids[:target_tokens])
            if prompt2 is None:
                return prompt
            prompt = prompt2
        else:
            need = target_tokens - n
            extra = (base_ids * (need // len(base_ids) + 1))[:need]
            extra_txt = tokenizer.decode(extra) or ""
            prompt = prompt + extra_txt

    return prompt


def build_chat_payload(model: str, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }


def build_completions_payload(model: str, prompt: str, max_tokens: int,
                             temperature: float) -> Dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }


def build_generate_payload(prompt: str, max_tokens: int,
                           temperature: float) -> Dict[str, Any]:
    # molinkv1.entrypoints.api_server expects SamplingParams fields at top-level.
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }


def _extract_openai_delta_text(evt: Dict[str, Any]) -> str:
    # /v1/completions stream: choices[0].text
    try:
        txt = evt.get("choices", [{}])[0].get("text")
        if isinstance(txt, str) and txt:
            return txt
    except Exception:
        pass

    # /v1/chat/completions stream: choices[0].delta.content
    try:
        delta = evt.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
    except Exception:
        pass

    return ""


async def send_one(
    client: httpx.AsyncClient,
    req_id: int,
    system: str,
    mode: str,
    base_url: str,
    model: str,
    bandwidth_mbps: int,
    rtt_ms: int,
    rate_rps: float,
    prompt: str,
    max_tokens: int,
    temperature: float,
    tokenizer: OptionalTokenizer,
    timeout_s: float,
) -> RequestResult:
    base = base_url.rstrip("/")
    if mode == "openai-chat":
        url = base + "/v1/chat/completions"
    elif mode == "openai-completions":
        url = base + "/v1/completions"
    elif mode == "generate":
        url = base + "/generate"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    start_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    end_ns: Optional[int] = None

    generated_text_parts: List[str] = []
    prompt_tokens = tokenizer.count_tokens(prompt)

    try:
        if mode == "openai-chat":
            payload = build_chat_payload(model, prompt, max_tokens, temperature)
        elif mode == "openai-completions":
            payload = build_completions_payload(model, prompt, max_tokens,
                                                temperature)
        else:
            payload = build_generate_payload(prompt, max_tokens, temperature)

        start_ns = now_ns()
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return RequestResult(
                    system=system,
                    mode=mode,
                    base_url=base_url,
                    model=model,
                    bandwidth_mbps=bandwidth_mbps,
                    rtt_ms=rtt_ms,
                    rate_rps=rate_rps,
                    req_id=req_id,
                    prompt=prompt,
                    prompt_len_chars=len(prompt),
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    start_time_ns=start_ns,
                    first_token_time_ns=None,
                    end_time_ns=now_ns(),
                    ttft_ms=None,
                    e2e_ms=ns_to_ms(now_ns() - start_ns),
                    tpot_ms=None,
                    generated_text_len_chars=0,
                    generated_tokens=None,
                    status=f"http_{resp.status_code}",
                    error=body[:4000].decode("utf-8", errors="replace"),
                )

            if mode in ("openai-chat", "openai-completions"):
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break

                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    chunk_text = _extract_openai_delta_text(evt)
                    if chunk_text:
                        if first_token_ns is None:
                            first_token_ns = now_ns()
                        generated_text_parts.append(chunk_text)
            else:
                # /generate streaming: each line is a JSON object like {"text": ["..."]}
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    texts = evt.get("text")
                    if isinstance(texts, list) and texts and isinstance(texts[0], str):
                        if first_token_ns is None:
                            first_token_ns = now_ns()
                        generated_text_parts.append(texts[0])

        end_ns = now_ns()

        if mode == "generate":
            # molinkv1 streams full text (often prompt+generated). Normalize to generated-only if possible.
            full = generated_text_parts[-1] if generated_text_parts else ""
            generated_text = full[len(prompt):] if full.startswith(prompt) else full
        else:
            generated_text = "".join(generated_text_parts)

        generated_tokens = tokenizer.count_tokens(generated_text)

        ttft_ms = None
        tpot_ms = None
        e2e_ms = None

        if start_ns is not None and end_ns is not None:
            e2e_ms = ns_to_ms(end_ns - start_ns)
        if start_ns is not None and first_token_ns is not None:
            ttft_ms = ns_to_ms(first_token_ns - start_ns)
        if first_token_ns is not None and end_ns is not None:
            if generated_tokens is not None and generated_tokens > 1:
                tpot_ms = ns_to_ms(end_ns - first_token_ns) / (generated_tokens - 1)

        return RequestResult(
            system=system,
            mode=mode,
            base_url=base_url,
            model=model,
            bandwidth_mbps=bandwidth_mbps,
            rtt_ms=rtt_ms,
            rate_rps=rate_rps,
            req_id=req_id,
            prompt=prompt,
            prompt_len_chars=len(prompt),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            start_time_ns=start_ns,
            first_token_time_ns=first_token_ns,
            end_time_ns=end_ns,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            tpot_ms=tpot_ms,
            generated_text_len_chars=len(generated_text),
            generated_tokens=generated_tokens,
            status="ok",
            error="",
        )

    except (httpx.TimeoutException, httpx.HTTPError) as e:
        end_ns = now_ns()
        e2e_ms = ns_to_ms(end_ns - start_ns) if start_ns is not None else None
        return RequestResult(
            system=system,
            mode=mode,
            base_url=base_url,
            model=model,
            bandwidth_mbps=bandwidth_mbps,
            rtt_ms=rtt_ms,
            rate_rps=rate_rps,
            req_id=req_id,
            prompt=prompt,
            prompt_len_chars=len(prompt),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            start_time_ns=start_ns,
            first_token_time_ns=None,
            end_time_ns=end_ns,
            ttft_ms=None,
            e2e_ms=e2e_ms,
            tpot_ms=None,
            generated_text_len_chars=0,
            generated_tokens=None,
            status="error",
            error=str(e),
        )
    except Exception as e:
        end_ns = now_ns()
        e2e_ms = ns_to_ms(end_ns - start_ns) if start_ns is not None else None
        return RequestResult(
            system=system,
            mode=mode,
            base_url=base_url,
            model=model,
            bandwidth_mbps=bandwidth_mbps,
            rtt_ms=rtt_ms,
            rate_rps=rate_rps,
            req_id=req_id,
            prompt=prompt,
            prompt_len_chars=len(prompt),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            start_time_ns=start_ns,
            first_token_time_ns=None,
            end_time_ns=end_ns,
            ttft_ms=None,
            e2e_ms=e2e_ms,
            tpot_ms=None,
            generated_text_len_chars=0,
            generated_tokens=None,
            status="error",
            error=repr(e),
        )


def make_arrival_times(rate_rps: float, n: int, poisson: bool, seed: int) -> List[float]:
    rng = random.Random(seed)
    t = 0.0
    times: List[float] = []

    if rate_rps <= 0:
        raise ValueError("rate_rps must be > 0")

    for _ in range(n):
        if poisson:
            dt = rng.expovariate(rate_rps)
        else:
            dt = 1.0 / rate_rps
        t += dt
        times.append(t)

    return times


async def run_once(args: argparse.Namespace) -> None:
    tokenizer = OptionalTokenizer(args.tokenizer)

    prompts: List[str]
    if args.prompt_tokens is not None:
        if args.prompt_file:
            raise ValueError("--prompt-tokens cannot be used with --prompt-file")
        prompt = make_prompt_with_target_tokens(
            tokenizer=tokenizer,
            target_tokens=args.prompt_tokens,
            seed_text=args.prompt_seed_text,
        )
        prompts = [prompt]
    elif args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        if not prompts:
            raise ValueError("prompt_file is empty")
    else:
        prompts = [args.prompt]

    arrival_times = make_arrival_times(
        rate_rps=args.rate_rps,
        n=args.num_requests,
        poisson=args.poisson,
        seed=args.seed,
    )

    sem = asyncio.Semaphore(args.max_in_flight)
    results: List[RequestResult] = []

    async def one_task(req_id: int, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        prompt = prompts[req_id % len(prompts)]

        async with sem:
            res = await send_one(
                client,
                req_id=req_id,
                system=args.system,
                mode=args.mode,
                base_url=args.base_url,
                model=args.model,
                bandwidth_mbps=args.bandwidth_mbps,
                rtt_ms=args.rtt_ms,
                rate_rps=args.rate_rps,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                tokenizer=tokenizer,
                timeout_s=args.timeout_s,
            )
            results.append(res)

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    limits = httpx.Limits(
        max_connections=args.max_in_flight * 2,
        max_keepalive_connections=args.max_in_flight * 2,
    )

    async with httpx.AsyncClient(headers=headers, limits=limits) as client:
        t0 = time.perf_counter()
        tasks = []
        for i, at in enumerate(arrival_times):
            delay_s = max(0.0, at - (time.perf_counter() - t0))
            tasks.append(asyncio.create_task(one_task(i, delay_s)))
        await asyncio.gather(*tasks)

    results.sort(key=lambda r: r.req_id)

    fieldnames = [f.name for f in RequestResult.__dataclass_fields__.values()]  # type: ignore

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)

    ok = sum(1 for r in results if r.status == "ok")
    err = len(results) - ok
    print(f"[done] wrote {args.out_csv} (ok={ok}, error={err})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI streaming load generator for TTFT/E2E/TPOT.")

    p.add_argument("--system", required=True, help="Label: vllm | vllm_chunked | molink")
    p.add_argument(
        "--mode",
        choices=["openai-completions", "openai-chat", "generate"],
        default="openai-completions",
        help="Request protocol: OpenAI completions/chat, or /generate (molinkv1).",
    )
    p.add_argument("--base-url", required=True, help="e.g., http://127.0.0.1:8000")
    p.add_argument("--model", required=True, help="e.g., Qwen/Qwen2.5-7B-Instruct")

    p.add_argument("--bandwidth-mbps", type=int, default=100)
    p.add_argument("--rtt-ms", type=int, default=30)

    p.add_argument("--rate-rps", type=float, required=True)
    p.add_argument("--num-requests", type=int, required=True)
    p.add_argument("--poisson", action="store_true", help="Poisson arrivals (Exp inter-arrival).")

    p.add_argument("--prompt", default="Write a short haiku about GPUs.")
    p.add_argument("--prompt-file", default=None, help="One prompt per line.")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="If set, synthesize a prompt with exactly this many tokens (requires --tokenizer).",
    )
    p.add_argument(
        "--prompt-seed-text",
        default=(
            "Benchmark prompt. Please follow the instructions carefully and answer clearly. "
            "Repeat and expand as needed."
        ),
        help="Seed text used to synthesize a --prompt-tokens prompt.",
    )

    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--timeout-s", type=float, default=600.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-in-flight", type=int, default=32)

    p.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer name/path for token counting (recommended: Qwen/Qwen2.5-7B-Instruct).",
    )

    p.add_argument("--out-csv", required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_once(args))


if __name__ == "__main__":
    main()
