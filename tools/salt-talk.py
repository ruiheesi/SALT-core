#!/usr/bin/env python3
"""Small streaming chat client for the local SALT Gemma server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time


DEFAULT_ENDPOINT = "http://127.0.0.1:8090/v1/chat/completions"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a response from the local SALT Gemma server.",
    )
    parser.add_argument("prompt", nargs="*", help="prompt text; reads interactively if omitted")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=2048)
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"salt-talk: {message}", file=sys.stderr)
    return 2


def main() -> int:
    args = arguments()
    if args.max_tokens < 1:
        return fail("--max-tokens must be positive")

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        try:
            prompt = input("You: ").strip()
        except EOFError:
            return fail("no prompt provided")
    if not prompt:
        return fail("prompt is empty")

    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    })

    started = time.monotonic()
    first_delta: float | None = None
    finished = started
    usage: dict[str, int] = {}
    server_elapsed: float | None = None
    printed = False
    interrupted = False
    done = False
    response_error: str | None = None
    non_sse: list[str] = []

    command = [
        "curl", "--fail-with-body", "--silent", "--show-error",
        "--no-buffer", "--connect-timeout", "5", "--max-time", "1800",
        "-H", "Content-Type: application/json",
        "--data-binary", body,
        args.endpoint,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )

    def stop_process(sig: int) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line.startswith("data:"):
                if line:
                    non_sse.append(line)
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                finished = time.monotonic()
                done = True
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "error" in event:
                error = event["error"]
                response_error = str(
                    error.get("message", error) if isinstance(error, dict) else error
                )
                break

            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            text = delta.get("content") or ""
            if text:
                if first_delta is None:
                    first_delta = time.monotonic()
                print(text, end="", flush=True)
                printed = True

            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            x_salt = event.get("x_salt")
            if isinstance(x_salt, dict) and isinstance(
                    x_salt.get("elapsed_s"), (int, float)):
                server_elapsed = float(x_salt["elapsed_s"])
        else:
            finished = time.monotonic()
    except KeyboardInterrupt:
        interrupted = True
        finished = time.monotonic()
    finally:
        if process.poll() is None:
            if done:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    stop_process(signal.SIGTERM)
            else:
                stop_process(signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            stop_process(signal.SIGKILL)
            process.wait()

    assert process.stderr is not None
    curl_error = process.stderr.read().strip()

    if printed:
        print(flush=True)
    finished = max(finished, time.monotonic())
    first = first_delta if first_delta is not None else finished
    prefill_s = max(0.0, first - started)
    decode_s = max(0.0, finished - first)
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    prefill_rate = prompt_tokens / prefill_s if prefill_s > 0.0 else 0.0
    prefill_ms_per_token = (
        prefill_s * 1000.0 / prompt_tokens if prompt_tokens > 0 else 0.0
    )
    decode_intervals = max(0, completion_tokens - 1)
    decode_rate = decode_intervals / decode_s if decode_s > 0.0 else 0.0
    server_text = "n/a" if server_elapsed is None else f"{server_elapsed:.3f}s"
    print(
        "[salt timing] "
        f"prefill/TTFT={prefill_s:.3f}s "
        f"prefill_rate={prefill_rate:.2f} tok/s "
        f"prefill_ms/token={prefill_ms_per_token:.2f} "
        f"decode={decode_s:.3f}s "
        f"decode_rate={decode_rate:.2f} tok/s "
        f"server_total={server_text} "
        f"tokens={prompt_tokens}->{completion_tokens}",
        file=sys.stderr,
    )
    if interrupted:
        print("salt-talk: interrupted", file=sys.stderr)
        return 130
    if response_error is not None:
        return fail(response_error)
    if process.returncode != 0 and not done:
        detail = "\n".join(non_sse) or curl_error or f"curl exited {process.returncode}"
        return fail(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
