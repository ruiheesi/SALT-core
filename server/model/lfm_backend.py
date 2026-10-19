"""Loopback OpenAI transport for the authenticated LFM2.5 native entry.

The exported LFM executable is intentionally one-shot.  This adapter therefore
replays the bounded text conversation for each request; it does not claim live
KV reuse.  Authentication, tokenization, inference, and sampling remain owned
by the native model package.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine_config import load_engine_config


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "lfm2.5-24b-a2b"
PACKAGE_SCHEMA = "salt.lfm25.package.v1"
PACKAGE_ABI = "salt-lfm25-affine4-6-f32-conv-v1"
BUILD_SCHEMA = "salt.lfm25.build.v1"
ZERO_SHA256 = "0" * 64
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_PROMPT_BYTES = 1024 * 1024
STATE_RE = re.compile(
    rb"(?:^|\n)LFM_STATE .*?state_sha256=([0-9a-f]{64}) ",
)
PERFORMANCE_RE = re.compile(
    rb"(?:^|\n)LFM_PERFORMANCE prompt_tokens=(\d+) completion_tokens=(\d+) "
    rb"request_ns=(\d+) prefill_ns=(\d+) decode_ns=(\d+) .*?"
    rb"peak_rss_bytes=(\d+)(?:\n|$)",
)


def _identity(value: os.stat_result) -> dict[str, int]:
    return {
        "dev": value.st_dev,
        "ino": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


class Engine:
    def __init__(self, args) -> None:
        if not args.lfm_package or not args.lfm_runner:
            raise ValueError("--lfm-package and --lfm-runner are required")
        self.package = Path(args.lfm_package).resolve(strict=True)
        self.runner = Path(args.lfm_runner).resolve(strict=True)
        self.lock = threading.Lock()
        self.active_lock = threading.Lock()
        self.active: subprocess.Popen[bytes] | None = None
        self.closed = False
        self.messages: list[dict[str, str]] = []
        self.history_turns = 0
        self.position = 0
        self.last_native_state_sha256 = ZERO_SHA256
        cfg = load_engine_config(model_dir=str(ROOT / "models/lfm2-24b-a2b"))
        self.context = _integer(
            int(cfg["LFM_CONTEXT"])
            if args.lfm_context is None else args.lfm_context,
            "context", 32, 131072,
        )
        self.batch = _integer(
            int(cfg["LFM_PREFILL_B"])
            if args.lfm_batch is None else args.lfm_batch,
            "batch", 1, 4096,
        )
        self.workers = _integer(
            int(cfg["LFM_WORKERS"])
            if args.lfm_workers is None else args.lfm_workers,
            "workers", 1, 64,
        )
        self.expert_slots = _integer(
            int(cfg["LFM_EXPERT_SLOTS"])
            if args.lfm_expert_slots is None else args.lfm_expert_slots,
            "expert slots", 4, 2432,
        )
        self.output_cap = min(512, int(cfg["LFM_OUTPUT_CAP"]))
        self.timeout = float(
            3600.0 if args.lfm_timeout_s is None else args.lfm_timeout_s,
        )
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("LFM timeout must be positive and finite")
        memory_gb = args.lfm_memory_limit_gb
        self.rss_limit = (
            int(memory_gb * 1_000_000_000) if memory_gb is not None
            else int(cfg["LFM_RSS_LIMIT_BYTES"])
        )
        if self.rss_limit < 1:
            raise ValueError("LFM memory limit must be positive")

        receipt = json.loads((self.package / "auth.json").read_text())
        if receipt.get("schema") != PACKAGE_SCHEMA or \
                receipt.get("abi") != PACKAGE_ABI or \
                receipt.get("model") != "lfm2-24b-a2b":
            raise ValueError("unsupported LFM package")
        expected = {
            "trunk.bin", "pool.bin", "index.txt", "tokenizer.json",
            "tokenizer_config.json", "config.json", "chat_template.jinja",
            "generation_config.json",
        }
        files = receipt.get("files")
        if not isinstance(files, dict) or set(files) != expected:
            raise ValueError("LFM package receipt file set mismatch")
        # setup.py owns this identity and intentionally uses json.dumps'
        # default separators.  Match that byte contract exactly.
        package_key = hashlib.sha256(json.dumps({
            "abi": receipt["abi"],
            "revision": receipt["source_revision"],
            "sha256": {name: row["sha256"] for name, row in files.items()},
        }, sort_keys=True).encode()).hexdigest()
        if package_key != receipt.get("package_id"):
            raise ValueError("LFM package identity mismatch")
        self.package_id = package_key

        build_path = self.runner.with_name(self.runner.name + ".build.json")
        build = json.loads(build_path.read_text())
        if build.get("schema") != BUILD_SCHEMA or \
                build.get("abi") != PACKAGE_ABI or \
                build.get("model") != "lfm2-24b-a2b" or \
                _sha256(self.runner) != build.get("binary_sha256"):
            raise ValueError("LFM binary receipt mismatch")
        for name, digest in build.get("source_sha256", {}).items():
            source = ROOT / name
            if not source.is_file() or _sha256(source) != digest:
                raise ValueError(f"source differs from built LFM runner: {name}")
        self.build_identity_sha256 = str(build["binary_sha256"])
        self.backend = str(build.get("backend", "unknown"))

        self.fds: dict[str, int] = {}
        try:
            for name, record in files.items():
                fd = os.open(
                    self.package / name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                self.fds[name] = fd
                if _identity(os.fstat(fd)) != record.get("identity"):
                    raise ValueError(
                        f"LFM package identity drift; reimport required: {name}"
                    )
            tokenizer_name = "tokenizer.json"
            if cfg.get("LFM_TOKENIZER_LAYOUT") == "compact-merges-v1":
                layout = json.loads(
                    (self.package / "tokenizer.runtime.auth.json").read_text()
                )
                if layout.get("schema") != "salt.lfm25.tokenizer-layout.v1" or \
                        layout.get("layout") != "compact-merges-v1" or \
                        layout.get("package_id") != self.package_id or \
                        layout.get("source_sha256") != files["tokenizer.json"]["sha256"] or \
                        layout.get("complete_document_round_trip") is not True:
                    raise ValueError("LFM runtime tokenizer provenance mismatch")
                tokenizer_name = "tokenizer.runtime.json"
                tokenizer_fd = os.open(
                    self.package / tokenizer_name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                self.fds[tokenizer_name] = tokenizer_fd
                if _identity(os.fstat(tokenizer_fd)) != layout.get("identity"):
                    raise ValueError("LFM runtime tokenizer identity drift")
            self.tokenizer_name = tokenizer_name
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.cancel()
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds.clear()

    def cancel(self) -> bool:
        with self.active_lock:
            process = self.active
        if process is None or process.poll() is not None:
            return False
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return False
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return True

    def _state_sha256(self) -> str:
        return hashlib.sha256(_canonical({
            "model": MODEL_ID,
            "messages": self.messages,
            "history_turns": self.history_turns,
            "position": self.position,
        })).hexdigest()

    def status(self) -> dict[str, object]:
        state_sha = self._state_sha256()
        return {
            "status": "ok",
            "model": MODEL_ID,
            "runtime_ready": not self.closed,
            "session_mode": "replay",
            "position": self.position,
            "session_position": self.position,
            "history_turns": self.history_turns,
            "turns": self.history_turns,
            "remaining_tokens": max(0, self.context - self.position),
            "mindset_end": 0,
            "state_sha256": state_sha,
            "facts_sha256": ZERO_SHA256,
            "build_identity_sha256": self.build_identity_sha256,
            "phase": "RUNNING" if self.active is not None else "READY",
            "active_request": 1 if self.active is not None else None,
            "backend": self.backend,
            "kv_reuse": False,
        }

    def clear(self) -> dict[str, object]:
        with self.lock:
            self.messages = []
            self.history_turns = 0
            self.position = 0
            self.last_native_state_sha256 = ZERO_SHA256
            return self.status()

    def export(self) -> dict[str, object]:
        with self.lock:
            status = self.status()
            manifest = {
                "schema": "salt.lfm25.replay-session.v1",
                "model": MODEL_ID,
                "messages": self.messages,
                "history_turns": self.history_turns,
                "position": self.position,
                "state_sha256": status["state_sha256"],
                "facts_sha256": ZERO_SHA256,
                "build_identity_sha256": self.build_identity_sha256,
            }
            return {
                "position": self.position,
                "state_sha256": status["state_sha256"],
                "facts_sha256": ZERO_SHA256,
                "session_manifest": manifest,
            }

    def import_session(self, manifest: object) -> dict[str, object]:
        if not isinstance(manifest, dict) or \
                manifest.get("schema") != "salt.lfm25.replay-session.v1" or \
                manifest.get("model") != MODEL_ID or \
                manifest.get("build_identity_sha256") != self.build_identity_sha256:
            raise ValueError("incompatible LFM replay session")
        messages = manifest.get("messages")
        turns = manifest.get("history_turns")
        position = manifest.get("position")
        if not isinstance(messages, list) or \
                type(turns) is not int or turns < 0 or \
                type(position) is not int or position < 0:
            raise ValueError("invalid LFM replay session")
        checked = self._validate_messages(messages, allow_empty=True)
        with self.lock:
            self.messages = checked
            self.history_turns = turns
            self.position = position
            if self._state_sha256() != manifest.get("state_sha256"):
                self.messages = []
                self.history_turns = 0
                self.position = 0
                raise ValueError("LFM replay session identity mismatch")
            return self.status()

    @staticmethod
    def _validate_messages(
            messages: object, *, allow_empty: bool = False,
            ) -> list[dict[str, str]]:
        if not isinstance(messages, list) or (not messages and not allow_empty):
            raise ValueError("messages are required")
        result: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"} or \
                    message.get("role") not in {"system", "user", "assistant"} or \
                    not isinstance(message.get("content"), str):
                raise ValueError("Liquid supports text system/user/assistant messages")
            content = str(message["content"])
            if not content or "\0" in content or any(
                    token in content for token in (
                        "<|startoftext|>", "<|im_start|>", "<|im_end|>",
                    )):
                raise ValueError("empty, NUL, or reserved control token in message")
            result.append({"role": str(message["role"]), "content": content})
        return result

    @staticmethod
    def _chat_prompt(messages: list[dict[str, str]]) -> str:
        parts = ["<|startoftext|>"]
        for message in messages:
            parts.append(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            )
        parts.append("<|im_start|>assistant\n<think>\n")
        return "".join(parts)

    def _run(self, prompt: str, output_tokens: int) -> tuple[str, dict[str, int]]:
        encoded = prompt.encode("utf-8")
        if not encoded or len(encoded) > MAX_PROMPT_BYTES:
            raise ValueError("prompt exceeds the Liquid native input bound")
        with tempfile.TemporaryFile(prefix="salt-lfm-prompt-") as prompt_file:
            prompt_file.write(encoded)
            prompt_file.flush()
            prompt_file.seek(0)
            command = [
                str(self.runner),
                str(self.fds["trunk.bin"]), str(self.fds["pool.bin"]),
                str(self.fds["index.txt"]), str(self.fds[self.tokenizer_name]),
                self.package_id, str(self.context), str(self.batch),
                str(self.workers), str(self.expert_slots), str(self.rss_limit),
                str(prompt_file.fileno()), str(output_tokens),
            ]
            pass_fds = tuple(self.fds.values()) + (prompt_file.fileno(),)
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, pass_fds=pass_fds,
                start_new_session=True,
            )
            with self.active_lock:
                self.active = process
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                self.cancel()
                stdout, stderr = process.communicate()
                raise RuntimeError("Liquid native request timed out") from exc
            finally:
                with self.active_lock:
                    if self.active is process:
                        self.active = None
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Liquid native request failed ({process.returncode}): {detail[-2000:]}"
            )
        performance_matches = list(PERFORMANCE_RE.finditer(stdout))
        state_matches = list(STATE_RE.finditer(stdout))
        text_start = stdout.find(b"\nTEXT ")
        if text_start >= 0:
            content_start = text_start + 6
        elif stdout.startswith(b"TEXT "):
            content_start = 5
        else:
            content_start = -1
        if not performance_matches or not state_matches or content_start < 0:
            raise RuntimeError("Liquid native response contract was incomplete")
        performance = performance_matches[-1]
        state = state_matches[-1]
        content_end = state.start()
        if content_end < content_start:
            raise RuntimeError("Liquid native response ordering was invalid")
        text = stdout[content_start:content_end].decode(
            "utf-8", errors="replace",
        )
        metrics = {
            "prompt_tokens": int(performance.group(1)),
            "completion_tokens": int(performance.group(2)),
            "request_ns": int(performance.group(3)),
            "prefill_ns": int(performance.group(4)),
            "decode_ns": int(performance.group(5)),
            "peak_rss_bytes": int(performance.group(6)),
        }
        self.last_native_state_sha256 = state.group(1).decode("ascii")
        return text, metrics

    def complete(self, body: object, *, chat: bool) -> dict[str, object]:
        if not isinstance(body, dict):
            raise ValueError("request must be an object")
        supported = {
            "model", "prompt", "messages", "max_tokens", "stream",
            "stream_options", "temperature", "reasoning_effort",
        }
        if set(body) - supported:
            raise ValueError(
                "unsupported Liquid request fields: " +
                ",".join(sorted(set(body) - supported))
            )
        if body.get("model", MODEL_ID) not in (MODEL_ID, "liquid"):
            raise ValueError("unknown Liquid model")
        if body.get("temperature", 0) not in (0, 0.0, None):
            raise ValueError("Liquid native entry currently supports greedy decode only")
        if type(body.get("stream", False)) is not bool:
            raise ValueError("stream must be boolean")
        requested_output = body.get("max_tokens", 256)
        if requested_output is None:
            requested_output = 256
        output = _integer(
            requested_output, "max_tokens", 1, self.output_cap,
        )
        incoming: list[dict[str, str]]
        if chat:
            incoming = self._validate_messages(body.get("messages"))
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("prompt must be text")
            incoming = [{"role": "user", "content": prompt}]
        started = time.monotonic()
        with self.lock:
            combined = self.messages + incoming
            prompt_text = self._chat_prompt(combined) if chat else incoming[0]["content"]
            text, metrics = self._run(prompt_text, output)
            self.messages = combined + [{"role": "assistant", "content": text}]
            self.history_turns += 1
            self.position = metrics["prompt_tokens"] + metrics["completion_tokens"]
            finish = "length" if metrics["completion_tokens"] >= output else "stop"
            choice: dict[str, object] = {"index": 0, "finish_reason": finish}
            if chat:
                choice["message"] = {"role": "assistant", "content": text}
            else:
                choice["text"] = text
            return {
                "id": f"lfm-{time.time_ns()}",
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": metrics["prompt_tokens"],
                    "completion_tokens": metrics["completion_tokens"],
                    "total_tokens": self.position,
                },
                "x_salt": {
                    "backend": self.backend,
                    "session_mode": "replay",
                    "kv_reuse": False,
                    "native_state_sha256": self.last_native_state_sha256,
                    "peak_rss_bytes": metrics["peak_rss_bytes"],
                    "prefill_ms_per_token": (
                        metrics["prefill_ns"] / 1e6 / metrics["prompt_tokens"]
                        if metrics["prompt_tokens"] else 0.0
                    ),
                    "decode_ms_per_token": (
                        metrics["decode_ns"] / 1e6 / metrics["completion_tokens"]
                        if metrics["completion_tokens"] else 0.0
                    ),
                    "elapsed_s": time.monotonic() - started,
                },
            }


def serve_from_args(args) -> int:
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Liquid serving is loopback-only")
    engine = Engine(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *values: object) -> None:
            print(f"[liquid-serve] {fmt % values}", flush=True)

        def reply(self, status: int, value: object) -> None:
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_body(self) -> dict[str, object]:
            if self.headers.get("Transfer-Encoding") or \
                    len(self.headers.get_all("Content-Length", [])) != 1:
                raise ValueError("one Content-Length is required")
            size = int(self.headers["Content-Length"])
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("request size is out of bounds")
            self.connection.settimeout(30)
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("short request")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("request must be an object")
            return value

        def do_GET(self) -> None:
            if self.path in ("/health", "/healthz"):
                self.reply(200, engine.status())
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{
                    "id": MODEL_ID, "object": "model", "owned_by": "local",
                }]})
            elif self.path == "/v1/session":
                self.reply(200, engine.status())
            else:
                self.reply(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            try:
                body = self.read_body()
                if self.path in ("/v1/chat/completions", "/v1/completions"):
                    result = engine.complete(
                        body, chat=self.path.endswith("/chat/completions"),
                    )
                    if body.get("stream"):
                        choice = result["choices"][0]
                        text = (
                            choice.get("message", {}).get("content", "")
                            if "message" in choice else choice.get("text", "")
                        )
                        first = {
                            "id": result["id"], "object": "chat.completion.chunk",
                            "created": result["created"], "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {"content": text},
                                         "finish_reason": None}],
                            "x_salt": result["x_salt"],
                        }
                        final = {
                            "id": result["id"], "object": "chat.completion.chunk",
                            "created": result["created"], "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {},
                                         "finish_reason": choice["finish_reason"]}],
                            "usage": result["usage"],
                        }
                        data = (
                            "data: " + json.dumps(first, ensure_ascii=False) + "\n\n" +
                            "data: " + json.dumps(final, ensure_ascii=False) + "\n\n" +
                            "data: [DONE]\n\n"
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        self.reply(200, result)
                elif self.path == "/v1/session/clear":
                    self.reply(200, engine.clear())
                elif self.path == "/v1/session/export":
                    self.reply(200, engine.export())
                elif self.path == "/v1/session/import":
                    self.reply(200, engine.import_session(body.get("manifest")))
                elif self.path == "/v1/session/cancel":
                    self.reply(200, {"cancelled": engine.cancel()})
                else:
                    self.reply(404, {"error": {"message": "not found"}})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.reply(400, {"error": {"message": str(exc)}})
            except (RuntimeError, OSError) as exc:
                self.reply(500, {"error": {"message": str(exc)}})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        print(
            f"[liquid-serve] API http://{args.host}:{args.port} model={MODEL_ID} "
            f"backend={engine.backend} CTX{engine.context}/O{engine.output_cap} "
            "session=replay kv-reuse=false runtime_ready=true",
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
    return 0
