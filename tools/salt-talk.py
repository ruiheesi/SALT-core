#!/usr/bin/env python3
"""Small streaming chat client for the local SALT Gemma server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit, urlunsplit


DEFAULT_ENDPOINT = "http://127.0.0.1:8090/v1/chat/completions"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_AGENT_DIR = Path(__file__).resolve().with_name("salt-agents")
DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1] / "build" / "run"
DEFAULT_SESSION_ID = "salt-talk-default"
SESSION_STATE_SCHEMA = "salt.talk.session.v1"
MAX_SESSION_STATE_BYTES = 4 * 1024 * 1024
SESSION_SLOT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}")
MAX_DOCUMENT_BYTES = 64 * 1024
MAX_FIND_SCAN_BYTES = 16 * 1024 * 1024
MAX_TERMINAL_BYTES = 64 * 1024
TERMINAL_TIMEOUT_S = 30.0
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".json", ".jsx", ".md", ".py", ".rs", ".sh",
    ".sql", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}


def dispatch_command() -> None:
    """Handle utility subcommands before parsing a chat prompt."""
    if len(sys.argv) < 2 or sys.argv[1] not in {
            "start", "restart", "down", "logs", "cli-start"}:
        return
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salt-start.sh")
    if sys.argv[1] == "cli-start":
        os.environ["GEMMA4_SESSION_MODE"] = "openai"
        os.environ["GEMMA4_COPILOT_CACHE_ID"] = ""
        os.execv(script, [script, "restart", *sys.argv[2:]])
    os.execv(script, [script, *sys.argv[1:]])


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a response from the local SALT Gemma server.",
        epilog=(
            "commands: st 'question'; st session; st summary FILE...; "
            "st find 'query' [PATH...]; st vision IMAGE [QUESTION]; "
            "st -a reader 'question'; st -mo 1024 'question'; "
            "st --new-session 'question'; st agents; st start; st down; "
            "st cli-start"
        ),
    )
    parser.add_argument("prompt", nargs="*", help="prompt text; reads interactively if omitted")
    parser.add_argument("-a", "--agent", default=os.environ.get("SALT_AGENT"),
                        help="named system prompt from tools/salt-agents")
    parser.add_argument("--system", help="literal system prompt (cannot be combined with -a)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-mo", "--max-tokens", type=int, default=256)
    parser.add_argument(
        "-n", "--new-session", action="store_true",
        help="discard the saved st conversation and start a new one",
    )
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def agent_directory() -> Path:
    return Path(os.environ.get("SALT_AGENT_DIR", DEFAULT_AGENT_DIR)).expanduser()


def available_agents() -> list[str]:
    root = agent_directory()
    if not root.is_dir():
        return []
    return sorted(path.stem for path in root.glob("*.txt") if path.is_file())


def load_system_prompt(agent: str | None, system: str | None) -> str | None:
    if agent and system:
        raise ValueError("use either --agent or --system, not both")
    if system is not None:
        value = system.strip()
        if not value:
            raise ValueError("--system cannot be empty")
        return value
    if agent is None:
        return None
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", agent) is None:
        raise ValueError("agent name must contain only lowercase letters, digits, - or _")
    path = agent_directory() / f"{agent}.txt"
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        known = ", ".join(available_agents()) or "none"
        raise ValueError(f"unknown agent {agent!r}; available: {known}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read agent {agent!r}: {exc}") from exc
    if len(raw) > 8 * 1024:
        raise ValueError(f"agent {agent!r} exceeds 8 KiB")
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"agent {agent!r} is not UTF-8 text") from exc
    if not value:
        raise ValueError(f"agent {agent!r} is empty")
    return value


def safe_text_file(value: str) -> tuple[str, str]:
    if value == "-":
        raw = sys.stdin.buffer.read(MAX_DOCUMENT_BYTES + 1)
        label = "stdin"
    else:
        path = Path(value).expanduser()
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError(f"cannot safely read {value}: {exc}") from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"refusing to read non-regular file: {value}")
            if info.st_size > MAX_DOCUMENT_BYTES:
                raise ValueError(
                    f"text input exceeds {MAX_DOCUMENT_BYTES // 1024} KiB: {value}"
                )
            chunks: list[bytes] = []
            remaining = MAX_DOCUMENT_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            label = str(path)
        finally:
            os.close(descriptor)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError(
            f"text input exceeds {MAX_DOCUMENT_BYTES // 1024} KiB: {label}"
        )
    try:
        return label, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 text") from exc


def file_read_prompt(path: str, question: list[str]) -> str:
    label, text = safe_text_file(path)
    request = " ".join(question).strip()
    if not request:
        request = (
            "Read this file carefully. Explain what it does, its important "
            "relationships and assumptions, and any notable risks."
        )
    return f"{request}\n\n--- {label} ---\n{text}"


def terminal_result_prompt(command: str, question: list[str]) -> str:
    command = command.strip()
    if not command:
        raise ValueError("terminal command cannot be empty")
    shell = os.environ.get("SHELL", "/bin/sh")
    if not os.path.isabs(shell) or not os.access(shell, os.X_OK):
        shell = "/bin/sh"
    timeout_text = os.environ.get(
        "SALT_TERMINAL_TIMEOUT", str(TERMINAL_TIMEOUT_S),
    )
    try:
        timeout = float(timeout_text)
    except ValueError as exc:
        raise ValueError("SALT_TERMINAL_TIMEOUT must be a number") from exc
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 300:
        raise ValueError("SALT_TERMINAL_TIMEOUT must be between 0 and 300 seconds")

    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                [shell, "-c", command],
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except OSError as exc:
            raise ValueError(f"cannot run terminal command: {exc}") from exc
        timed_out = False
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            status = process.returncode

        stdout_size = stdout_file.tell()
        stderr_size = stderr_file.tell()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout_budget = min(stdout_size, MAX_TERMINAL_BYTES)
        stdout = stdout_file.read(stdout_budget)
        stderr_budget = max(0, MAX_TERMINAL_BYTES - len(stdout))
        stderr = stderr_file.read(stderr_budget)
    truncated = stdout_size + stderr_size > MAX_TERMINAL_BYTES
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    request = " ".join(question).strip() or (
        "Analyze this terminal result. Answer concisely and identify any "
        "errors or important details."
    )
    outcome = (
        f"timed out after {timeout:g}s; terminated with status {status}"
        if timed_out else f"exit status {status}"
    )
    suffix = (
        f"\n[output truncated to {MAX_TERMINAL_BYTES // 1024} KiB]"
        if truncated else ""
    )
    return (
        f"{request}\n\n"
        f"Terminal command (executed locally in {Path.cwd()}):\n{command}\n\n"
        f"Result: {outcome}\n"
        f"--- stdout ---\n{stdout_text}"
        f"\n--- stderr ---\n{stderr_text}{suffix}"
    )


def document_summary_prompt(paths: list[str]) -> str:
    if not paths:
        raise ValueError("summary requires a file path, or - for stdin")
    documents: list[str] = []
    total = 0
    for value in paths:
        label, text = safe_text_file(value)
        total += len(text.encode("utf-8"))
        if total > MAX_DOCUMENT_BYTES:
            raise ValueError(
                f"document input exceeds {MAX_DOCUMENT_BYTES // 1024} KiB; "
                "split it into smaller files"
            )
        documents.append(f"--- {label} ---\n{text}")
    return (
        "Summarize the following document text. Give the main points first, "
        "then important details and any decisions or action items. Be concise "
        "and do not invent missing information.\n\n" + "\n\n".join(documents)
    )


def _text_paths(values: list[str]) -> list[Path]:
    roots = [Path(value) for value in values] if values else [Path.cwd()]
    found: list[Path] = []
    ignored = {".git", ".venv", "build", "node_modules", "vendor"}
    for root in roots:
        if root.is_file():
            found.append(root)
            continue
        if not root.is_dir():
            raise ValueError(f"path does not exist: {root}")
        for directory, names, files in os.walk(root):
            names[:] = sorted(name for name in names if name not in ignored)
            for name in sorted(files):
                path = Path(directory) / name
                if path.suffix.lower() in TEXT_SUFFIXES:
                    found.append(path)
    return found


def semantic_find_prompt(values: list[str]) -> str:
    if not values or not values[0].strip():
        raise ValueError("find requires a query")
    query = values[0].strip()
    terms = set(re.findall(r"[a-z0-9_]{3,}", query.lower()))
    chunks: list[tuple[int, str]] = []
    scanned = 0
    for path in _text_paths(values[1:]):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        scanned += len(raw)
        if scanned > MAX_FIND_SCAN_BYTES:
            break
        if len(raw) > 512 * 1024:
            continue
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for start in range(0, len(lines), 60):
            block = "\n".join(lines[start:start + 80])
            lowered = block.lower()
            score = sum(lowered.count(term) for term in terms)
            if query.lower() in lowered:
                score += 100
            label = f"--- {path}:{start + 1} ---\n{block}"
            chunks.append((score, label))
    if not chunks:
        raise ValueError("find found no readable text files")
    chunks.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    size = 0
    for _, chunk in chunks:
        encoded = len(chunk.encode("utf-8"))
        if size + encoded > MAX_DOCUMENT_BYTES:
            continue
        selected.append(chunk)
        size += encoded
        if len(selected) == 16:
            break
    return (
        f"Find the passages most relevant to this query: {query}\n"
        "Answer directly, cite file paths and line numbers from the supplied "
        "chunks, and say when the evidence is uncertain.\n\n" +
        "\n\n".join(selected)
    )


def _bmp_to_ppm(data: bytes) -> tuple[bytes, int, int]:
    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError("image conversion did not produce a BMP")
    offset = int.from_bytes(data[10:14], "little")
    width = int.from_bytes(data[18:22], "little", signed=True)
    signed_height = int.from_bytes(data[22:26], "little", signed=True)
    bits = int.from_bytes(data[28:30], "little")
    compression = int.from_bytes(data[30:34], "little")
    if width < 1 or signed_height == 0 or bits not in (24, 32) or compression != 0:
        raise ValueError("unsupported BMP produced by image conversion")
    height = abs(signed_height)
    pixel_bytes = bits // 8
    stride = ((width * pixel_bytes + 3) // 4) * 4
    if offset + stride * height > len(data):
        raise ValueError("truncated BMP produced by image conversion")
    rows: list[bytes] = []
    order = range(height - 1, -1, -1) if signed_height > 0 else range(height)
    for row_index in order:
        row = data[offset + row_index * stride:offset + row_index * stride +
                   width * pixel_bytes]
        rgb = bytearray(width * 3)
        for column in range(width):
            source = column * pixel_bytes
            target = column * 3
            rgb[target:target + 3] = row[source + 2:source - 1 if source else None:-1]
        rows.append(bytes(rgb))
    return (f"P6\n{width} {height}\n255\n".encode("ascii") + b"".join(rows),
            width, height)


def _sips_bmp(source: Path, target: Path, size: tuple[int, int] | None = None) -> bytes:
    command = ["sips"]
    if size is not None:
        command.extend(("-z", str(size[1]), str(size[0])))
    command.extend(("-s", "format", "bmp", str(source), "--out", str(target)))
    try:
        completed = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"cannot convert image with sips: {exc}") from exc
    if completed.returncode != 0:
        raise ValueError("cannot convert image: " + completed.stderr.strip())
    return target.read_bytes()


def vision_content(values: list[str]) -> tuple[str, str]:
    if not values:
        raise ValueError("vision requires an image path")
    source = Path(values[0]).expanduser()
    if not source.is_file():
        raise ValueError(f"image does not exist: {source}")
    question = " ".join(values[1:]).strip() or "Describe this image concisely."
    raw = source.read_bytes()
    if raw.startswith(b"P6"):
        ppm = raw
    else:
        with tempfile.TemporaryDirectory(prefix="salt-vision-") as directory:
            first = Path(directory) / "source.bmp"
            _, width, height = _bmp_to_ppm(_sips_bmp(source, first))
            factor = math.sqrt((70.0 * 9.0 * 256.0) / (width * height))
            target = (
                math.floor(factor * width / 48.0) * 48,
                math.floor(factor * height / 48.0) * 48,
            )
            if min(target) < 48:
                raise ValueError("image aspect ratio is too extreme")
            resized = Path(directory) / "resized.bmp"
            ppm, _, _ = _bmp_to_ppm(_sips_bmp(source, resized, target))
    if len(ppm) > 2 * 1024 * 1024:
        raise ValueError("converted image exceeds 2 MiB")
    encoded = base64.b64encode(ppm).decode("ascii")
    return question, "data:image/x-portable-pixmap;base64," + encoded


def fail(message: str) -> int:
    print(f"salt-talk: {message}", file=sys.stderr)
    return 2


def session_url(endpoint: str, path: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def session_scope(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def session_state_path() -> Path:
    configured = os.environ.get("SALT_TALK_SESSION_STATE")
    if configured:
        return Path(configured).expanduser()
    run_dir = Path(os.environ.get("SALT_TALK_RUN_DIR", DEFAULT_RUN_DIR)).expanduser()
    return run_dir / "salt-talk-session.json"


def session_cache_id() -> str:
    value = os.environ.get("SALT_TALK_SESSION_ID", DEFAULT_SESSION_ID)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value) is None:
        raise ValueError("invalid SALT_TALK_SESSION_ID")
    return value


def session_checkpoint(name: str | None) -> tuple[Path, str, str]:
    """Resolve a user-visible checkpoint name to private local state."""
    if name is None or name == "default":
        return session_state_path(), session_cache_id(), "default"
    if SESSION_SLOT_PATTERN.fullmatch(name) is None:
        raise ValueError(
            "checkpoint name must be 1-32 letters, digits, '.', '_', or '-'"
        )
    primary = session_state_path()
    digest = hashlib.sha256(
        f"{session_cache_id()}\0{name}".encode("utf-8"),
    ).hexdigest()[:8]
    cache_id = f"salt-talk-{name}-{digest}"
    slot_dir = primary.parent / f"{primary.stem}-sessions"
    return slot_dir / f"{name}.json", cache_id, name


def session_profile_sha256(model: str, system_prompt: str | None) -> str:
    encoded = json.dumps(
        {"model": model, "system": system_prompt},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_json(
        method: str, url: str, body: dict[str, object] | None = None,
        *, timeout: int = 60) -> dict[str, object]:
    command = [
        "curl", "--fail-with-body", "--silent", "--show-error",
        "--connect-timeout", "5", "--max-time", str(timeout),
    ]
    if method == "POST":
        assert body is not None
        command.extend([
            "-H", "Content-Type: application/json",
            "--data-binary", json.dumps(body, separators=(",", ":")),
        ])
    elif method != "GET":
        raise ValueError(f"unsupported HTTP method: {method}")
    command.append(url)
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        response = None
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        if isinstance(response, dict):
            error = response.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                detail = error["message"]
        raise ValueError(detail or f"curl exited {completed.returncode}")
    if not isinstance(response, dict):
        raise ValueError(f"invalid JSON response from {url}")
    return response


def load_session_state(path: Path) -> dict[str, object] | None:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"cannot read session state {path}: {exc}") from exc
    if len(raw) > MAX_SESSION_STATE_BYTES:
        raise ValueError(f"session state is too large: {path}")
    try:
        state = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid saved session {path}; use --new-session to replace it"
        ) from exc
    required = {
        "schema", "id", "endpoint", "model", "profile_sha256", "position",
        "history_turns", "state_sha256", "build_identity_sha256",
        "session_manifest",
    }
    if not isinstance(state, dict) or set(state) != required or \
            state.get("schema") != SESSION_STATE_SCHEMA:
        raise ValueError(
            f"invalid saved session {path}; use --new-session to replace it"
        )
    if (not isinstance(state["id"], str) or
            not isinstance(state["endpoint"], str) or
            not isinstance(state["model"], str) or
            not isinstance(state["position"], int) or state["position"] < 0 or
            not isinstance(state["history_turns"], int) or
            state["history_turns"] < 0 or
            not isinstance(state["session_manifest"], dict) or
            any(re.fullmatch(r"[0-9a-f]{64}", str(state[key])) is None
                for key in ("profile_sha256", "state_sha256",
                            "build_identity_sha256"))):
        raise ValueError(
            f"invalid saved session {path}; use --new-session to replace it"
        )
    return state


def save_session_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ) + "\n").encode("utf-8")
    if len(encoded) > MAX_SESSION_STATE_BYTES:
        raise ValueError("session manifest exceeds the local state limit")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def clear_session(endpoint: str) -> None:
    try:
        request_json(
            "POST", session_url(endpoint, "/v1/session/clear"), {}, timeout=30,
        )
    except ValueError as exc:
        raise ValueError(f"cannot clear plain session: {exc}") from exc


def prepare_session(
        endpoint: str, model: str, system_prompt: str | None,
        *, force_new: bool, has_image: bool, quiet: bool = False,
        ) -> tuple[str, int]:
    path = session_state_path()
    cache_id = session_cache_id()
    if force_new:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError(f"cannot discard saved session {path}: {exc}") from exc
        saved = None
    else:
        saved = load_session_state(path)
    status = request_json("GET", session_url(endpoint, "/v1/session"))
    if status.get("session_mode") == "openai":
        raise ValueError(
            "server is in stateless OpenAI mode; restart it with `st restart`"
        )
    profile = session_profile_sha256(model, system_prompt)
    compatible = (
        saved is not None and not has_image and
        saved["id"] == cache_id and
        saved["endpoint"] == session_scope(endpoint) and
        saved["model"] == model and
        saved["profile_sha256"] == profile and
        saved["build_identity_sha256"] == status.get("build_identity_sha256")
    )
    live = (
        compatible and
        saved["position"] == status.get("position") and
        saved["history_turns"] == status.get("history_turns") and
        saved["state_sha256"] == status.get("state_sha256")
    )
    if live:
        return "live", int(saved["history_turns"])

    if status.get("position") != status.get("mindset_end") or \
            status.get("history_turns") != 0:
        clear_session(endpoint)
    if compatible:
        turns = int(saved["history_turns"])
        if not quiet:
            print(
                f"salt-talk: reloading saved KV ({turns} prior "
                f"{'turn' if turns == 1 else 'turns'})...",
                file=sys.stderr, flush=True,
            )
        imported = request_json(
            "POST", session_url(endpoint, "/v1/session/import"),
            {"id": cache_id, "manifest": saved["session_manifest"]},
            timeout=300,
        )
        if (imported.get("position") != saved["position"] or
                imported.get("history_turns") != saved["history_turns"] or
                imported.get("state_sha256") != saved["state_sha256"]):
            raise ValueError("imported session does not match saved KV identity")
        return "resumed from saved KV", turns
    return "new", 0


def persist_session(
        endpoint: str, model: str, system_prompt: str | None, *,
        cache_id: str | None = None, state_path: Path | None = None,
        ) -> tuple[int, int]:
    cache_id = session_cache_id() if cache_id is None else cache_id
    state_path = session_state_path() if state_path is None else state_path
    exported = request_json(
        "POST", session_url(endpoint, "/v1/session/export"),
        {"id": cache_id, "replace": True}, timeout=300,
    )
    raw_manifest = exported.get("session_manifest")
    if not isinstance(raw_manifest, dict):
        raise ValueError("session export omitted its manifest")
    manifest = dict(raw_manifest)
    position = exported.get("position")
    history_turns = manifest.get("history_turns")
    state_sha256 = exported.get("state_sha256")
    facts_sha256 = exported.get("facts_sha256")
    # Older local servers returned these import-authority fields only at the
    # top level. Keep the client compatible while the corrected server rolls in.
    manifest.setdefault("state_sha256", state_sha256)
    manifest.setdefault("facts_sha256", facts_sha256)
    status = request_json("GET", session_url(endpoint, "/v1/session"))
    build_identity = status.get("build_identity_sha256")
    if (not isinstance(position, int) or position < 0 or
            not isinstance(history_turns, int) or history_turns < 0 or
            not isinstance(state_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", state_sha256) is None or
            not isinstance(facts_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", facts_sha256) is None or
            not isinstance(build_identity, str) or
            re.fullmatch(r"[0-9a-f]{64}", build_identity) is None):
        raise ValueError("session export returned invalid identity metadata")
    state: dict[str, object] = {
        "schema": SESSION_STATE_SCHEMA,
        "id": cache_id,
        "endpoint": session_scope(endpoint),
        "model": model,
        "profile_sha256": session_profile_sha256(model, system_prompt),
        "position": position,
        "history_turns": history_turns,
        "state_sha256": state_sha256,
        "build_identity_sha256": build_identity,
        "session_manifest": manifest,
    }
    save_session_state(state_path, state)
    return history_turns, position


def save_session_checkpoint(
        endpoint: str, model: str, system_prompt: str | None,
        name: str | None) -> tuple[str, int, int]:
    path, cache_id, label = session_checkpoint(name)
    turns, position = persist_session(
        endpoint, model, system_prompt,
        cache_id=cache_id, state_path=path,
    )
    return label, turns, position


def load_session_checkpoint(
        endpoint: str, model: str, system_prompt: str | None,
        name: str | None) -> tuple[str, int, int]:
    path, cache_id, label = session_checkpoint(name)
    saved = load_session_state(path)
    if saved is None:
        raise ValueError(f"saved session {label!r} does not exist")
    status = request_json("GET", session_url(endpoint, "/v1/session"))
    if status.get("session_mode") == "openai":
        raise ValueError(
            "server is in stateless OpenAI mode; restart it with `st restart`"
        )
    profile = session_profile_sha256(model, system_prompt)
    if (saved["id"] != cache_id or
            saved["endpoint"] != session_scope(endpoint) or
            saved["model"] != model or
            saved["profile_sha256"] != profile or
            saved["build_identity_sha256"] !=
                status.get("build_identity_sha256")):
        raise ValueError(
            f"saved session {label!r} is incompatible with this server/model/agent"
        )
    live = (
        saved["position"] == status.get("position") and
        saved["history_turns"] == status.get("history_turns") and
        saved["state_sha256"] == status.get("state_sha256")
    )
    if not live:
        if status.get("position") != status.get("mindset_end") or \
                status.get("history_turns") != 0:
            clear_session(endpoint)
        imported = request_json(
            "POST", session_url(endpoint, "/v1/session/import"),
            {"id": cache_id, "manifest": saved["session_manifest"]},
            timeout=300,
        )
        if (imported.get("position") != saved["position"] or
                imported.get("history_turns") != saved["history_turns"] or
                imported.get("state_sha256") != saved["state_sha256"]):
            raise ValueError("imported session does not match saved KV identity")
    # A named checkpoint is immutable until /save updates it. Clone the loaded
    # live KV into the default slot so subsequent turns auto-resume this branch.
    turns, position = persist_session(endpoint, model, system_prompt)
    return label, turns, position


def cancel_inference(endpoint: str) -> bool:
    parsed = urlsplit(endpoint)
    cancel_url = urlunsplit((
        parsed.scheme, parsed.netloc, "/v1/session/cancel", "", "",
    ))
    completed = subprocess.run(
        [
            "curl", "--fail-with-body", "--silent", "--show-error",
            "--connect-timeout", "2", "--max-time", "10",
            "-H", "Content-Type: application/json", "--data-binary", "{}",
            cancel_url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"curl exited {completed.returncode}"
        raise ValueError(f"cannot stop active inference: {detail}")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid inference-cancel response") from exc
    return response.get("cancelled") is True


def wait_for_server_ready(endpoint: str, timeout: float = 30.0) -> None:
    parsed = urlsplit(endpoint)
    health_url = urlunsplit((
        parsed.scheme, parsed.netloc, "/healthz", "", "",
    ))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        completed = subprocess.run(
            [
                "curl", "--silent", "--show-error",
                "--connect-timeout", "1", "--max-time", "2", health_url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if completed.returncode == 0:
            try:
                status = json.loads(completed.stdout)
            except json.JSONDecodeError:
                status = None
            if (isinstance(status, dict) and status.get("runtime_ready") is True and
                    status.get("phase") == "READY" and
                    status.get("active_request") is None):
                return
        time.sleep(0.25)
    raise ValueError("server did not recover after inference cancellation")


def print_receipt(
        *, mode: str, agent: str | None, system_prompt: str | None,
        prompt: str, has_image: bool, session_kind: str,
        prior_turns: int) -> None:
    context_characters = len(prompt)
    if system_prompt is not None:
        context_characters += len(system_prompt)
    labels = [f"{context_characters} characters"]
    if agent is not None:
        labels.append(f"agent {agent}")
    elif system_prompt is not None:
        labels.append("custom system")
    if mode != "ask":
        labels.append(mode)
    if has_image:
        labels.append("image")
    turn_label = "turn" if prior_turns == 1 else "turns"
    labels.append(f"session {session_kind} ({prior_turns} prior {turn_label})")
    print(
        "i hear you, context " + ", ".join(labels) + ", thinking now",
        file=sys.stderr,
        flush=True,
    )


def start_spinner() -> tuple[threading.Event | None, threading.Thread | None]:
    if not sys.stderr.isatty():
        return None, None
    stopped = threading.Event()

    def animate() -> None:
        frames = ("|", "-", "\\", "/")
        index = 0
        while not stopped.is_set():
            sys.stderr.write(f"\rthinking {frames[index % len(frames)]}")
            sys.stderr.flush()
            index += 1
            stopped.wait(0.12)

    thread = threading.Thread(target=animate, name="salt-spinner", daemon=True)
    thread.start()
    return stopped, thread


def stop_spinner(
        stopped: threading.Event | None,
        thread: threading.Thread | None) -> None:
    if stopped is None or thread is None:
        return
    if stopped.is_set():
        return
    stopped.set()
    thread.join(timeout=1.0)
    sys.stderr.write("\r\033[2K")
    sys.stderr.flush()


def session_child_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--endpoint", args.endpoint,
        "--model", args.model,
        "-mo", str(args.max_tokens),
        "--quiet",
    ]
    if args.agent is not None:
        command.extend(["--agent", args.agent])
    if args.system is not None:
        command.extend(["--system", args.system])
    return command


def run_session_turn(
        args: argparse.Namespace, prompt: list[str], *, fresh: bool) -> int:
    command = session_child_command(args)
    if fresh:
        command.append("--new-session")
    command.extend(["--", *prompt])
    process = subprocess.Popen(command, start_new_session=True)
    previous_handler = signal.getsignal(signal.SIGINT)

    def interrupt_child(_signum, _frame) -> None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, interrupt_child)
    try:
        return process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def reset_session(endpoint: str) -> None:
    clear_session(endpoint)
    path = session_state_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ValueError(f"cannot discard saved session {path}: {exc}") from exc


def print_session_stats(endpoint: str) -> None:
    status = request_json("GET", session_url(endpoint, "/v1/session"))
    turns = status.get("history_turns", status.get("turns", 0))
    position = status.get("position", 0)
    remaining = status.get("remaining_tokens", "unknown")
    phase = status.get("phase", "unknown")
    active = status.get("active_request")
    activity = "idle" if active is None else f"request {active}"
    print(
        f"session: {turns} {'turn' if turns == 1 else 'turns'}, "
        f"{position} KV tokens, {remaining} context tokens remaining, "
        f"{phase}, {activity}",
        file=sys.stderr,
    )


def continuous_session(
        args: argparse.Namespace, system_prompt: str | None) -> int:
    print(
        "salt-talk: continuous conversation "
        "(/file, /terminal, /save, /load, /stats, /new, /exit)",
        file=sys.stderr,
    )
    pending: list[str] | None = list(args.prompt) if args.prompt else None
    fresh = bool(args.new_session)
    while True:
        if pending is None:
            try:
                line = input("You> ").strip()
            except EOFError:
                print(file=sys.stderr)
                return 0
            except KeyboardInterrupt:
                print("\nsalt-talk: leaving continuous session", file=sys.stderr)
                return 130
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                return 0
            if line == "/new":
                try:
                    reset_session(args.endpoint)
                except (OSError, ValueError) as exc:
                    print(f"salt-talk: {exc}", file=sys.stderr)
                else:
                    fresh = False
                    print("salt-talk: session new (0 prior turns)", file=sys.stderr)
                continue
            if line == "/file" or line.startswith("/file "):
                try:
                    file_command = shlex.split(line)
                except ValueError as exc:
                    print(f"salt-talk: invalid /file command: {exc}", file=sys.stderr)
                    continue
                if len(file_command) < 2:
                    print(
                        'usage: /file "path" [question]',
                        file=sys.stderr,
                    )
                    continue
                pending = file_command[1:]
                continue
            if line == "/terminal" or line.startswith("/terminal "):
                try:
                    terminal_command = shlex.split(line)
                except ValueError as exc:
                    print(
                        f"salt-talk: invalid /terminal command: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if len(terminal_command) < 2:
                    print(
                        'usage: /terminal "command" [question]',
                        file=sys.stderr,
                    )
                    continue
                try:
                    terminal_prompt = terminal_result_prompt(
                        terminal_command[1], terminal_command[2:],
                    )
                except ValueError as exc:
                    print(f"salt-talk: {exc}", file=sys.stderr)
                    continue
                pending = [terminal_prompt]
                continue
            if line == "/save" or line.startswith("/save "):
                try:
                    checkpoint_command = shlex.split(line)
                except ValueError as exc:
                    print(f"salt-talk: invalid /save command: {exc}", file=sys.stderr)
                    continue
                if len(checkpoint_command) > 2:
                    print("usage: /save [name]", file=sys.stderr)
                    continue
                try:
                    label, turns, position = save_session_checkpoint(
                        args.endpoint, args.model, system_prompt,
                        checkpoint_command[1] if len(checkpoint_command) == 2 else None,
                    )
                except (OSError, ValueError) as exc:
                    print(f"salt-talk: {exc}", file=sys.stderr)
                else:
                    print(
                        f"salt-talk: saved {label!r} "
                        f"({turns} {'turn' if turns == 1 else 'turns'}, "
                        f"{position} KV tokens)",
                        file=sys.stderr,
                    )
                continue
            if line == "/load" or line.startswith("/load "):
                try:
                    checkpoint_command = shlex.split(line)
                except ValueError as exc:
                    print(f"salt-talk: invalid /load command: {exc}", file=sys.stderr)
                    continue
                if len(checkpoint_command) > 2:
                    print("usage: /load [name]", file=sys.stderr)
                    continue
                try:
                    label, turns, position = load_session_checkpoint(
                        args.endpoint, args.model, system_prompt,
                        checkpoint_command[1] if len(checkpoint_command) == 2 else None,
                    )
                except (OSError, ValueError) as exc:
                    print(f"salt-talk: {exc}", file=sys.stderr)
                else:
                    fresh = False
                    print(
                        f"salt-talk: loaded {label!r} "
                        f"({turns} {'turn' if turns == 1 else 'turns'}, "
                        f"{position} KV tokens)",
                        file=sys.stderr,
                    )
                continue
            if line == "/help":
                print(
                    'commands: /file "path" [question], '
                    '/terminal "command" [question], /save [name], '
                    '/load [name], /new, /stats, '
                    "/exit, /quit, /help",
                    file=sys.stderr,
                )
                continue
            if line == "/stats":
                try:
                    print_session_stats(args.endpoint)
                except (OSError, ValueError) as exc:
                    print(f"salt-talk: {exc}", file=sys.stderr)
                continue
            pending = [line]
        result = run_session_turn(args, pending, fresh=fresh)
        pending = None
        fresh = False
        if result == 130:
            print("salt-talk: session remains open", file=sys.stderr)
        elif result != 0:
            print(
                f"salt-talk: turn failed with status {result}; session remains open",
                file=sys.stderr,
            )


def main() -> int:
    dispatch_command()
    continuous = len(sys.argv) > 1 and sys.argv[1] == "session"
    if continuous:
        del sys.argv[1]
    if len(sys.argv) == 2 and sys.argv[1] == "agents":
        names = available_agents()
        print("\n".join(names) if names else "no agents installed")
        return 0
    mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in {
        "summary", "find", "vision",
    } else "ask"
    if mode != "ask":
        del sys.argv[1]
    args = arguments()
    if args.max_tokens < 1:
        return fail("--max-tokens must be positive")
    try:
        system_prompt = load_system_prompt(args.agent, args.system)
    except ValueError as exc:
        return fail(str(exc))
    if continuous:
        return continuous_session(args, system_prompt)

    image_url: str | None = None
    if mode == "summary":
        try:
            prompt = document_summary_prompt(args.prompt)
        except ValueError as exc:
            return fail(str(exc))
    elif mode == "find":
        try:
            prompt = semantic_find_prompt(args.prompt)
        except ValueError as exc:
            return fail(str(exc))
    elif mode == "vision":
        try:
            prompt, image_url = vision_content(args.prompt)
        except (OSError, ValueError) as exc:
            return fail(str(exc))
    else:
        first = args.prompt[0] if args.prompt else None
        if first == "-" or (first is not None and Path(first).expanduser().exists()):
            try:
                prompt = file_read_prompt(first, args.prompt[1:])
            except ValueError as exc:
                return fail(str(exc))
        else:
            prompt = " ".join(args.prompt).strip()
    if not prompt:
        try:
            prompt = input("You: ").strip()
        except EOFError:
            return fail("no prompt provided")
    if not prompt:
        return fail("prompt is empty")

    try:
        session_kind, prior_turns = prepare_session(
            args.endpoint, args.model, system_prompt,
            force_new=args.new_session, has_image=image_url is not None,
            quiet=args.quiet,
        )
    except (OSError, ValueError) as exc:
        return fail(str(exc))
    if not args.quiet:
        print_receipt(
            mode=mode, agent=args.agent, system_prompt=system_prompt,
            prompt=prompt, has_image=image_url is not None,
            session_kind=session_kind, prior_turns=prior_turns,
        )
    spinner_stop, spinner_thread = start_spinner()

    content: str | list[dict[str, object]] = prompt
    if image_url is not None:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    messages: list[dict[str, object]] = []
    if system_prompt is not None and prior_turns == 0:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    body = json.dumps({
        "model": args.model,
        "messages": messages,
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
                    stop_spinner(spinner_stop, spinner_thread)
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
        stop_spinner(spinner_stop, spinner_thread)
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

    if interrupted:
        print("salt-talk: stopping active inference...", file=sys.stderr)
        try:
            cancelled = cancel_inference(args.endpoint)
            if cancelled:
                wait_for_server_ready(args.endpoint)
        except (OSError, ValueError) as exc:
            print(f"salt-talk: {exc}", file=sys.stderr)

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
    if not args.quiet:
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
    try:
        turns, position = persist_session(
            args.endpoint, args.model, system_prompt,
        )
    except (OSError, ValueError) as exc:
        print(
            f"salt-talk: answer completed, but session save failed: {exc}",
            file=sys.stderr,
        )
        return 2
    if not args.quiet:
        print(
            f"[salt session] saved {turns} "
            f"{'turn' if turns == 1 else 'turns'}, {position} KV tokens",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
