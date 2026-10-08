#!/usr/bin/env python3
"""Lightweight continuous-session utility contracts."""

from __future__ import annotations

import importlib.util
import builtins
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "tools" / "salt-talk.py"
spec = importlib.util.spec_from_file_location("salt_talk_session_test", CLIENT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def main() -> None:
    original_shell = os.environ.get("SHELL")
    original_timeout = os.environ.get("SALT_TERMINAL_TIMEOUT")
    os.environ["SHELL"] = "/bin/sh"
    try:
        result = module.terminal_result_prompt(
            "printf 'alpha\\n'; printf 'beta\\n' >&2; exit 7",
            ["explain", "the", "failure"],
        )
        assert "explain the failure" in result
        assert "exit status 7" in result
        assert "--- stdout ---\nalpha\n" in result
        assert "--- stderr ---\nbeta\n" in result

        result = module.terminal_result_prompt(
            "yes x | head -c 70000", [],
        )
        assert "exit status 0" in result
        assert "[output truncated to 64 KiB]" in result

        os.environ["SALT_TERMINAL_TIMEOUT"] = "0.05"
        started = time.monotonic()
        result = module.terminal_result_prompt("sleep 5", [])
        assert time.monotonic() - started < 3.0
        assert "timed out after 0.05s" in result

        try:
            module.terminal_result_prompt("", [])
        except ValueError:
            pass
        else:
            raise AssertionError("empty terminal command was accepted")
    finally:
        if original_shell is None:
            os.environ.pop("SHELL", None)
        else:
            os.environ["SHELL"] = original_shell
        if original_timeout is None:
            os.environ.pop("SALT_TERMINAL_TIMEOUT", None)
        else:
            os.environ["SALT_TERMINAL_TIMEOUT"] = original_timeout

    original_state = os.environ.get("SALT_TALK_SESSION_STATE")
    original_id = os.environ.get("SALT_TALK_SESSION_ID")
    original_request_json = module.request_json
    zero = "0" * 64
    state_hash = "1" * 64
    facts_hash = "2" * 64
    build_hash = "3" * 64
    endpoint = "http://127.0.0.1:8090/v1/chat/completions"
    model = "gemma-test"
    system = "test agent"
    with tempfile.TemporaryDirectory(prefix="salt-talk-session-test-") as directory:
        os.environ["SALT_TALK_SESSION_STATE"] = str(
            Path(directory) / "active.json"
        )
        os.environ["SALT_TALK_SESSION_ID"] = "test-default"
        live = {
            "session_mode": "stateful",
            "position": 17,
            "history_turns": 2,
            "mindset_end": 0,
            "state_sha256": state_hash,
            "build_identity_sha256": build_hash,
        }
        calls: list[tuple[str, str, object]] = []

        def fake_request_json(method, url, body=None, *, timeout=60):
            calls.append((method, url, body))
            if method == "GET" and url.endswith("/v1/session"):
                return dict(live)
            if url.endswith("/v1/session/clear"):
                live.update({
                    "position": 0, "history_turns": 0,
                    "state_sha256": zero,
                })
                return {"cleared": True}
            if url.endswith("/v1/session/import"):
                live.update({
                    "position": 17, "history_turns": 2,
                    "state_sha256": state_hash,
                })
                return {
                    "position": 17, "history_turns": 2,
                    "state_sha256": state_hash,
                }
            if url.endswith("/v1/session/export"):
                return {
                    "position": 17,
                    "state_sha256": state_hash,
                    "facts_sha256": facts_hash,
                    "session_manifest": {
                        "history_turns": 2,
                        "state_sha256": state_hash,
                        "facts_sha256": facts_hash,
                    },
                }
            raise AssertionError((method, url, body, timeout))

        module.request_json = fake_request_json
        try:
            label, turns, position = module.save_session_checkpoint(
                endpoint, model, system, "work",
            )
            assert (label, turns, position) == ("work", 2, 17)
            slot_path, slot_id, _ = module.session_checkpoint("work")
            saved = module.load_session_state(slot_path)
            assert saved is not None and saved["id"] == slot_id
            assert saved["position"] == 17 and saved["history_turns"] == 2

            live.update({
                "position": 9, "history_turns": 1,
                "state_sha256": "4" * 64,
            })
            label, turns, position = module.load_session_checkpoint(
                endpoint, model, system, "work",
            )
            assert (label, turns, position) == ("work", 2, 17)
            assert any(url.endswith("/v1/session/clear") for _, url, _ in calls)
            assert any(
                url.endswith("/v1/session/import") and
                isinstance(body, dict) and body.get("id") == slot_id
                for _, url, body in calls
            )
            active = module.load_session_state(Path(directory) / "active.json")
            assert active is not None and active["id"] == "test-default"
            assert active["position"] == 17

            try:
                module.session_checkpoint("../escape")
            except ValueError:
                pass
            else:
                raise AssertionError("unsafe checkpoint name was accepted")
        finally:
            module.request_json = original_request_json
            if original_state is None:
                os.environ.pop("SALT_TALK_SESSION_STATE", None)
            else:
                os.environ["SALT_TALK_SESSION_STATE"] = original_state
            if original_id is None:
                os.environ.pop("SALT_TALK_SESSION_ID", None)
            else:
                os.environ["SALT_TALK_SESSION_ID"] = original_id

    original_input = builtins.input
    original_save_checkpoint = module.save_session_checkpoint
    commands = iter(["/save general", "/exit"])
    captured: list[tuple[str, str, str | None, str | None]] = []

    def fake_save_checkpoint(endpoint, model, system_prompt, name):
        captured.append((endpoint, model, system_prompt, name))
        return "general", 3, 99

    builtins.input = lambda _prompt="": next(commands)
    module.save_session_checkpoint = fake_save_checkpoint
    try:
        args = SimpleNamespace(
            prompt=[], new_session=False, endpoint=endpoint, model=model,
            max_tokens=512, agent="reader", system=None,
        )
        assert module.continuous_session(args, "resolved reader") == 0
        assert captured == [(endpoint, model, "resolved reader", "general")]
    finally:
        builtins.input = original_input
        module.save_session_checkpoint = original_save_checkpoint

    print("salt-talk continuous-session utilities: PASS")


if __name__ == "__main__":
    main()
