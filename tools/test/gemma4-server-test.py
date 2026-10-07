#!/usr/bin/env python3
"""No-model contract tests for the experimental Gemma 4 server backend."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "server" / "model" / "gemma4_backend.py"
SERVE = ROOT / "server" / "serve.py"

spec = importlib.util.spec_from_file_location("salt_gemma4_server", BACKEND)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def expect_request_error(function, text: str) -> None:
    try:
        function()
    except module.RequestError as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f"expected RequestError containing {text!r}")


def expect_backend_error(function, text: str) -> None:
    try:
        function()
    except module.BackendError as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f"expected BackendError containing {text!r}")


def runner_fixture(modality: str, response: str, *, prompt_tokens: int = 23,
                   output_ids: tuple[int, ...] = (818, 106), cap: int = 50,
                   stop_token: int | None = None):
    marker = "GEMMA4_MM" if modality == "image" else "GEMMA4_QA"
    native_stop = output_ids[-1] if stop_token is None else stop_token
    payload = response.encode("utf-8")
    stdout = (
        f"{marker}_RESPONSE_BEGIN\n".encode("ascii") + payload +
        (b"" if payload.endswith(b"\n") else b"\n") +
        f"{marker}_RESPONSE_END\n".encode("ascii") +
        (b"runtime_ready=true\n" if modality == "image" else b"")
    )
    done = (
        f"{marker}_EXECUTION_DONE prompt_tokens={prompt_tokens} "
        + ("image_tokens=64 " if modality == "image" else "") +
        f"output_steps={len(output_ids)} stop_token={native_stop} "
        f"response_bytes={len(payload)} kv_loaded_tokens=0 "
        f"kv_saved_tokens=0 runtime_ready=true"
    )
    stderr = (
        "runtime_ready=true\n" +
        "output_ids=" + ",".join(str(value) for value in output_ids) + "\n" +
        done + "\n"
    ).encode("utf-8")
    return module.parse_runner_output(stdout, stderr, modality, cap)


def main() -> None:
    # Token aliases are exact and bounded by one startup-owned native capacity.
    controls = module.resolve_controls({}, server_kv_budget_gb=1.0)
    assert (controls.context_tokens, controls.output_tokens) == (512, 128)
    assert controls.reasoning_effort == "none"
    assert controls.enable_thinking is False
    assert controls.kv_bytes == 230_686_720
    assert controls.sampler_abi == "salt-greedy-v1"
    assert controls.sampler_abi_id == 1
    assert controls.temperature == 0.0
    assert controls.temperature_bits == "00000000"
    assert controls.seed == 0
    assert controls.top_k == 1
    assert controls.rng_abi == "none"

    copilot_default = module.resolve_controls(
        {}, server_kv_budget_gb=1.25,
        context_tokens=20_200, max_output_tokens=4_096,
    )
    assert copilot_default.output_tokens == 1_024

    sampled = module.resolve_controls(
        {"temperature": 0.8, "seed": 42, "top_k": 20},
        server_kv_budget_gb=1.0,
    )
    assert sampled.sampler_abi == "salt-temperature-counter-v1"
    assert sampled.sampler_abi_id == 2
    assert sampled.temperature_bits == "3f4ccccd"
    assert sampled.seed == 42
    assert sampled.top_k == 20
    assert sampled.rng_abi == "splitmix64-counter-v1"

    low = module.resolve_controls({"reasoning_effort": "low"},
                                  server_kv_budget_gb=1.0)
    medium = module.resolve_controls({"reasoning": {"effort": "medium"}},
                                     server_kv_budget_gb=1.0)
    high = module.resolve_controls({"reasoning_effort": "high"},
                                   server_kv_budget_gb=1.0)
    assert (low.context_tokens, low.output_tokens) == (512, 128)
    assert (medium.context_tokens, medium.output_tokens) == (512, 128)
    assert (high.context_tokens, high.output_tokens) == (512, 128)
    assert low.enable_thinking and medium.enable_thinking and high.enable_thinking
    assert high.kv_bytes == 230_686_720

    explicit = module.resolve_controls({"max_completion_tokens": 128},
                                       server_kv_budget_gb=1.0)
    assert (explicit.context_tokens, explicit.output_tokens) == (512, 128)
    assert explicit.enable_thinking is False
    expect_request_error(
        lambda: module.resolve_controls(
            {"max_tokens": 50, "max_output_tokens": 128},
            server_kv_budget_gb=1.0),
        "conflicting output-token",
    )
    bounded = module.resolve_controls({"max_tokens": 64},
                                      server_kv_budget_gb=1.0)
    assert (bounded.context_tokens, bounded.output_tokens) == (512, 64)
    thinking_bounded = module.resolve_controls(
        {"reasoning_effort": "high", "max_tokens": 50},
        server_kv_budget_gb=1.0,
    )
    assert thinking_bounded.output_tokens == 50
    assert thinking_bounded.enable_thinking is True
    long_context = module.resolve_controls(
        {"max_tokens": 128}, server_kv_budget_gb=1.0,
        context_tokens=8192, max_output_tokens=256,
    )
    assert long_context.context_tokens == 8192
    assert long_context.kv_bytes == 754_974_720
    old_prefill = module.GEMMA_ENGINE_CONFIG.get("SALT_PREFILL_B")
    old_prefill_env = os.environ.pop("GEMMA4_PREFILL_CHUNK_TOKENS", None)
    try:
        module.GEMMA_ENGINE_CONFIG["SALT_PREFILL_B"] = "256"
        assert module._gemma_prefill_chunk_setting(None) == 256
        expect_request_error(
            lambda: module._gemma_prefill_chunk_setting(512),
            "must match selected model/platform recipe",
        )
    finally:
        if old_prefill is None:
            module.GEMMA_ENGINE_CONFIG.pop("SALT_PREFILL_B", None)
        else:
            module.GEMMA_ENGINE_CONFIG["SALT_PREFILL_B"] = old_prefill
        if old_prefill_env is not None:
            os.environ["GEMMA4_PREFILL_CHUNK_TOKENS"] = old_prefill_env
    old_engine_config = dict(module.GEMMA_ENGINE_CONFIG)
    try:
        module.GEMMA_ENGINE_CONFIG.update({
            "SALT_GEMMA_COMPUTE_NODE": "full",
            "SALT_TEXT_FINE_TOKEN": "1",
            "SALT_GPU_BOUNDED_WEIGHTS": "1",
            "SALT_GPU_WEIGHT_ADDRESSABILITY": "pageable",
        })
        assert module._gemma_kv_layout_setting() == "hybrid"
        module.GEMMA_ENGINE_CONFIG["SALT_TEXT_FINE_TOKEN"] = "0"
        assert module._gemma_kv_layout_setting() == "hybrid"
        # Registered GPU attention must admit the same bounded native seats.
        module.GEMMA_ENGINE_CONFIG.update({
            "SALT_GPU_BOUNDED_WEIGHTS": "0",
            "SALT_GEMMA_GPU_KV_RING": "0",
            "SALT_GPU_WEIGHT_ADDRESSABILITY": "bounded-window",
        })
        assert module._gemma_kv_layout_setting() == "hybrid"
        assert module.gemma4_kv_capacity_bytes(
            12288, module._gemma_kv_layout_setting(),
        ) == 922_746_880
    finally:
        module.GEMMA_ENGINE_CONFIG.clear()
        module.GEMMA_ENGINE_CONFIG.update(old_engine_config)
    full_gpu_context = module.resolve_controls(
        {"max_tokens": 128}, server_kv_budget_gb=4.0,
        context_tokens=8192, max_output_tokens=256,
        kv_layout="absolute",
    )
    assert full_gpu_context.kv_bytes == 3_690_987_520
    expect_request_error(
        lambda: module.resolve_controls(
            {"max_tokens": 128}, server_kv_budget_gb=3.0,
            context_tokens=8192, max_output_tokens=256,
            kv_layout="absolute",
        ),
        "KV allowance",
    )
    expect_request_error(
        lambda: module.resolve_controls(
            {"max_tokens": 257}, server_kv_budget_gb=1.0,
            context_tokens=8192, max_output_tokens=256,
        ),
        "server maximum 256",
    )
    expect_request_error(
        lambda: module.resolve_controls({"reasoning_effort": "high"},
                                        server_kv_budget_gb=0.2),
        "KV allowance",
    )
    expect_request_error(
        lambda: module.resolve_controls({"kv_allowance_gb": 1.1},
                                        server_kv_budget_gb=1.0),
        "server maximum",
    )

    status_backend = module.Gemma4Backend(module.Gemma4Config(
        source_dir=Path("."), pool=Path("."), receipt=Path("."),
        context_tokens=8192, max_output_tokens=256,
        prefill_chunk_tokens=512, kv_budget_gb=1.0,
    ))
    status_backend.session_position = 1500
    status = status_backend.session_status()
    assert status["context_limit"] == 8192
    assert status["output_reserve"] == 256
    assert status["remaining_tokens"] == 6692

    # The authenticated artifact cannot be cosmetically relabeled.
    try:
        module.Gemma4Backend(module.Gemma4Config(
            source_dir=Path("."), pool=Path("."), receipt=None,
            text_wrapper=Path("."), multimodal_wrapper=Path("."),
            text_runner=Path("."), multimodal_runner=Path("."),
            model_id="cosmetic-alias",
        ))
    except module.BackendError as exc:
        assert "model id must be" in str(exc)
    else:
        raise AssertionError("cosmetic Gemma model id was accepted")

    # Gemma-only environment parsing must not break the default Qwen parser.
    gemma_env = {
        "GEMMA4_WORKERS": "not-an-int",
        "GEMMA4_KV_BUDGET_GB": "not-a-float",
        "GEMMA4_TIMEOUT_S": "not-a-float",
    }
    previous_env = {key: os.environ.get(key) for key in gemma_env}
    os.environ.update(gemma_env)
    try:
        serve_spec = importlib.util.spec_from_file_location(
            "salt_gemma4_qwen_parser", SERVE,
        )
        assert serve_spec is not None and serve_spec.loader is not None
        serve_module = importlib.util.module_from_spec(serve_spec)
        serve_spec.loader.exec_module(serve_module)
        parsed = serve_module.build_parser().parse_args(["--backend", "qwen"])
        assert parsed.backend == "qwen"
        assert parsed.gemma_workers is None
        assert parsed.gemma_kv_budget_gb is None
        assert parsed.gemma_timeout_s is None
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # Sampling is either qualified greedy or seeded counter-temperature play.
    expect_request_error(
        lambda: module.Gemma4Backend._validate_sampling({"top_p": 0.9}),
        "greedy decode",
    )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_sampling({"seed": 7}),
        "seed requires positive temperature",
    )
    module.Gemma4Backend._validate_sampling(
        {"temperature": 0.8, "seed": 7, "top_k": 20},
    )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_sampling({"temperature": 2.1}),
        "between 0 and 2",
    )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_sampling(
            {"temperature": 0.8, "seed": -1},
        ),
        "unsigned 64-bit",
    )
    for field, value in (
        ("temperature", False), ("top_p", True), ("n", True),
        ("logprobs", 0),
    ):
        expect_request_error(
            lambda field=field, value=value: module.Gemma4Backend._validate_sampling(
                {field: value},
            ),
            field,
        )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_request_schema(
            {"messages": [], "stream": "false"}, endpoint="chat",
        ),
        "stream must be a boolean",
    )
    module.Gemma4Backend._validate_request_schema(
        {"input": "x", "stream": True}, endpoint="responses",
    )
    module.Gemma4Backend._validate_request_schema(
        {"messages": [], "stream": True,
         "stream_options": {"include_usage": True}}, endpoint="chat",
    )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_request_schema(
            {"messages": [], "stream": False, "stream_options": {}},
            endpoint="chat",
        ),
        "requires Chat stream=true",
    )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_request_schema(
            {"messages": [], "stream": True,
             "stream_options": {"include_usage": 1}}, endpoint="chat",
        ),
        "include_usage must be a boolean",
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    module.Gemma4Backend._validate_request_schema({
        "messages": [], "tools": tools, "tool_choice": "auto",
    }, endpoint="chat")
    module.Gemma4Backend._validate_request_schema({
        "messages": [], "tools": tools,
        "tool_choice": {
            "type": "function", "function": {"name": "get_weather"},
        },
    }, endpoint="chat")
    expect_request_error(
        lambda: module.Gemma4Backend._validate_request_schema({
            "messages": [], "tools": tools,
            "tool_choice": {
                "type": "function", "function": {"name": "missing"},
            },
        }, endpoint="chat"),
        "unavailable function",
    )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_request_schema(
            {"messages": [], "logit_bias": {}}, endpoint="chat",
        ),
        "unsupported Chat fields",
    )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_request_schema(
            {"input": "x", "tools": tools}, endpoint="responses",
        ),
        "Responses API tool calls",
    )
    expect_request_error(
        lambda: module.Gemma4Backend._validate_request_schema(
            {"input": "x", "text": {"format": {"type": "json_schema"}}},
            endpoint="responses",
        ),
        "unsupported Responses fields",
    )
    response_messages = module._responses_messages({
        "instructions": "Be exact.", "input": "Answer once.",
    })
    assert response_messages == [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "Answer once."},
    ]
    expect_request_error(
        lambda: module._responses_messages({"instructions": 7, "input": "x"}),
        "instructions must be a string",
    )
    expect_request_error(
        lambda: module._responses_messages({
            "input": [{"role": "user", "content": "x", "ignored": True}],
        }),
        "unsupported Responses message fields",
    )

    # Provider/API credentials are not inherited by authenticated local runners.
    secret_name = "SALT_GEMMA4_TEST_TOKEN"
    old_secret = os.environ.get(secret_name)
    os.environ[secret_name] = "must-not-be-inherited"
    try:
        clean = module.Gemma4Backend._clean_env()
        assert secret_name not in clean
        assert clean["SALT_PREFILL_CHUNK"] == "1", clean
        assert clean["SALT_PREFILL_B"] == "512", clean
        assert clean["SALT_GPU_B_QKV"] == "1024", clean
        assert clean["SALT_GPU_B_Z"] == "1024", clean
        assert clean["SALT_GPU_B_O"] == "1024", clean
        assert clean["SALT_M2_BATCH"] == "1", clean
        assert clean["SALT_EXPERT_PRELOAD_LAYERS"] == "none", clean
        for inert in (
                "MODEL", "QUANT", "LAYERS", "SALT_GPU",
                "SALT_GPU_RESIDENT", "SALT_GPU_MOE", "SALT_GPU_OVERLAP",
                "SALT_GPU_TRUNK_PREFILL_ONLY", "SALT_M2_BATCH_B",
                "SALT_CACHE_MODE"):
            assert inert not in clean, (inert, clean)
        original_config = getattr(module, "GEMMA_ENGINE_CONFIG")
        try:
            setattr(module, "GEMMA_ENGINE_CONFIG", dict(
                original_config,
                SALT_EXPERT_BUDGET_GB="10",
                SALT_EXPERT_PRELOAD_LAYERS="0",
            ))
            os.environ["SALT_CACHE_LAYER_QUOTAS"] = "conflict"
            selective = module.Gemma4Backend._clean_env()
            quotas = selective["SALT_CACHE_LAYER_QUOTAS"].split(",")
            assert quotas[0] == "128" and quotas[1:] == ["0"] * 29, quotas
        finally:
            setattr(module, "GEMMA_ENGINE_CONFIG", original_config)
            os.environ.pop("SALT_CACHE_LAYER_QUOTAS", None)
    finally:
        if old_secret is None:
            os.environ.pop(secret_name, None)
        else:
            os.environ[secret_name] = old_secret

    # A server timeout must kill and reap both the Python wrapper and its
    # native descendant, not merely the direct child process.
    with tempfile.TemporaryDirectory() as tmp:
        child_pid_file = Path(tmp) / "native.pid"
        wrapper = (
            "from pathlib import Path; import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(60)']); "
            "Path(sys.argv[1]).write_text(str(p.pid)); p.wait()"
        )
        try:
            module._run_process_group(
                [sys.executable, "-c", wrapper, str(child_pid_file)],
                timeout=0.5, env=module.Gemma4Backend._clean_env(),
            )
        except subprocess.TimeoutExpired:
            pass
        else:
            raise AssertionError("process-group timeout was not raised")
        assert child_pid_file.is_file(), "native timeout probe did not start"
        child_pid = int(child_pid_file.read_text())
        deadline = time.monotonic() + 2.0
        child_alive = True
        while child_alive and time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                child_alive = False
            else:
                time.sleep(0.01)
        assert not child_alive, f"orphaned native child pid={child_pid}"

    # Native Gemma turn framing; reasoning trace is opt-in and split from content.
    rendered = module.render_chat(
        [{"role": "user", "content": "Hello"}], enable_thinking=False)
    assert rendered.prompt == (
        "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"
        "<|channel>thought\n<channel|>"
    )
    assert rendered.image_url is None
    expect_request_error(
        lambda: module.render_chat([
            {"role": "user", "content": "Hello", "name": "projected-away"},
        ], enable_thinking=False),
        "unsupported message fields",
    )

    tool_prompt = module.render_chat(
        [{"role": "user", "content": "What is the weather in Boston?"}],
        enable_thinking=False, tools=tools, tool_choice="required",
    )
    assert "<|tool>declaration:get_weather{" in tool_prompt.prompt
    assert "type:<|\"|>OBJECT<|\"|>" in tool_prompt.prompt
    assert "You must call one or more provided tools" in tool_prompt.prompt
    assert tool_prompt.prompt.endswith(
        "<|turn>model\n<|channel>thought\n<channel|>"
    )

    tool_followup = module.render_chat([
        {"role": "user", "content": "What is the weather in Boston?"},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "call_test", "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Boston"}',
                },
            }],
        },
        {
            "role": "tool", "tool_call_id": "call_test",
            "content": "72 F and sunny",
        },
    ], enable_thinking=False, tools=tools)
    assert (
        '<|tool_call>call:get_weather{city:<|"|>Boston<|"|>}<tool_call|>'
        in tool_followup.prompt
    )
    assert (
        '<|tool_response>response:get_weather{value:<|"|>72 F and sunny<|"|>}'
        '<tool_response|>' in tool_followup.prompt
    )
    assert tool_followup.prompt.endswith(
        "<tool_response|><|channel>thought\n<channel|>"
    )

    # Copilot resends the full history after executing a tool.  The server can
    # authenticate that prefix and append only the tool result to the live KV.
    tool_initial = module.render_chat([
        {"role": "user", "content": "What is the weather in Boston?"},
    ], enable_thinking=False, tools=tools)
    tool_call_text = (
        '<|tool_call>call:get_weather{city:<|"|>Boston<|"|>}<tool_call|>'
    )
    tool_continuation = module._render_tool_continuation([
        {
            "role": "tool", "tool_call_id": "call_test",
            "content": "72 F and sunny",
        },
    ], enable_thinking=False, pending_calls={"call_test": "get_weather"})
    cached_tool_exchange = (
        tool_initial.prompt + tool_call_text + "<|tool_response>" +
        tool_continuation.prompt
    )
    assert cached_tool_exchange.endswith(
        '<|tool_call>call:get_weather{city:<|"|>Boston<|"|>}<tool_call|>'
        '<|tool_response>response:get_weather{value:<|"|>72 F and sunny<|"|>}'
        '<tool_response|><|channel>thought\n<channel|>'
    )
    assert "<|channel>thought\n<channel|><|tool_call>" in cached_tool_exchange
    user_continuation = module._render_tool_continuation([
        {"role": "user", "content": "What about tomorrow?"},
    ], enable_thinking=False, pending_calls={})
    assert user_continuation.prompt == (
        "\n<|turn>user\nWhat about tomorrow?<turn|>\n<|turn>model\n"
        "<|channel>thought\n<channel|>"
    )
    literal_tool_value = 'source <bos> <|tool_response> <|"|>'
    literal_tool_signature = module._tool_history_signature([{
        "role": "tool", "tool_call_id": "call_test",
        "content": literal_tool_value,
    }])
    assert json.loads(literal_tool_signature[0])["content"] == \
        literal_tool_value
    escaped_tool_continuation = module._render_tool_continuation([{
        "role": "tool", "tool_call_id": "call_test",
        "content": literal_tool_value,
    }], enable_thinking=False, pending_calls={"call_test": "get_weather"})
    assert "source <bos>" not in escaped_tool_continuation.prompt
    assert "source \\u003cbos> \\u003c|tool_response> \\u003c|\"|>" in \
        escaped_tool_continuation.prompt
    retry_tool_continuation = module._render_tool_continuation([
        {
            "role": "tool", "tool_call_id": "call_test",
            "content": literal_tool_value,
        },
        {"role": "user", "content": "Try Again"},
    ], enable_thinking=False, pending_calls={"call_test": "get_weather"})
    assert retry_tool_continuation == escaped_tool_continuation
    escaped_assistant_replay = module.render_chat([
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Literal <bos> and <channel|>"},
        {"role": "user", "content": "Next"},
    ], enable_thinking=False)
    assert "Literal \\u003cbos> and \\u003cchannel|>" in \
        escaped_assistant_replay.prompt
    canonical_call_a = module._tool_history_signature([{
        "role": "assistant", "content": None,
        "tool_calls": [{
            "id": "call_test", "type": "function",
            "function": {
                "name": "get_weather", "arguments": '{"city":"Boston"}',
            },
        }],
    }])
    canonical_call_b = module._tool_history_signature([{
        "role": "assistant", "content": "", "reasoning_content": "hidden",
        "tool_calls": [{
            "id": "call_test", "type": "function",
            "function": {
                "name": "get_weather", "arguments": {"city": "Boston"},
            },
        }],
    }])
    assert canonical_call_a == canonical_call_b
    expect_request_error(
        lambda: module._render_tool_continuation([
            {
                "role": "tool", "tool_call_id": "call_one",
                "content": "first",
            },
        ], enable_thinking=False, pending_calls={
            "call_one": "first_tool", "call_two": "second_tool",
        }),
        "answer every pending tool call",
    )

    thought = module.render_chat([
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "Four."},
        {"role": "user", "content": "Again?"},
    ], enable_thinking=True)
    assert thought.prompt.startswith(
        "<bos><|turn>system\n<|think|>\nBe exact.<turn|>\n"
    )
    assert "<|turn>model\nFour.<turn|>\n" in thought.prompt
    assert thought.prompt.endswith("<|turn>model\n")

    ppm = b"P6\n384 384\n255\n" + bytes(384 * 384 * 3)
    image_url = (
        "data:image/x-portable-pixmap;base64," +
        base64.b64encode(ppm).decode("ascii")
    )
    rendered_image = module.render_chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "What is shown?"},
    ]}], enable_thinking=False)
    assert rendered_image.image_url == image_url
    assert rendered_image.prompt.count(module.IMAGE_TOKEN) == 64
    assert module.IMAGE_BEGIN + module.IMAGE_TOKEN in rendered_image.prompt
    assert module.IMAGE_TOKEN + module.IMAGE_END in rendered_image.prompt
    expect_request_error(
        lambda: module.render_chat([
            {"role": "user", "content": "inject " + module.IMAGE_BLOCK},
        ], enable_thinking=False),
        "reserved Gemma control token",
    )
    expect_request_error(
        lambda: module.render_chat([{"role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"},
        }]}], enable_thinking=False),
        "unsupported image_url fields",
    )

    with module.prepared_image(image_url, image_root=None) as prepared:
        assert prepared.path.is_file()
        assert (prepared.width, prepared.height) == (384, 384)
        prepared_path = prepared.path
    assert not prepared_path.exists()

    bad_ppm = b"P6\n224 224\n255\n" + bytes(224 * 224 * 3)
    bad_url = (
        "data:image/x-portable-pixmap;base64," +
        base64.b64encode(bad_ppm).decode("ascii")
    )
    expect_request_error(
        lambda: module.validate_canonical_ppm(bad_ppm),
        "requires bicubic resize",
    )
    expect_request_error(
        lambda: module.render_chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": bad_url}},
        ]}], enable_thinking=False),
        "exactly one image",
    )
    expect_request_error(
        lambda: module.prepared_image("https://example.test/image.ppm", None).__enter__(),
        "remote image URLs",
    )

    # Native response framing, actual token accounting, stop/length, and trace split.
    direct = runner_fixture("text", "The capital is Paris.")
    assert direct.content == "The capital is Paris."
    assert direct.reasoning_content is None
    assert direct.prompt_tokens == 23
    assert direct.kv_loaded_tokens == 0
    assert direct.kv_saved_tokens == 0
    assert direct.completion_tokens == 2
    assert direct.finish_reason == "stop"

    # Generated assistant text is recorded as opaque validated output. A
    # control-token spelling must not turn a committed streaming response into
    # a post-decode client-input validation error.
    generated_control = runner_fixture("text", "Literal <bos> spelling.")
    generated_signature = module._tool_result_history_signature(
        generated_control
    )
    assert json.loads(generated_signature)["content"] == \
        "Literal <bos> spelling."
    assert module._tool_history_signature([{
        "role": "assistant", "content": "Literal <bos> spelling.",
    }]) == (generated_signature,)

    tool_result = runner_fixture(
        "text",
        '<|tool_call>call:get_weather{city:<|"|>Boston<|"|>,days:2}'
        '<tool_call|>',
        output_ids=(48, 818, 49, 50), stop_token=50,
    )
    assert tool_result.finish_reason == "tool_calls"
    assert tool_result.content == ""
    assert len(tool_result.tool_calls) == 1
    assert tool_result.tool_calls[0].name == "get_weather"
    assert tool_result.tool_calls[0].arguments == '{"city":"Boston","days":2}'
    assert tool_result.tool_calls[0].id.startswith("call_")
    assert module._tool_result_history_signature(tool_result) == \
        module._tool_history_signature([{
            "role": "assistant", "content": tool_result.content,
            "tool_calls": [
                call.as_dict() for call in tool_result.tool_calls
            ],
        }])[0]
    streamed_deltas: list[tuple[str, str]] = []
    accumulator = module.NativeV2Accumulator(
        lambda kind, value: streamed_deltas.append((kind, value)),
    )
    streamed_call = '<|tool_call>call:get_weather{city:<|"|>Boston<|"|>,days:2}'
    accumulator.feed(1, 48, 0, b"<|tool_call>")
    accumulator.feed(2, 818, 0, streamed_call.encode("utf-8"))
    accumulator.feed(3, 49, 0, (streamed_call + "<tool_call|>").encode("utf-8"))
    accumulator.feed(4, 50, 1, (streamed_call + "<tool_call|>").encode("utf-8"))
    accumulator.finish(tool_result)
    assert streamed_deltas == []

    traced = runner_fixture(
        "text", "<|channel>thought\nAdd two pairs.\n<channel|>Four.",
        output_ids=tuple(range(50)), cap=50, stop_token=-1,
    )
    assert traced.stop_token == -1
    assert traced.reasoning_content == "Add two pairs."
    assert traced.content == "Four."
    assert traced.finish_reason == "length"

    stopped_at_cap = runner_fixture(
        "text", "done", output_ids=tuple(range(49)) + (106,),
        cap=50, stop_token=106,
    )
    assert stopped_at_cap.finish_reason == "stop"
    expect_backend_error(
        lambda: runner_fixture(
            "text", "drift", output_ids=(1, 2), cap=50, stop_token=-1,
        ),
        "without a stop token before the generation cap",
    )
    marker_payload = "answer GEMMA4_QA_RESPONSE_END safely"
    assert runner_fixture("text", marker_payload).content == marker_payload

    fields = (
        b"runtime_ready=true\noutput_ids=818,106\n"
        b"GEMMA4_QA_EXECUTION_DONE prompt_tokens=23 output_steps=2 "
        b"stop_token=106 response_bytes=2 kv_loaded_tokens=0 "
        b"kv_saved_tokens=0 runtime_ready=true\n"
    )
    expect_backend_error(
        lambda: module.parse_runner_output(
            b"GEMMA4_QA_RESPONSE_BEGIN\nok\nGEMMA4_QA_RESPONSE_END\ntrailing",
            fields, "text", 50,
        ),
        "framing drift",
    )
    image_result = runner_fixture("image", "red mug")
    assert image_result.content == "red mug"
    newline_image_result = runner_fixture("image", "red\n")
    assert newline_image_result.content == "red"

    completion = module.Completion(
        result=traced,
        controls=module.Controls(
            512, 128, "low", True, 1.0, 230_686_720,
            "salt-greedy-v1", 1, 0.0, "00000000", 0, 1, "none",
        ),
        modality="text", elapsed_s=0.0,
        kv_cache=module.KVCacheOutcome("none", None, 0, 0),
    )
    incomplete = module._responses_payload(
        completion, model=module.DEFAULT_MODEL_ID,
        response_id="resp_test", created=0,
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["incomplete_details"] == {"reason": "max_output_tokens"}
    assert incomplete["x_salt"]["sampler"] == {
        "mode": "greedy", "sampler_abi": "salt-greedy-v1",
        "temperature": 0.0, "temperature_bits": "00000000",
        "seed": None, "top_k": 1, "rng_abi": "none", "rng_draw_count": 0,
    }

    tool_completion = module.Completion(
        result=tool_result, controls=completion.controls,
        modality="text", elapsed_s=0.0,
        kv_cache=module.KVCacheOutcome("none", None, 0, 0),
    )
    tool_payload = module._chat_payload(
        tool_completion, model=module.DEFAULT_MODEL_ID,
        completion_id="chatcmpl_tool", created=0,
    )
    tool_choice = tool_payload["choices"][0]
    assert tool_choice["finish_reason"] == "tool_calls"
    assert tool_choice["message"]["content"] is None
    assert tool_choice["message"]["tool_calls"][0]["type"] == "function"
    assert (
        tool_choice["message"]["tool_calls"][0]["function"]["arguments"]
        == '{"city":"Boston","days":2}'
    )

    expect_request_error(
        lambda: module.parse_runner_output(
            b"GEMMA4_QA_RESPONSE_BEGIN\nx\nGEMMA4_QA_RESPONSE_END\n",
            b"runtime_ready=false\n", "text", 50),
        "runtime readiness",
    )

    # File URLs are accepted only under an explicit root and copied to a private temp.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "image.ppm"
        source.write_bytes(ppm)
        with module.prepared_image(source.as_uri(), image_root=root) as prepared:
            assert prepared.path != source
            assert prepared.path.read_bytes() == ppm
        outside = Path(tmp).parent / "outside.ppm"
        outside.write_bytes(ppm)
        try:
            expect_request_error(
                lambda: module.prepared_image(
                    outside.as_uri(), image_root=root).__enter__(),
                "outside configured image root",
            )
        finally:
            outside.unlink(missing_ok=True)

    print("gemma4 server contract: PASS")


if __name__ == "__main__":
    main()
