#!/usr/bin/env bash
# Start the local authenticated Gemma server used by salt-talk.py.
set -euo pipefail

ROOT=${SALT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
HOST=${SALT_TALK_HOST:-127.0.0.1}
PORT=${SALT_TALK_PORT:-8090}
MODEL=${SALT_TALK_MODEL:-gemma-4-26b-a4b-it}
PYTHON=${SALT_TALK_PYTHON:-$(command -v python3)}
SOURCE_DIR=${GEMMA4_SOURCE_DIR:-$ROOT/build/models/gemma4-26b-a4b-mlx4}
POOL=${GEMMA4_POOL:-$ROOT/build/models/gemma4-26b-a4b-runtime/pool.bin}
RECEIPT=${GEMMA4_RECEIPT:-${POOL}.import.json}
AUTH_RECEIPT=${GEMMA4_AUTH_RECEIPT:-${POOL}.auth.json}
BUILD_RECEIPT=${GEMMA4_SERVER_BUILD_RECEIPT:-$ROOT/gemma4-server.build.json}
RUN_DIR=${SALT_TALK_RUN_DIR:-$ROOT/build/run}
PID_FILE=${SALT_TALK_PID_FILE:-$RUN_DIR/salt-talk-server.pid}
LOG_FILE=${SALT_TALK_LOG_FILE:-$RUN_DIR/salt-talk-server.log}
CONTEXT_TOKENS=${GEMMA4_CONTEXT_TOKENS:-20200}
MAX_OUTPUT_TOKENS=${GEMMA4_MAX_OUTPUT_TOKENS:-4096}
PREFILL_CHUNK_TOKENS=${GEMMA4_PREFILL_CHUNK_TOKENS:-1024}
TIMEOUT_S=${GEMMA4_TIMEOUT_S:-3600}
WORKERS=${GEMMA4_WORKERS:-4}
EXPERT_BUDGET_GB=${GEMMA4_EXPERT_BUDGET_GB:-2}
KV_BUDGET_GB=${GEMMA4_KV_BUDGET_GB:-1.25}
MEMORY_LIMIT_GB=${GEMMA4_MEMORY_LIMIT_GB:-6}
COPILOT_CACHE_ID=${GEMMA4_COPILOT_CACHE_ID:-copilot-default}
RECIPE=${SALT_GEMMA_PLATFORM_RECIPE:-mac-metal}
BASE_URL="http://$HOST:$PORT"

fail() {
    printf 'salt-start: %s\n' "$*" >&2
    exit 2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || \
      ${2:-} == "-h" || ${2:-} == "--help" ]]; then
    cat <<EOF
Usage: st start
       st restart
       st logs

Starts the local SALT Gemma server in the background at $BASE_URL.
The restart command safely replaces this repo's server and native engine.
The logs command follows the latest server log until interrupted.
Log: $LOG_FILE
EOF
    exit 0
fi
ACTION=${1:-start}
[[ $# -le 1 ]] || fail "unexpected arguments: $*"
[[ "$ACTION" == "start" || "$ACTION" == "restart" || "$ACTION" == "logs" ]] || \
    fail "expected 'start', 'restart', or 'logs', got: $ACTION"

if [[ "$ACTION" == "logs" ]]; then
    [[ -f "$LOG_FILE" ]] || fail "log does not exist yet: $LOG_FILE"
    exec tail -n "${SALT_TALK_LOG_LINES:-80}" -f "$LOG_FILE"
fi

[[ -x "$PYTHON" ]] || fail "Python is not executable: $PYTHON"
[[ -f "$ROOT/server/serve.py" ]] || fail "invalid repository root: $ROOT"
[[ -x "$ROOT/gemma4-server" ]] || fail "missing binary: $ROOT/gemma4-server"
for artifact in "$SOURCE_DIR/config.json" "$POOL" "$RECEIPT" \
        "$AUTH_RECEIPT" "$BUILD_RECEIPT"; do
    [[ -e "$artifact" ]] || fail "missing required artifact: $artifact"
done

listener_pid() {
    command -v lsof >/dev/null 2>&1 || return 1
    lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null
}

stop_server() {
    local pid command_text remaining
    pid=$(listener_pid || true)
    [[ -n "$pid" ]] || return 0
    [[ "$pid" != *$'\n'* && "$pid" =~ ^[0-9]+$ ]] || \
        fail "expected one listener PID on port $PORT, got: $pid"
    command_text=$(ps -p "$pid" -o command= 2>/dev/null || true)
    [[ "$command_text" == *"$ROOT/server/serve.py"* && \
       "$command_text" == *"--backend gemma4"* ]] || \
        fail "port $PORT is not owned by this repo's Gemma server"

    printf 'salt-start: stopping server and native engine (PID %s)\n' "$pid"
    kill -INT "$pid"
    for ((remaining = 0; remaining < 40; remaining++)); do
        [[ -z "$(listener_pid || true)" ]] && return 0
        sleep 0.25
    done
    kill -TERM "$pid" 2>/dev/null || true
    for ((remaining = 0; remaining < 20; remaining++)); do
        [[ -z "$(listener_pid || true)" ]] && return 0
        sleep 0.25
    done
    fail "server PID $pid did not release port $PORT"
}

if [[ "$ACTION" == "restart" ]]; then
    stop_server
elif curl --fail --silent --max-time 2 "$BASE_URL/healthz" >/dev/null 2>&1; then
    printf 'salt-start: already ready at %s (arena %sGB, KV %sGB, CTX%s/O%s, B%s, timeout %ss, cache %s)\n' \
        "$BASE_URL" "$EXPERT_BUDGET_GB" "$KV_BUDGET_GB" \
        "$CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_CHUNK_TOKENS" \
        "$TIMEOUT_S" "$COPILOT_CACHE_ID"
    exit 0
fi

if command -v lsof >/dev/null 2>&1 && \
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "port $PORT is already occupied; not starting a second server"
fi

mkdir -p "$RUN_DIR"
cd "$ROOT"

nohup env -u PYTHONPATH \
    SALT_GEMMA_PLATFORM_RECIPE="$RECIPE" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON" "$ROOT/server/serve.py" \
    --backend gemma4 \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --gemma-source-dir "$SOURCE_DIR" \
    --gemma-pool "$POOL" \
    --gemma-receipt "$RECEIPT" \
    --gemma-auth-receipt "$AUTH_RECEIPT" \
    --gemma-server-build-receipt "$BUILD_RECEIPT" \
    --gemma-workers "$WORKERS" \
    --gemma-expert-budget-gb "$EXPERT_BUDGET_GB" \
    --gemma-memory-limit-gb "$MEMORY_LIMIT_GB" \
    --gemma-context-tokens "$CONTEXT_TOKENS" \
    --gemma-max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --gemma-prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
    --gemma-timeout-s "$TIMEOUT_S" \
    --gemma-kv-budget-gb "$KV_BUDGET_GB" \
    --gemma-copilot-cache-id "$COPILOT_CACHE_ID" \
    > >("$PYTHON" "$ROOT/tools/salt-logstamp.py" >>"$LOG_FILE") \
    2>&1 </dev/null &
server_pid=$!
printf '%s\n' "$server_pid" >"$PID_FILE"

for ((attempt = 0; attempt < 120; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        printf 'salt-start: server exited during startup; log follows:\n' >&2
        tail -n 30 "$LOG_FILE" >&2 || true
        exit 1
    fi
    if curl --fail --silent --max-time 2 "$BASE_URL/healthz" >/dev/null 2>&1; then
        printf 'salt-start: ready at %s (PID %s, arena %sGB, KV %sGB, CTX%s/O%s, B%s, timeout %ss, cache %s)\n' \
            "$BASE_URL" "$server_pid" "$EXPERT_BUDGET_GB" "$KV_BUDGET_GB" \
            "$CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_CHUNK_TOKENS" \
            "$TIMEOUT_S" "$COPILOT_CACHE_ID"
        printf 'salt-start: log %s\n' "$LOG_FILE"
        exit 0
    fi
    sleep 0.25
done

printf 'salt-start: timed out waiting for %s; see %s\n' \
    "$BASE_URL" "$LOG_FILE" >&2
exit 1
