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
CONTEXT_TOKENS=${GEMMA4_CONTEXT_TOKENS:-16384}
MAX_OUTPUT_TOKENS=${GEMMA4_MAX_OUTPUT_TOKENS:-2048}
WORKERS=${GEMMA4_WORKERS:-4}
KV_BUDGET_GB=${GEMMA4_KV_BUDGET_GB:-1.25}
RECIPE=${SALT_GEMMA_PLATFORM_RECIPE:-mac-metal}
BASE_URL="http://$HOST:$PORT"

fail() {
    printf 'salt-start: %s\n' "$*" >&2
    exit 2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    cat <<EOF
Usage: st_start

Starts the local SALT Gemma server in the background at $BASE_URL.
Log: $LOG_FILE
EOF
    exit 0
fi
[[ $# -eq 0 ]] || fail "unexpected arguments: $*"

[[ -x "$PYTHON" ]] || fail "Python is not executable: $PYTHON"
[[ -f "$ROOT/server/serve.py" ]] || fail "invalid repository root: $ROOT"
[[ -x "$ROOT/gemma4-server" ]] || fail "missing binary: $ROOT/gemma4-server"
for artifact in "$SOURCE_DIR/config.json" "$POOL" "$RECEIPT" \
        "$AUTH_RECEIPT" "$BUILD_RECEIPT"; do
    [[ -e "$artifact" ]] || fail "missing required artifact: $artifact"
done

if curl --fail --silent --max-time 2 "$BASE_URL/healthz" >/dev/null 2>&1; then
    printf 'salt-start: already ready at %s (CTX%s/O%s)\n' \
        "$BASE_URL" "$CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS"
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
    --gemma-context-tokens "$CONTEXT_TOKENS" \
    --gemma-max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --gemma-kv-budget-gb "$KV_BUDGET_GB" \
    >>"$LOG_FILE" 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "$server_pid" >"$PID_FILE"

for ((attempt = 0; attempt < 120; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        printf 'salt-start: server exited during startup; log follows:\n' >&2
        tail -n 30 "$LOG_FILE" >&2 || true
        exit 1
    fi
    if curl --fail --silent --max-time 2 "$BASE_URL/healthz" >/dev/null 2>&1; then
        printf 'salt-start: ready at %s (PID %s, CTX%s/O%s)\n' \
            "$BASE_URL" "$server_pid" "$CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS"
        printf 'salt-start: log %s\n' "$LOG_FILE"
        exit 0
    fi
    sleep 0.25
done

printf 'salt-start: timed out waiting for %s; see %s\n' \
    "$BASE_URL" "$LOG_FILE" >&2
exit 1
