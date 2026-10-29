#!/usr/bin/env bash
# Configure and run the local model server used by salt-talk.py.
set -euo pipefail

ROOT=${SALT_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
HOST=${SALT_TALK_HOST:-127.0.0.1}
PORT=${SALT_TALK_PORT:-8090}
PYTHON=${SALT_TALK_PYTHON:-$(command -v python3)}
RUN_DIR=${SALT_TALK_RUN_DIR:-$ROOT/build/run}
CONFIG_FILE=${SALT_TALK_CONFIG_FILE:-$RUN_DIR/salt-talk-model}
PROFILE=${SALT_TALK_PROFILE:-}
if [[ -z "$PROFILE" && -f "$CONFIG_FILE" ]]; then
    PROFILE=$(<"$CONFIG_FILE")
fi
PROFILE=${PROFILE:-gemma}
INVALID_PROFILE=
case "$PROFILE" in
    gemma|gemma4|gemma-4-26b-a4b-it) PROFILE=gemma ;;
    liquid|lfm|lfm2|lfm2.5-24b-a2b) PROFILE=liquid ;;
    *) INVALID_PROFILE=$PROFILE; PROFILE=gemma ;;
esac

GEMMA_MODEL_ID=gemma-4-26b-a4b-it
LIQUID_MODEL_ID=lfm2.5-24b-a2b
if [[ "$PROFILE" == "liquid" ]]; then
    MODEL=${SALT_TALK_MODEL:-$LIQUID_MODEL_ID}
else
    MODEL=${SALT_TALK_MODEL:-$GEMMA_MODEL_ID}
fi
SOURCE_DIR=${GEMMA4_SOURCE_DIR:-$ROOT/build/models/gemma4-26b-a4b-mlx4}
POOL=${GEMMA4_POOL:-$ROOT/build/models/gemma4-26b-a4b-runtime/pool.bin}
RECEIPT=${GEMMA4_RECEIPT:-${POOL}.import.json}
AUTH_RECEIPT=${GEMMA4_AUTH_RECEIPT:-${POOL}.auth.json}
ROTATION_RUNNER=${GEMMA4_SERVER_RUNNER:-$ROOT/rotation/bin/gemma4-server}
ROTATION_CONFIG=${GEMMA4_ROTATION_CONFIG:-$ROOT/rotation/engine.config}
BUILD_RECEIPT=${GEMMA4_SERVER_BUILD_RECEIPT:-${ROTATION_RUNNER}.build.json}
LFM_SOURCE=${LFM_SOURCE:-$ROOT/build/models/lfm2-24b-a2b-mlx4}
LFM_PACKAGE=${LFM_PACKAGE:-$ROOT/build/models/lfm2-24b-a2b-runtime}
LFM_RUNNER=${LFM_RUNNER:-$ROOT/build/bin/lfm-server}
LFM_CONTEXT=${LFM_CONTEXT:-2048}
LFM_PREFILL_B=${LFM_PREFILL_B:-32}
LFM_WORKERS=${LFM_WORKERS:-8}
LFM_EXPERT_SLOTS=${LFM_EXPERT_SLOTS:-376}
LFM_MEMORY_LIMIT_GB=${LFM_MEMORY_LIMIT_GB:-8}
LFM_TIMEOUT_S=${LFM_TIMEOUT_S:-3600}
PID_FILE=${SALT_TALK_PID_FILE:-$RUN_DIR/salt-talk-server.pid}
LOG_FILE=${SALT_TALK_LOG_FILE:-$RUN_DIR/salt-talk-server.log}
CONTEXT_TOKENS=${GEMMA4_CONTEXT_TOKENS:-20200}
MAX_OUTPUT_TOKENS=${GEMMA4_MAX_OUTPUT_TOKENS:-4096}
PREFILL_CHUNK_TOKENS=${GEMMA4_PREFILL_CHUNK_TOKENS:-1024}
TIMEOUT_S=${GEMMA4_TIMEOUT_S:-3600}
WORKERS=${GEMMA4_WORKERS:-4}
EXPERT_BUDGET_GB=${GEMMA4_EXPERT_BUDGET_GB:-4}
KV_BUDGET_GB=${GEMMA4_KV_BUDGET_GB:-1.25}
MEMORY_LIMIT_GB=${GEMMA4_MEMORY_LIMIT_GB:-8}
COPILOT_CACHE_ID=${GEMMA4_COPILOT_CACHE_ID-copilot-default}
SESSION_MODE=${GEMMA4_SESSION_MODE:-stateful}
RECIPE=${SALT_GEMMA_PLATFORM_RECIPE:-mac-metal}
RECIPE_CONFIG="$ROOT/models/gemma4-26b-a4b/configs/$RECIPE/engine.config"
BASE_URL="http://$HOST:$PORT"
CACHE_LABEL=${COPILOT_CACHE_ID:-off}

fail() {
    printf 'salt-start: %s\n' "$*" >&2
    exit 2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || \
      ${2:-} == "-h" || ${2:-} == "--help" ]]; then
    cat <<EOF
Usage: st start [-x N]
       st restart [-x N]
       st setup
       st down
       st cli-start [-x N]
       st logs
       st config [model] [gemma|liquid]

Starts the configured local SALT model server in the background at $BASE_URL.
The restart command safely replaces this repo's server and native engine.
The setup command fully rehashes Gemma artifacts and records this machine/path
as the authenticated baseline. Run it after copying or restoring the model.
The down command safely stops this repo's server and native engine.
The cli-start command selects stateless Gemma chat without Copilot cache.
The -x/--target-x option verifies the X width owned by rotation/engine.config.
The config command shows or persists the model used by st and st start.
The logs command follows the latest server log until interrupted.
Log: $LOG_FILE
EOF
    exit 0
fi
ACTION=${1:-start}
shift || true
[[ "$ACTION" == "start" || "$ACTION" == "restart" || \
   "$ACTION" == "setup" || \
   "$ACTION" == "down" || "$ACTION" == "logs" || \
   "$ACTION" == "config" ]] || \
    fail "expected 'start', 'restart', 'setup', 'down', 'logs', or 'config', got: $ACTION"

if [[ "$ACTION" == "config" ]]; then
    if [[ ${1:-} == "model" ]]; then
        shift
    fi
    if [[ $# -gt 1 ]]; then
        fail "usage: st config [model] [gemma|liquid]"
    fi
    if [[ $# -eq 1 ]]; then
        case "$1" in
            gemma|gemma4|gemma-4-26b-a4b-it) PROFILE=gemma ;;
            liquid|lfm|lfm2|lfm2.5-24b-a2b) PROFILE=liquid ;;
            *) fail "unknown model '$1'; choose gemma or liquid" ;;
        esac
        mkdir -p "$RUN_DIR"
        config_temp=$(mktemp "$CONFIG_FILE.tmp.XXXXXX")
        chmod 600 "$config_temp"
        printf '%s\n' "$PROFILE" >"$config_temp"
        mv "$config_temp" "$CONFIG_FILE"
        printf 'salt-start: configured model %s\n' "$PROFILE"
        INVALID_PROFILE=
    fi
    [[ -z "$INVALID_PROFILE" ]] || \
        fail "invalid configured model profile: $INVALID_PROFILE"
    if [[ "$PROFILE" == "liquid" ]]; then
        ready=yes
        [[ -x "$LFM_RUNNER" && -f "$LFM_RUNNER.build.json" && \
           -f "$LFM_PACKAGE/auth.json" && \
           -f "$LFM_PACKAGE/tokenizer.runtime.auth.json" ]] || ready=no
        printf 'model=liquid id=%s ready=%s\n' "$LIQUID_MODEL_ID" "$ready"
        printf 'runner=%s\npackage=%s\n' "$LFM_RUNNER" "$LFM_PACKAGE"
        if [[ "$ready" == "no" ]]; then
            printf 'setup: see models/lfm2-24b-a2b/README.md (source=%s package=%s)\n' "$LFM_SOURCE" "$LFM_PACKAGE"
            printf 'build: LFM_BACKEND=metal LFM_BIN_DIR=%s make lfm-server\n' "$ROOT/build/bin"
        fi
    else
        ready=yes
        [[ -x "$ROTATION_RUNNER" && -f "$BUILD_RECEIPT" && \
           -f "$ROTATION_CONFIG" && -f "$AUTH_RECEIPT" ]] || ready=no
        printf 'model=gemma id=%s runtime=rotation ready=%s\n' \
            "$GEMMA_MODEL_ID" "$ready"
        printf 'runner=%s\nrotation-config=%s\n' \
            "$ROTATION_RUNNER" "$ROTATION_CONFIG"
        if [[ "$ready" == "no" ]]; then
            printf 'build: bash rotation/build-server.sh\n'
        fi
    fi
    printf 'config=%s\n' "$CONFIG_FILE"
    exit 0
fi

if [[ -n "$INVALID_PROFILE" && "$ACTION" != "down" && \
      "$ACTION" != "logs" ]]; then
    fail "invalid configured model profile: $INVALID_PROFILE"
fi

TARGET_X_OVERRIDE=
while [[ $# -gt 0 ]]; do
    case "$1" in
        -x|--target-x)
            [[ $# -ge 2 ]] || fail "$1 requires a row count"
            TARGET_X_OVERRIDE=$2
            shift 2
            ;;
        *)
            fail "unexpected argument: $1"
            ;;
    esac
done

if [[ "$ACTION" == "logs" ]]; then
    [[ -z "$TARGET_X_OVERRIDE" ]] || fail "-x is not valid with logs"
    [[ -f "$LOG_FILE" ]] || fail "log does not exist yet: $LOG_FILE"
    exec tail -n "${SALT_TALK_LOG_LINES:-80}" -f "$LOG_FILE"
fi

[[ -x "$PYTHON" ]] || fail "Python is not executable: $PYTHON"
[[ -f "$ROOT/server/serve.py" ]] || fail "invalid repository root: $ROOT"

recipe_value() {
    local key=$1 value
    value=$(sed -n "s/^${key}=//p" "$RECIPE_CONFIG" | tail -n 1)
    [[ -n "$value" ]] || fail "recipe $RECIPE does not define $key"
    printf '%s' "$value"
}

rotation_value() {
    local key=$1 value
    value=$(sed -n "s/^${key}=//p" "$ROTATION_CONFIG" | tail -n 1)
    [[ -n "$value" ]] || fail "rotation config does not define $key"
    printf '%s' "$value"
}

listener_pid() {
    command -v lsof >/dev/null 2>&1 || return 1
    lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null
}

stop_server() {
    local pid command_text command_cwd remaining
    pid=$(listener_pid || true)
    [[ -n "$pid" ]] || return 1
    [[ "$pid" != *$'\n'* && "$pid" =~ ^[0-9]+$ ]] || \
        fail "expected one listener PID on port $PORT, got: $pid"
    command_text=$(ps -p "$pid" -o command= 2>/dev/null || true)
    command_cwd=$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | \
        sed -n 's/^n//p' | head -n 1)
    [[ ( "$command_text" == *"--backend gemma4"* || \
         "$command_text" == *"--backend lfm"* ) && \
       ( "$command_text" == *"$ROOT/server/serve.py"* || \
         ( "$command_cwd" == "$ROOT" && \
           "$command_text" == *"server/serve.py"* ) ) ]] || \
        fail "port $PORT is not owned by this repo's SALT model server"

    printf 'salt-start: stopping server and native engine (PID %s)\n' "$pid"
    kill -INT "$pid"
    for ((remaining = 0; remaining < 40; remaining++)); do
        if [[ -z "$(listener_pid || true)" ]] && ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.25
    done
    kill -TERM "$pid" 2>/dev/null || true
    for ((remaining = 0; remaining < 20; remaining++)); do
        if [[ -z "$(listener_pid || true)" ]] && ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.25
    done
    fail "server PID $pid did not release port $PORT"
}

if [[ "$ACTION" == "down" ]]; then
    [[ -z "$TARGET_X_OVERRIDE" ]] || fail "-x is not valid with down"
    if stop_server; then
        rm -f "$PID_FILE"
        printf 'salt-start: stopped server and native engine\n'
    else
        rm -f "$PID_FILE"
        printf 'salt-start: already stopped\n'
    fi
    exit 0
fi

if [[ "$PROFILE" == "gemma" ]]; then
    [[ "$(uname -s)" == "Darwin" ]] || \
        fail "Gemma rotation currently requires Darwin"
    [[ -x "$ROTATION_RUNNER" ]] || \
        fail "missing rotation runner: $ROTATION_RUNNER (run: bash rotation/build-server.sh)"
    [[ -f "$BUILD_RECEIPT" ]] || \
        fail "missing rotation build receipt: $BUILD_RECEIPT (run: bash rotation/build-server.sh)"
    [[ -f "$ROTATION_CONFIG" ]] || \
        fail "missing rotation config: $ROTATION_CONFIG"
    [[ -f "$RECIPE_CONFIG" ]] || fail "missing recipe config: $RECIPE_CONFIG"
    for artifact in "$SOURCE_DIR/config.json" "$POOL" "$RECEIPT" \
            "$BUILD_RECEIPT"; do
        [[ -e "$artifact" ]] || fail "missing required artifact: $artifact"
    done
    if [[ "$ACTION" != "setup" ]]; then
        [[ -f "$AUTH_RECEIPT" ]] || \
            fail "missing authentication baseline: $AUTH_RECEIPT (run: st setup)"
    fi
    TARGET_ROUTE_N=$(recipe_value SALT_TARGET_ROUTE_N)
    TARGET_ROUTE_F=$(recipe_value SALT_TARGET_ROUTE_F)
    TARGET_ROUTE_Q=$(recipe_value SALT_TARGET_ROUTE_Q)
    ROTATION_EXPERT_SEATS=$(rotation_value decode_expert_seats)
    [[ "$ROTATION_EXPERT_SEATS" =~ ^[1-9][0-9]*$ ]] || \
        fail "rotation decode_expert_seats must be a canonical positive integer"
    TARGET_X=$(rotation_value native_target_x)
    [[ "$TARGET_X" =~ ^[1-9][0-9]*$ ]] || \
        fail "rotation native_target_x must be a canonical positive integer"
    (( TARGET_X <= 128 )) || fail "rotation native_target_x must be between 1 and 128"
    TARGET_WARM_X=$TARGET_X
    if [[ -n "$TARGET_X_OVERRIDE" ]]; then
        [[ "$TARGET_X_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || \
            fail "X must be a canonical positive integer"
        [[ "$TARGET_X_OVERRIDE" == "$TARGET_X" ]] || \
            fail "rotation owns X=$TARGET_X; set native_target_x in $ROTATION_CONFIG and restart"
    fi
else
    [[ "$ACTION" != "setup" ]] || \
        fail "st setup currently establishes the Gemma baseline; run 'st config gemma' first"
    [[ -z "$TARGET_X_OVERRIDE" ]] || fail "-x is Gemma-only"
    [[ -x "$LFM_RUNNER" ]] || fail "missing Liquid Metal runner: $LFM_RUNNER (build with LFM_BACKEND=metal LFM_BIN_DIR=$ROOT/build/bin make lfm-server)"
    [[ -f "$LFM_RUNNER.build.json" ]] || fail "missing Liquid build receipt: $LFM_RUNNER.build.json"
    [[ -f "$LFM_PACKAGE/auth.json" ]] || fail "missing Liquid runtime package: $LFM_PACKAGE (run models/lfm2-24b-a2b/setup.py fetch, import, and prepare-tokenizer)"
    [[ -f "$LFM_PACKAGE/tokenizer.runtime.auth.json" ]] || fail "missing Liquid runtime tokenizer; run setup.py prepare-tokenizer --package $LFM_PACKAGE"
fi

build_gemma_server_command() {
    server_command=(
        env -u PYTHONPATH -u GEMMA4_COPILOT_CACHE_ID
        SALT_GEMMA_PLATFORM_RECIPE="$RECIPE"
        PYTHONUNBUFFERED=1
        "$PYTHON" "$ROOT/server/serve.py"
        --backend gemma4
        --host "$HOST"
        --port "$PORT"
        --model "$MODEL"
        --gemma-session-mode "$SESSION_MODE"
        --gemma-source-dir "$SOURCE_DIR"
        --gemma-pool "$POOL"
        --gemma-receipt "$RECEIPT"
        --gemma-auth-receipt "$AUTH_RECEIPT"
        --gemma-server-runner "$ROTATION_RUNNER"
        --gemma-server-build-receipt "$BUILD_RECEIPT"
        --gemma-workers "$WORKERS"
        --gemma-expert-budget-gb "$EXPERT_BUDGET_GB"
        --gemma-memory-limit-gb "$MEMORY_LIMIT_GB"
        --gemma-context-tokens "$CONTEXT_TOKENS"
        --gemma-max-output-tokens "$MAX_OUTPUT_TOKENS"
        --gemma-prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS"
        --gemma-target-route-n "$TARGET_ROUTE_N"
        --gemma-target-x "$TARGET_X"
        --gemma-target-warm-x "$TARGET_WARM_X"
        --gemma-target-route-f "$TARGET_ROUTE_F"
        --gemma-target-route-q "$TARGET_ROUTE_Q"
        --gemma-timeout-s "$TIMEOUT_S"
        --gemma-kv-budget-gb "$KV_BUDGET_GB"
    )
    if [[ -n "$COPILOT_CACHE_ID" ]]; then
        server_command+=(--gemma-copilot-cache-id "$COPILOT_CACHE_ID")
    fi
}

if [[ "$ACTION" == "setup" ]]; then
    [[ -z "$TARGET_X_OVERRIDE" ]] || fail "-x is not valid with setup"
    stop_server || true
    rm -f "$PID_FILE"
    mkdir -p "$RUN_DIR"
    cd "$ROOT"
    "$PYTHON" "$ROOT/tools/gemma4-rebase-receipt.py" \
        --pool "$POOL" --receipt "$RECEIPT"
    build_gemma_server_command
    printf 'salt-start: fully rehashing Gemma source, expert pool, and runtime...\n'
    printf 'salt-start: this reads about 28GB and may take several minutes.\n'
    "${server_command[@]}" --gemma-verify-only | \
        sed -n '/^GEMMA4_SERVER_SETUP_OK$/p'
    printf 'salt-start: Gemma authentication baseline established.\n'
    printf 'salt-start: receipt %s\n' "$AUTH_RECEIPT"
    printf 'salt-start: run `st start` to launch the server.\n'
    exit 0
fi

if [[ "$ACTION" == "restart" ]]; then
    stop_server || true
elif curl --fail --silent --max-time 2 "$BASE_URL/healthz" >/dev/null 2>&1; then
    running_model=$(curl --fail --silent --max-time 2 "$BASE_URL/healthz" | \
        "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("model", ""))' \
        2>/dev/null || true)
    [[ "$running_model" == "$MODEL" ]] || \
        fail "server is running model '$running_model', configured model is '$MODEL'; use 'st restart'"
    if [[ "$PROFILE" == "gemma" ]]; then
        running_pid=$(listener_pid || true)
        running_command=
        if [[ -n "$running_pid" && "$running_pid" != *$'\n'* ]]; then
            running_command=$(ps -p "$running_pid" -o command= 2>/dev/null || true)
        fi
        [[ "$running_command" == *"--gemma-server-runner $ROTATION_RUNNER"* ]] || \
            fail "Gemma is running through the legacy runner; use 'st restart' to activate rotation"
    fi
    if [[ -n "$TARGET_X_OVERRIDE" ]]; then
        fail "server is already running; use 'st restart -x $TARGET_X' to change X"
    fi
    if [[ "$PROFILE" == "gemma" ]]; then
        printf 'salt-start: already ready at %s (model gemma, runtime rotation, mode %s, rotation-seats %s, expert-budget %sGB, memory-limit %sGB, KV %sGB, CTX%s/O%s, B%s, X%s, timeout %ss, cache %s)\n' \
            "$BASE_URL" "$SESSION_MODE" "$ROTATION_EXPERT_SEATS" "$EXPERT_BUDGET_GB" "$MEMORY_LIMIT_GB" "$KV_BUDGET_GB" \
            "$CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_CHUNK_TOKENS" "$TARGET_X" \
            "$TIMEOUT_S" "$CACHE_LABEL"
    else
        printf 'salt-start: already ready at %s (model liquid, replay sessions, CTX%s/O512, B%s, workers %s, slots %s, memory-limit %sGB, timeout %ss)\n' \
            "$BASE_URL" "$LFM_CONTEXT" "$LFM_PREFILL_B" "$LFM_WORKERS" \
            "$LFM_EXPERT_SLOTS" "$LFM_MEMORY_LIMIT_GB" "$LFM_TIMEOUT_S"
    fi
    exit 0
fi

if command -v lsof >/dev/null 2>&1 && \
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "port $PORT is already occupied; not starting a second server"
fi

mkdir -p "$RUN_DIR"
cd "$ROOT"

launch_log_bytes=0
if [[ -f "$LOG_FILE" ]]; then
    launch_log_bytes=$(wc -c <"$LOG_FILE")
fi

if [[ "$PROFILE" == "gemma" ]]; then
    build_gemma_server_command
else
    server_command=(
        env -u PYTHONPATH
        PYTHONUNBUFFERED=1
        "$PYTHON" "$ROOT/server/serve.py"
        --backend lfm
        --host "$HOST"
        --port "$PORT"
        --model "$MODEL"
        --lfm-package "$LFM_PACKAGE"
        --lfm-runner "$LFM_RUNNER"
        --lfm-context "$LFM_CONTEXT"
        --lfm-batch "$LFM_PREFILL_B"
        --lfm-workers "$LFM_WORKERS"
        --lfm-expert-slots "$LFM_EXPERT_SLOTS"
        --lfm-memory-limit-gb "$LFM_MEMORY_LIMIT_GB"
        --lfm-timeout-s "$LFM_TIMEOUT_S"
    )
fi

nohup "$PYTHON" "$ROOT/tools/salt-logstamp.py" \
    --output "$LOG_FILE" -- \
    "${server_command[@]}" \
    >/dev/null 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "$server_pid" >"$PID_FILE"

for ((attempt = 0; attempt < 120; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        printf 'salt-start: server exited during startup; log follows:\n' >&2
        tail -n 30 "$LOG_FILE" >&2 || true
        if tail -c "+$((launch_log_bytes + 1))" "$LOG_FILE" 2>/dev/null | \
                grep -Eq 'cached file identity drift|authenticated package receipt drift'; then
            printf 'salt-start: authentication baseline is stale; run `st setup`, then `st start`.\n' >&2
        fi
        exit 1
    fi
    if curl --fail --silent --max-time 2 "$BASE_URL/healthz" >/dev/null 2>&1; then
        if [[ "$PROFILE" == "gemma" ]]; then
            printf 'salt-start: ready at %s (PID %s, model gemma, runtime rotation, mode %s, rotation-seats %s, expert-budget %sGB, memory-limit %sGB, KV %sGB, CTX%s/O%s, B%s, X%s, timeout %ss, cache %s)\n' \
                "$BASE_URL" "$server_pid" "$SESSION_MODE" "$ROTATION_EXPERT_SEATS" "$EXPERT_BUDGET_GB" "$MEMORY_LIMIT_GB" "$KV_BUDGET_GB" \
                "$CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_CHUNK_TOKENS" "$TARGET_X" \
                "$TIMEOUT_S" "$CACHE_LABEL"
        else
            printf 'salt-start: ready at %s (PID %s, model liquid, replay sessions, CTX%s/O512, B%s, workers %s, slots %s, memory-limit %sGB, timeout %ss)\n' \
                "$BASE_URL" "$server_pid" "$LFM_CONTEXT" "$LFM_PREFILL_B" \
                "$LFM_WORKERS" "$LFM_EXPERT_SLOTS" "$LFM_MEMORY_LIMIT_GB" "$LFM_TIMEOUT_S"
        fi
        printf 'salt-start: log %s\n' "$LOG_FILE"
        exit 0
    fi
    sleep 0.25
done

printf 'salt-start: timed out waiting for %s; see %s\n' \
    "$BASE_URL" "$LOG_FILE" >&2
exit 1
