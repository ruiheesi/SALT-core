#!/usr/bin/env bash
# Source this file once. The function always executes the latest client script.

salt_talk() {
    "${SALT_REPO_ROOT:-/Users/her2/projects/SALT-core}/tools/salt-talk.py" "$@"
}

salt_start() {
    "${SALT_REPO_ROOT:-/Users/her2/projects/SALT-core}/tools/salt-start.sh" "$@"
}

# Short forms:
#   st start [-x N]
#   st restart [-x N]
#   st cli-start [-x N]
#   st logs
#   st "your prompt"
#   st session
#   st session -a reader -mo 512
#     /file "path/to/file" "optional question"
#     /terminal "ls -la" "optional question"
#     /save work snapshots the live SALT KV as a named checkpoint
#     /load work restores it without replaying conversation text
#     /stats shows session/KV/context status on demand
#   st -mo 1024 "your prompt"
#   st --new-session "start a fresh conversation"
#   st -a reader "read this argument carefully"
#   st -a reader script.sh
#   st -a reader script.sh "what assumptions does this make?"
#   st agents
#   st summary README.md
#   cat notes.txt | st summary -
#   st find "where is cache import handled?" server/
#   st vision screenshot.png "what is wrong here?"
st() {
    if [[ ${1:-} == "start" || ${1:-} == "restart" || \
          ${1:-} == "cli-start" || ${1:-} == "logs" ]]; then
        salt_start "$@"
        return
    fi
    salt_talk "$@"
}

# Start the local server used by st.
st_start() {
    salt_start "$@"
}
