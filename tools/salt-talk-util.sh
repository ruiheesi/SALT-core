#!/usr/bin/env bash
# Source this file once. The function always executes the latest client script.

salt_talk() {
    "${SALT_REPO_ROOT:-/Users/her2/projects/SALT-core}/tools/salt-talk.py" "$@"
}

salt_start() {
    "${SALT_REPO_ROOT:-/Users/her2/projects/SALT-core}/tools/salt-start.sh" "$@"
}

# Short form: st "your prompt"
st() {
    salt_talk "$@"
}

# Start the local server used by st.
st_start() {
    salt_start "$@"
}
