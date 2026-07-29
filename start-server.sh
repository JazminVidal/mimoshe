#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${MOSHI_PORT:-8998}"
HOST="${MOSHI_HOST:-0.0.0.0}"
HF_REPO="${HF_REPO:-kyutai/moshiko-pytorch-bf16}"
BACKGROUND=false
BUILD_CLIENT=true
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Build the custom Moshi UI and run it together with the Python model server.

Usage:
  ./start-server.sh [--background] [--no-build] [-- SERVER_ARGUMENTS...]

Environment:
  MOSHI_HOST       Bind address (default: 0.0.0.0)
  MOSHI_PORT       HTTP/HTTPS port (default: 8998)
  HF_REPO          Hugging Face model repository
  MOSHI_SSL_DIR    Directory containing cert.pem and key.pem; enables HTTPS

Normal remote access:
  ./start-server.sh
  Open http://<container-host-LAN-IP>:8998 on the remote computer.

Do not use https:// unless MOSHI_SSL_DIR was set when starting the server.
EOF
}

while (($#)); do
  case "$1" in
    --background)
      BACKGROUND=true
      shift
      ;;
    --no-build)
      BUILD_CLIENT=false
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown launcher option: $1" >&2
      echo "Pass Moshi server options after --." >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$BUILD_CLIENT" == true ]]; then
  if [[ ! -d "$REPO_ROOT/client/node_modules" ]]; then
    echo "Installing web client dependencies"
    (cd "$REPO_ROOT/client" && npm ci)
  fi
  echo "Building the custom Moshi UI"
  (cd "$REPO_ROOT/client" && npm run build)
elif [[ ! -f "$REPO_ROOT/client/dist/index.html" ]]; then
  echo "client/dist is missing; run without --no-build first." >&2
  exit 1
fi

mapfile -t SERVER_PIDS < <(
  pgrep -f "python(3)? -m moshi\.server.*--port ${PORT}([[:space:]]|$)" || true
)
if ((${#SERVER_PIDS[@]})); then
  if curl --fail --silent --max-time 2 "http://127.0.0.1:${PORT}/" |
      grep --quiet "Moshi Live Lab"; then
    echo "Moshi is already serving the custom UI on port ${PORT}."
    echo "Open http://<container-host-LAN-IP>:${PORT}"
    exit 0
  fi
  echo "Port ${PORT} is already used by Moshi PID(s): ${SERVER_PIDS[*]}" >&2
  echo "That process is not serving this repository's custom UI." >&2
  echo "Stop the listed PID, then run ./start-server.sh again." >&2
  exit 1
fi

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$REPO_ROOT/moshi/.venv/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "No project Python environment found." >&2
  echo "Create it with: uv venv .venv && uv pip install --python .venv/bin/python -e moshi" >&2
  exit 1
fi

SERVER_COMMAND=(
  "$PYTHON_BIN"
  -m moshi.server
  --host "$HOST"
  --port "$PORT"
  --static "$REPO_ROOT/client/dist"
  --hf-repo "$HF_REPO"
)

PROTOCOL=http
if [[ -n "${MOSHI_SSL_DIR:-}" ]]; then
  if [[ ! -f "$MOSHI_SSL_DIR/cert.pem" || ! -f "$MOSHI_SSL_DIR/key.pem" ]]; then
    echo "MOSHI_SSL_DIR must contain cert.pem and key.pem." >&2
    exit 1
  fi
  SERVER_COMMAND+=(--ssl "$MOSHI_SSL_DIR")
  PROTOCOL=https
fi
SERVER_COMMAND+=("${EXTRA_ARGS[@]}")

echo "Starting Moshi model server and custom UI"
echo "Local URL:  ${PROTOCOL}://localhost:${PORT}"
echo "Remote URL: ${PROTOCOL}://<container-host-LAN-IP>:${PORT}"

if [[ "$BACKGROUND" == true ]]; then
  LOG_PATH="${MOSHI_LOG_PATH:-/tmp/moshi-server-${PORT}.log}"
  nohup "${SERVER_COMMAND[@]}" >"$LOG_PATH" 2>&1 </dev/null &
  SERVER_PID=$!
  sleep 2
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Moshi failed to start. See $LOG_PATH" >&2
    tail -n 40 "$LOG_PATH" >&2
    exit 1
  fi
  echo "Moshi is loading in PID ${SERVER_PID}. Follow startup with:"
  echo "  tail -f ${LOG_PATH}"
  exit 0
fi

exec "${SERVER_COMMAND[@]}"
