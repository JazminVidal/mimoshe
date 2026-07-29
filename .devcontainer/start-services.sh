#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspaces/moshi

exec "$WORKSPACE/start-server.sh" --background
