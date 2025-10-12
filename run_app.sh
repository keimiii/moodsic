#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Docker Compose is required but not found. Install Docker Desktop or docker-compose." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- up --build
fi

echo "Running: ${COMPOSE_CMD[*]} $*"
"${COMPOSE_CMD[@]}" "$@"
