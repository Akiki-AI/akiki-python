#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TMPDIR="${TMPDIR:-$project_root/.cache/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_root/.cache/uv}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
python3 "$project_root/verify_dist.py" --all-wheels --require-native --require-arm64 --require-desktop
version="$(python3 "$project_root/verify_dist.py" --version)"
shopt -s nullglob
wheels=("$project_root/dist/akiki-${version}-"*.whl)
exec uv publish "$@" "${wheels[@]}" "$project_root/dist/akiki-${version}.tar.gz"
