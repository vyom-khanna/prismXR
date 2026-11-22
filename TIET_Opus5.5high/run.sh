#!/usr/bin/env bash
# Single-video inference entry point:  ./run.sh <video.mp4> <out.json>
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <video.mp4> <out.json>" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PRISM_PYTHON:-python}"
export PYTHONPATH="$HERE/code${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m prism.join_weights   # first run only: rejoin weights/*.partNNN (GitHub 100 MB file limit), MD5-checked
exec "$PY" -m prism.pipeline "$1" "$2"
