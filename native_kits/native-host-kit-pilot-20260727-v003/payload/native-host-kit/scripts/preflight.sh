#!/usr/bin/env bash
set -euo pipefail

KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$KIT_ROOT"
RUN_ID=${1:-native-preflight-native-host-kit-pilot-20260727-v003}
if [ "$#" -gt 0 ]; then shift; fi
if [ -e "runs/$RUN_ID" ]; then
  echo "Refusing to overwrite existing preflight: runs/$RUN_ID" >&2
  exit 2
fi
PYTHONPATH="$KIT_ROOT/source" .venv/bin/python -m carla_vision.native.preflight \
  --scenario-plan "scenario-plan/scenario-plan-native-integration-pilot-v1" \
  --dataset-id "ds-carla0916-native-pilot-v001" \
  --runs-root runs \
  --run-id "$RUN_ID" \
  --host "172.20.10.7" \
  --port "2000" \
  --partition train \
  --max-episodes 1 \
  "$@"
