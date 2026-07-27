#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 READY_PREFLIGHT_RUN_ID I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK" >&2
  exit 2
fi
if [ "$2" != "I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK" ]; then
  echo "explicit destructive confirmation token is required" >&2
  exit 3
fi
KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$KIT_ROOT"
if [ -e "datasets/ds-carla0916-native-pilot-v001" ]; then
  echo "Refusing to overwrite existing dataset: datasets/ds-carla0916-native-pilot-v001" >&2
  exit 4
fi
export PYTHONPATH="$KIT_ROOT/source"
.venv/bin/python -m carla_vision.native.host_gate \
  --kit-plan kit-plan.json \
  --preflight "runs/$1"
.venv/bin/python -m carla_vision.native.worker \
  --scenario-plan "scenario-plan/scenario-plan-native-integration-pilot-v1" \
  --dataset-id "ds-carla0916-native-pilot-v001" \
  --datasets-root datasets \
  --host "172.20.10.7" \
  --port "2000" \
  --partition train \
  --max-episodes 1 \
  --timeout 30 \
  --sensor-timeout 10 \
  --acknowledge-exclusive-tick-owner
