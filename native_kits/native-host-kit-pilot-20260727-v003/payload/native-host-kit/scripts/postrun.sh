#!/usr/bin/env bash
set -euo pipefail

KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$KIT_ROOT"
QA_RUN_ID="ds-carla0916-native-pilot-v001-qa"
if [ -e "runs/$QA_RUN_ID" ]; then
  echo "Refusing to overwrite existing QA run: runs/$QA_RUN_ID" >&2
  exit 2
fi
export PYTHONPATH="$KIT_ROOT/source"
.venv/bin/python -m carla_vision.verification \
  "datasets/ds-carla0916-native-pilot-v001" --reject-unregistered
.venv/bin/python -m carla_vision.dataset.qa \
  --dataset "datasets/ds-carla0916-native-pilot-v001" \
  --runs-root runs \
  --run-id "$QA_RUN_ID"
.venv/bin/python -m carla_vision.verification \
  "runs/$QA_RUN_ID" --reject-unregistered
