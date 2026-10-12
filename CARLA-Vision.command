#!/bin/sh
set -eu
ApplicationRoot=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$ApplicationRoot/deploy/desktop/launch.sh"
