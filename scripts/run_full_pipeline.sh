#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/protocol_50_20_41.json}"
python -m rbafl validate-config --config "${CONFIG}"
python -m rbafl all --config "${CONFIG}"

