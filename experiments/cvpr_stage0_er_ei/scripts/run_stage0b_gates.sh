#!/usr/bin/env bash
# Run Fine-tune gate B (one try). If Fgt is not clearly > 0, escalate to A (one try).
# Never starts the three-arm grid.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

bash "${ROOT}/scripts/run_gate_B.sh" "${ROOT}/recorded_smoke/stage0b_gate_B"
python3 "${ROOT}/scripts/write_gate_readme.py" "${ROOT}/recorded_smoke/stage0b_gate_B"

VERDICT=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("recorded_smoke/stage0b_gate_B/metrics_nbuf4_seed1.json")
payload = json.loads(p.read_text())
print(payload.get("forgetting_gate", {}).get("verdict", "FAIL"))
PY
)
echo "Gate B verdict=${VERDICT}"
if [ "${VERDICT}" = "PASS" ]; then
  echo "B PASS. Stop. Do not start the three-arm grid."
  exit 0
fi

echo "B FAIL (Fgt not clearly > 0). Escalating to gate A (knee→brain, domain-incremental)."
bash "${ROOT}/scripts/run_gate_A.sh" "${ROOT}/recorded_smoke/stage0b_gate_A"
python3 "${ROOT}/scripts/write_gate_readme.py" "${ROOT}/recorded_smoke/stage0b_gate_A"

VERDICT_A=$(python3 - <<'PY'
import json
from pathlib import Path
p = Path("recorded_smoke/stage0b_gate_A/metrics_nbuf4_seed1.json")
payload = json.loads(p.read_text())
print(payload.get("forgetting_gate", {}).get("verdict", "FAIL"))
PY
)
echo "Gate A verdict=${VERDICT_A}"
echo "Stop. Do not start the three-arm grid until the room says so."
