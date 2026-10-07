#!/usr/bin/env bash
# Seconda coda: quando la campagna mondi e' chiusa, riesegue la coppia
# governo+amministratori su scarce_resources con il parser corretto (alias dei
# pilastri e regola nulla non fatale, 2026-09-06 sera), cosi' i due mondi si
# confrontano con lo stesso codice. La baseline non chiama il parser e resta.
set -u
cd "$(dirname "$0")/.."
LOG="runs/mondi/coda.log"  # solo letto: questo script scrive su coda_bis.log
COMUNI="--seeds 3 4 5 6 7 --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25"
echo "[coda-bis] $(date -Is) attendo la fine della campagna mondi"
until grep -q "CAMPAGNA MONDI COMPLETA" "$LOG" 2>/dev/null; do sleep 120; done
echo "[coda-bis] $(date -Is) MONDO scarce_resources: governo+amministratori, parser corretto"
python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
  --wait 120 --administrators --map-profile scarce_resources $COMUNI \
  --out "runs/mondi/scarce_resources/llm_completo_amm_v2" || echo "[coda-bis] ERRORE"
echo "[coda-bis] $(date -Is) CODA BIS COMPLETA"
