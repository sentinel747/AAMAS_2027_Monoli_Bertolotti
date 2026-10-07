#!/usr/bin/env bash
# Coda della campagna "mondi": aspetta la chiusura della replica 3 (cinque
# righe `vivi=` nel suo console.log), poi per ogni profilo di mappa esegue la
# baseline (none) e la coppia governo+amministratori (llm+amm), semi 3-7.
#
# Stesso disegno del main scenario (300 coloni, 1000 passi, cadenza 25,
# dotazione 0,6, gpt-oss:20b sulla farm gratuita): cambia SOLO il mondo.
# Una run alla volta: le tornate degli amministratori saturano gia' la farm.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_mondi.sh > runs/mondi/coda.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
REP3="runs/campagna_scarsa_v3/llm_completo_amm_rep3/console.log"
PROFILI=${PROFILI:-"scarce_resources high_hazard"}
SEMI="3 4 5 6 7"
COMUNI="--seeds $SEMI --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25"

echo "[coda] $(date -Is) attendo la replica 3 ($REP3)"
while [ "$(grep -c 'vivi=' "$REP3" 2>/dev/null || echo 0)" -lt 5 ]; do
  sleep 120
done
echo "[coda] $(date -Is) replica 3 chiusa, parte la campagna mondi: $PROFILI"

for profilo in $PROFILI; do
  echo "[coda] $(date -Is) MONDO $profilo: baseline"
  python scripts/run_governor_experiment.py --arms none --map-profile "$profilo" $COMUNI \
    --out "runs/mondi/$profilo/ctrl_none" || echo "[coda] ERRORE baseline $profilo"
  echo "[coda] $(date -Is) MONDO $profilo: governo+amministratori"
  python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
    --wait 120 --administrators --map-profile "$profilo" $COMUNI \
    --out "runs/mondi/$profilo/llm_completo_amm" || echo "[coda] ERRORE llm+amm $profilo"
done
echo "[coda] $(date -Is) CAMPAGNA MONDI COMPLETA"
