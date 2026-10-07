#!/usr/bin/env bash
# Coda "accentramento / decentramento": aspetta la fine della coda modelli e
# livelli, poi sul mondo di riferimento (balanced, dotazione 0,6, semi 3-7,
# 300 coloni, 1000 passi, cadenza 25, gpt-oss:20b) riesegue con il parser
# corretto i due bracci che mancano al confronto pulito con la coppia v3:
#   - governatore SOLO (accentrato):      llm, senza amministratori;
#   - amministratori SOLI (solo locale):  none + admin-arm llm.
# Stessa configurazione della coppia di riferimento (runs/campagna_scarsa_v3):
# --wait 120, snapshot 25, nessun flag di thinking (il registro riporta '').
# I bracci vecchi (campagna_scarsa/llm_completo, amm_soli) restano come misura
# del difetto del parser; il confronto in tesi usa questi.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_governo.sh >> runs/governo/coda_governo.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
ATTESA=${ATTESA:-"runs/modelli/coda_modelli.log"}
SEMI="3 4 5 6 7"
COMUNI="--seeds $SEMI --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --wait 120 --map-profile balanced"
mkdir -p runs/governo

echo "[coda-governo] $(date -Is) attendo 'CAMPAGNA MODELLI E LIVELLI COMPLETA' in $ATTESA"
while ! grep -q "CAMPAGNA MODELLI E LIVELLI COMPLETA" "$ATTESA" 2>/dev/null; do
  sleep 300
done
echo "[coda-governo] $(date -Is) coda modelli chiusa, parte il confronto accentramento/decentramento"

echo "[coda-governo] $(date -Is) BRACCIO governatore solo -> runs/governo/llm_completo"
python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b $COMUNI \
  --out "runs/governo/llm_completo" || echo "[coda-governo] ERRORE governatore solo"

echo "[coda-governo] $(date -Is) BRACCIO amministratori soli -> runs/governo/amm_soli"
python scripts/run_governor_experiment.py --arms none --provider gpu_farm --model gpt-oss:20b \
  --administrators --admin-arm llm $COMUNI \
  --out "runs/governo/amm_soli" || echo "[coda-governo] ERRORE amministratori soli"

echo "[coda-governo] $(date -Is) CAMPAGNA GOVERNO COMPLETA"
