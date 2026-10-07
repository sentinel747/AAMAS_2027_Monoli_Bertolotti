#!/usr/bin/env bash
# Replica del seme 4 del livello "nomi veri" (campagna dei gradini di contesto,
# 10/09): quel seme e' l'unico dell'intera tesi in cui la coppia
# governatore+amministratori espande invece di contenersi (209 celle contro le
# 125 della baseline, 3995 morti contro 1482), con zero tornate perse. Gli
# altri quattro semi dello stesso livello stanno fra 82 e 113 celle.
#
# Due repliche dello stesso seme, stessa configurazione della campagna: il
# mondo e' identico, cambia soltanto il campionamento del modello. Se il regime
# di espansione ritorna, e' un effetto del livello di contesto e va raccontato;
# se non ritorna, il seme 4 e' rumore del modello e il livello va riportato per
# mediana come gli altri due.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/replica_nomi_veri_seme4.sh \
#         > runs/livelli/nomi_veri/replica_seme4.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
COMUNI="--seeds 4 --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --wait 120 --administrators --thinking off"

for rep in rep2 rep3; do
  echo "[replica-seme4] $(date -Is) REPLICA $rep -> runs/livelli/nomi_veri/llm_nomi_veri_amm_$rep"
  python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
    --context-levels nomi_veri --map-profile balanced $COMUNI \
    --out "runs/livelli/nomi_veri/llm_nomi_veri_amm_$rep" || echo "[replica-seme4] ERRORE $rep"
done
echo "[replica-seme4] $(date -Is) REPLICHE SEME 4 COMPLETE"
