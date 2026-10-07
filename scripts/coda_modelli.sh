#!/usr/bin/env bash
# Coda "modelli e livelli" (punto 5 della rotta del 07/09): aspetta la riga
# "CAMPAGNA MONDI COMPLETA" nel log della coda mondi 2, poi sul mondo di
# riferimento (balanced, dotazione 0,6, semi 3-7, 300 coloni, 1000 passi,
# cadenza 25) esegue la coppia governatore+amministratori con gli altri tre
# modelli della campagna per famiglie, e infine i gradini di contesto del
# governatore con gpt-oss:20b. La baseline appaiata e' gia' eseguita:
# runs/campagna_scarsa/ctrl_none (stessi semi, stessa configurazione).
#
# I quattro modelli e le latenze misurate (docs/MARS_ABM_GOVERNATORI_LLM_ESECUZIONE.md):
#   gpt-oss:20b   gpu_farm   8,7 s   (gia' eseguito: campagna_scarsa_v3, tre repliche)
#   gpt-oss:120b  gpu_farm  20,7 s   thinking off -> low (non sa spegnersi)
#   Qwen3.8-27B   gpu_farm4  8,6 s   thinking off
#   qwen3.6:27b   gpu_farm  31,1 s   thinking off -> reasoning_effort none (dal 07/09)
# `--wait 120` copre anche il piu' lento. Tutti gratuiti sulla farm: nessun
# credito API esterno (CLAUDE.md).
#
# I gradini di contesto agiscono sul solo governatore (gli amministratori
# leggono sempre il prompt completo): il braccio misura quanto del dominio il
# governo vede, non quanto ne vede il distretto.
#
# Una run alla volta: le tornate degli amministratori saturano gia' la farm e
# due simulazioni sulla stessa CPU si rallentano a vicenda. Rilanciabile: il
# runner salta i semi gia' in results.jsonl.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_modelli.sh > runs/modelli/coda_modelli.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
ATTESA=${ATTESA:-"runs/mondi/coda_mondi2.log"}
SEMI="3 4 5 6 7"
COMUNI="--seeds $SEMI --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --wait 120 --administrators --thinking off"
mkdir -p runs/modelli runs/livelli

echo "[coda-modelli] $(date -Is) attendo 'CAMPAGNA MONDI COMPLETA' in $ATTESA"
while ! grep -q "CAMPAGNA MONDI COMPLETA" "$ATTESA" 2>/dev/null; do
  sleep 300
done
echo "[coda-modelli] $(date -Is) coda mondi chiusa, partono i modelli"

# provider|modello|cartella
MODELLI=${MODELLI:-"gpu_farm|gpt-oss:120b|gptoss120b gpu_farm4|Qwen/Qwen3.8-27B-FP8|qwen38_27b gpu_farm|qwen3.6:27b|qwen36_27b"}
for voce in $MODELLI; do
  IFS='|' read -r provider modello cartella <<< "$voce"
  echo "[coda-modelli] $(date -Is) MODELLO $modello ($provider) -> runs/modelli/$cartella"
  python scripts/run_governor_experiment.py --arms llm --provider "$provider" --model "$modello" \
    --map-profile balanced $COMUNI \
    --out "runs/modelli/$cartella/llm_completo_amm" || echo "[coda-modelli] ERRORE modello $modello"
done
echo "[coda-modelli] $(date -Is) MODELLI COMPLETI"

LIVELLI=${LIVELLI:-"cieco senza_aiuti nomi_veri"}
for livello in $LIVELLI; do
  echo "[coda-modelli] $(date -Is) LIVELLO $livello (gpt-oss:20b) -> runs/livelli/$livello"
  python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
    --context-levels "$livello" --map-profile balanced $COMUNI \
    --out "runs/livelli/$livello/llm_${livello}_amm" || echo "[coda-modelli] ERRORE livello $livello"
done
echo "[coda-modelli] $(date -Is) CAMPAGNA MODELLI E LIVELLI COMPLETA"
