#!/usr/bin/env bash
# Coda "amministratori misti": aspetta la fine della coda accentramento /
# decentramento, poi sul mondo di riferimento (balanced, dotazione 0,6, semi
# 3-7, 300 coloni, 1000 passi, cadenza 25) esegue UNA sola configurazione, la
# piu' informativa che l'artefatto permetta:
#
#   governatore gpt-oss:20b  +  amministratori pescati fra i QUATTRO modelli
#   della campagna per famiglie, uno per distretto, riproducibile a parita' di
#   seme e distretto (src/governors/config.py, fabbrica()).
#
# Perche' vale la pena. Nella campagna per famiglie ogni modello amministrava
# una colonia diversa, quindi il confronto passava per cinque semi e per la
# varianza fra esecuzioni identiche, che sui vivi vale 241 coloni. Qui i
# quattro modelli amministrano LA STESSA colonia, nello stesso passo, sotto lo
# stesso governatore e sullo stesso stato del mondo: la quota di riscritture e
# i morti per distretto diventano confrontabili DENTRO la run, dove la varianza
# del campionamento non entra. Il registro per tornata
# (administrator_decisions.jsonl) porta provider, modello, token e latenza
# accanto a ogni decisione di distretto, quindi l'analisi puo' separarli.
#
# Costo: la tornata e' bloccante e ogni passo aspetta il distretto piu' lento,
# quindi il tempo e' governato da qwen3.6 (31,1 s di latenza mediana, 9,9 ore
# sui cinque semi da solo). Stima fra le sei e le dieci ore. Tutto sulla farm
# gratuita: nessun credito di API commerciali (CLAUDE.md).
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_misto.sh \
#         >> runs/misto/coda_misto.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
ATTESA=${ATTESA:-"runs/governo/coda_governo.log"}
SEMI="3 4 5 6 7"
COMUNI="--seeds $SEMI --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --wait 120 --map-profile balanced --thinking off"
MODELLI_AMM=${MODELLI_AMM:-"gpu_farm:gpt-oss:20b gpu_farm:gpt-oss:120b gpu_farm4:Qwen/Qwen3.8-27B-FP8 gpu_farm:qwen3.6:27b"}
mkdir -p runs/misto

echo "[coda-misto] $(date -Is) attendo 'CAMPAGNA GOVERNO COMPLETA' in $ATTESA"
while ! grep -q "CAMPAGNA GOVERNO COMPLETA" "$ATTESA" 2>/dev/null; do
  sleep 300
done
echo "[coda-misto] $(date -Is) coda governo chiusa, parte la campagna mista"

echo "[coda-misto] $(date -Is) BRACCIO amministratori misti -> runs/misto/amm_misto"
python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
  --administrators --admin-models $MODELLI_AMM $COMUNI \
  --out "runs/misto/amm_misto" || echo "[coda-misto] ERRORE amministratori misti"

echo "[coda-misto] $(date -Is) CAMPAGNA MISTA COMPLETA"
