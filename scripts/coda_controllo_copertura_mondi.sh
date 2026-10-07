#!/usr/bin/env bash
# Controllo della copertura elettrica sugli altri mondi (2026-09-26).
#
# Ripete ESATTAMENTE il braccio governo+amministratori della campagna mondi
# (scripts/coda_mondi.sh: gpt-oss:20b sulla farm, 300 coloni, 1000 passi,
# cadenza 25, dotazione 0,6) con in piu' --copertura-elettrica-vera.
# Le baseline `ctrl_none` del 7 settembre NON si riproducono col codice di oggi:
# il confronto usa `ctrl_none_oggi_s<seme>` (scripts/coda_controllo_copertura_mondi_none.sh).
#
# Ordine per SEME e non per mondo: se la coda si ferma a meta', ogni mondo ha
# gia' qualche seme. Una run gia' finita (results.json presente) si salta:
# rilanciare lo script il giorno dopo riprende da dove si era fermato.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 bash scripts/coda_controllo_copertura_mondi.sh
#
# **Attesa 600 s, non 120 (27/09).** La campagna della tesi girava una run alla
# volta e il governatore rispondeva in 5,5 s (mediana): zero tornate perse su
# tutte le run. Con piu' run in parallelo la farm si satura, il governatore
# risponde in 45-50 s con code oltre i 120 s, e ogni risposta oltre l'attesa e'
# una tornata persa (resta la legge precedente per 25 passi): 1-3 su 40 con tre
# run, fino a 9 su 28 con cinque. `--wait` e' solo il tempo massimo di attesa,
# quindi alzarlo non cambia il protocollo: ristabilisce la condizione della
# tesi, che ogni tornata arrivi. Le run con tornate perse sono archiviate in
# `runs/controllo_copertura_mondi/_tornate_perse/`.
set -u
cd "$(dirname "$0")/.."
OUT=runs/controllo_copertura_mondi
MONDI=${MONDI:-"ice_rich fragmented high_hazard scarce_resources"}
PARALLELE=${PARALLELE:-3}
ATTESA=${ATTESA:-600}
mkdir -p "$OUT"
: > "$OUT/coda.txt"
for seme in 3 4 5 6 7; do
  for mondo in $MONDI; do
    dir="$OUT/$mondo/llm_amm_vera_s$seme"
    [ -f "$dir/results.json" ] && continue
    rm -rf "$dir"
    echo "python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b --wait $ATTESA --administrators --map-profile $mondo --seeds $seme --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --copertura-elettrica-vera --out $dir" >> "$OUT/coda.txt"
  done
done
echo "[coda] $(date -Is) $(wc -l < "$OUT/coda.txt") run da fare" >> "$OUT/coda.log"
xargs -P "$PARALLELE" -I CMD sh -c 'CMD > /dev/null 2>&1; echo "[coda] $(date -Is) fine: $(echo CMD | grep -o "\-\-out [^ ]*")"' < "$OUT/coda.txt" >> "$OUT/coda.log" 2>&1
echo "[coda] $(date -Is) coda esaurita" >> "$OUT/coda.log"
