#!/usr/bin/env bash
# Il confondente del controllo sui mondi (2026-09-27): l'indicatore COM'ERA,
# sul codice di oggi.
#
# Fra la campagna mondi della tesi (7 settembre) e il controllo con
# --copertura-elettrica-vera e' cambiato anche il motore (correzioni 11-14/09).
# Questo braccio e' identico al controllo SENZA l'interruttore: stesso codice,
# stessa baseline di oggi, cambia solo l'indicatore. Separa i due effetti:
#   tesi -> questo braccio      = effetto delle correzioni al motore
#   questo braccio -> corretto  = effetto dell'indicatore
#
# Ordine per SEME e non per mondo: se la coda si ferma a meta', ogni mondo ha
# gia' qualche seme. Una run gia' finita (results.json presente) si salta:
# rilanciare lo script il giorno dopo riprende da dove si era fermato.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 bash scripts/coda_controllo_copertura_mondi_com_era.sh
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
: > "$OUT/coda_com_era.txt"
for seme in 3 4 5 6 7; do
  for mondo in $MONDI; do
    dir="$OUT/$mondo/llm_amm_com_era_s$seme"
    [ -f "$dir/results.json" ] && continue
    rm -rf "$dir"
    echo "python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b --wait $ATTESA --administrators --map-profile $mondo --seeds $seme --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --out $dir" >> "$OUT/coda_com_era.txt"
  done
done
echo "[com_era] $(date -Is) $(wc -l < "$OUT/coda_com_era.txt") run da fare" >> "$OUT/coda.log"
xargs -P "$PARALLELE" -I CMD sh -c 'CMD > /dev/null 2>&1; echo "[com_era] $(date -Is) fine: $(echo CMD | grep -o "\-\-out [^ ]*")"' < "$OUT/coda_com_era.txt" >> "$OUT/coda.log" 2>&1
echo "[com_era] $(date -Is) coda esaurita" >> "$OUT/coda.log"
