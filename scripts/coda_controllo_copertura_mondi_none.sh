#!/usr/bin/env bash
# Baseline `none` di oggi per il controllo della copertura sugli altri mondi.
#
# La colonia senza governo della campagna mondi (7 settembre) NON si riproduce
# bit per bit col codice attuale (ice_rich seme 3 diverge dal passo ~198: le
# correzioni al motore dell'11-14 settembre non sono tutte dietro interruttore).
# Le run con --copertura-elettrica-vera vanno quindi confrontate con una
# baseline calcolata oggi, stessa configurazione. Una run per seme, riprendibile.
#
# Uso:  bash scripts/coda_controllo_copertura_mondi_none.sh
set -u
cd "$(dirname "$0")/.."
OUT=runs/controllo_copertura_mondi
MONDI=${MONDI:-"ice_rich fragmented high_hazard scarce_resources"}
PARALLELE=${PARALLELE:-2}
mkdir -p "$OUT"
: > "$OUT/coda_none.txt"
for seme in 3 4 5 6 7; do
  for mondo in $MONDI; do
    dir="$OUT/$mondo/ctrl_none_oggi_s$seme"
    [ -f "$dir/results.json" ] && continue
    rm -rf "$dir"
    echo "python scripts/run_governor_experiment.py --arms none --map-profile $mondo --seeds $seme --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --out $dir" >> "$OUT/coda_none.txt"
  done
done
echo "[none] $(date -Is) $(wc -l < "$OUT/coda_none.txt") baseline da fare" >> "$OUT/coda.log"
xargs -P "$PARALLELE" -I CMD sh -c 'CMD > /dev/null 2>&1; echo "[none] $(date -Is) fine: $(echo CMD | grep -o "\-\-out [^ ]*")"' < "$OUT/coda_none.txt" >> "$OUT/coda.log" 2>&1
echo "[none] $(date -Is) baseline esaurite" >> "$OUT/coda.log"
