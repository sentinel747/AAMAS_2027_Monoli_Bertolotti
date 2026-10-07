#!/usr/bin/env bash
# Due simulazioni insieme convengono, o la farm le mette in fila?
#
# **La domanda vale diciassette ore.** Un terzo del tempo di una run e' calcolo e
# due terzi sono attesa della farm, e la CPU sta al 20-30 per cento: sulla carta
# due run insieme si sovrappongono quasi perfettamente. Sulla carta. Se la farm
# accoda le richieste, la seconda run aspetta il modello della prima e il
# guadagno si mangia da solo.
#
# Si misura invece di discuterne: la STESSA run, con lo stesso seme e la stessa
# lunghezza, prima da sola e poi in coppia. Tre esecuzioni, un quarto d'ora.
#
#   T(coppia) ~ T(sola)        -> la farm regge due: si guadagna quasi il doppio
#   T(coppia) ~ 2 x T(sola)    -> la farm accoda: il parallelismo non serve
#   in mezzo                   -> si guadagna quanto dice il rapporto
#
# Scrive in C:/tmp e non tocca `runs/`: e' una misura, non un dato.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 bash scripts/prova_parallelo.sh
set -u
cd "$(dirname "$0")/.."

FUORI=${FUORI:-C:/tmp/prova_parallelo}
SEME=${SEME:-3}
PASSI=${PASSI:-250}
COMUNI="--steps $PASSI --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 \
--wait 120 --map-profile balanced --strato-ambientale --correzioni-motore"
MODELLO="--arms llm --provider gpu_farm --model gpt-oss:20b --variante-prompt D --administrators"

rm -rf "$FUORI"
mkdir -p "$FUORI"
dillo() { echo "[parallelo] $(date -Is) $*"; }

corri() {  # cartella
  python scripts/run_governor_experiment.py $MODELLO $COMUNI \
    --seeds "$SEME" --out "$1" > "$1.log" 2>&1
}

dillo "1) una sola run, per avere il metro"
inizio=$(date +%s)
corri "$FUORI/sola"
sola=$(( $(date +%s) - inizio ))
dillo "   una sola: ${sola}s"

dillo "2) due run identiche insieme"
inizio=$(date +%s)
corri "$FUORI/coppia_a" &
primo=$!
corri "$FUORI/coppia_b" &
secondo=$!
wait $primo $secondo
coppia=$(( $(date +%s) - inizio ))
dillo "   due insieme: ${coppia}s"

echo
echo "=== ESITO ==="
echo "una sola      : ${sola}s"
echo "due insieme   : ${coppia}s  (due di fila costerebbero $((sola * 2))s)"
if [ "$sola" -gt 0 ]; then
  guadagno=$(( (sola * 2 * 100) / (coppia > 0 ? coppia : 1) ))
  echo "resa del parallelismo: ${guadagno}% (100 = non serve, 200 = raddoppia)"
fi
echo
echo "Le run di prova stanno in $FUORI e non vanno usate come dati."
