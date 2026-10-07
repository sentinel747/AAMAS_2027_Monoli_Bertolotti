#!/usr/bin/env bash
# Le istantanee mancanti della baseline del mondo di riferimento.
#
# **Perche' serve.** L'indice di dispersione
# (`scripts/indice_dispersione.py`) si calcola sull'ultima istantanea di ogni
# run. La baseline del riferimento, `runs/campagna_scarsa/ctrl_none`, e'
# anteriore agli snapshot a ogni tornata e non ne ha nessuna: senza le due mappe
# il confronto appaiato su quel mondo non esiste, e il mondo di riferimento --
# quello su cui si fonda tutta l'analisi -- resta fuori da una tabella dove
# devono esserci tutti e cinque.
#
# **Rieseguire e' legittimo qui e non lo sarebbe altrove.** La baseline non
# interroga alcun modello e la simulazione e' deterministica fra processi
# distinti (`tests/test_state_digest.py`): la riesecuzione produce gli STESSI
# numeri, e aggiunge solo gli artifact che allora non venivano scritti. Non e'
# una nuova misura, e' la stessa misura con piu' tracce. Lo script lo verifica:
# se i vivi o le celle non coincidono con quelli in archivio, lo dice invece di
# lasciar passare due baseline diverse con lo stesso nome.
#
# I parametri sono copiati da `runs/campagna_scarsa/ctrl_none`, `temperature` e
# `thinking` compresi (entrambi vuoti), piu' il solo `--snapshot-interval 25`
# che e' la ragione della riesecuzione.
#
# Aspetta che la macchina sia libera: la campagna sui prompt viene prima.
#
# Uso:
#   nohup bash scripts/coda_baseline_riferimento.sh \
#     >> runs/prompt/coda_baseline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

SEMI=${SEMI:-"3 4 5 6 7"}
FUORI=${FUORI:-runs/riferimento_snapshot/ctrl_none}
ARCHIVIO=${ARCHIVIO:-runs/campagna_scarsa/ctrl_none}

mkdir -p "$(dirname "$FUORI")"
dillo() { echo "[baseline] $(date -Is) $*"; }

. "$(dirname "$0")/lucchetto_macchina.sh"

BLOCCATI=${BLOCCATI:-$(grep -v '^#' runs/.pid_bloccati 2>/dev/null | grep -E '^[0-9]+$' | paste -sd, -)}
occupata() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "\$relitti = @($BLOCCATI); if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_governor_experiment*' -and \$relitti -notcontains \$_.ProcessId }) { 'si' } else { 'no' }" \
    2>/dev/null | tr -d '\r' | grep -q '^si$'
}

dillo "in coda dietro alle campagne sui prompt"
for piano in ${PIANI_PRIMA:-runs/prompt/piano.txt runs/prompt/piano_inglese.txt}; do
  aspetta_piano "$piano" "baseline"
done
while occupata; do sleep 120; done
prendi_lucchetto "baseline"
trap lascia_lucchetto EXIT INT TERM
dillo "macchina libera, si parte. Cinque baseline deterministiche, ~13 min l'una."

for seme in $SEMI; do
  if grep -q "\"seed\": $seme," "$FUORI/results.jsonl" 2>/dev/null; then
    dillo "seme $seme gia' fatto, salto."
    continue
  fi
  dillo "seme $seme"
  python scripts/run_governor_experiment.py --arms none --seeds "$seme" \
    --steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 \
    --snapshot-interval 25 --wait 120 --map-profile balanced \
    --out "$FUORI" || dillo "ERRORE nel seme $seme"
done

dillo "verifica di identita' con l'archivio"
python scripts/confronta_baseline.py --nuova "$FUORI" --archivio "$ARCHIVIO" \
  || dillo "ATTENZIONE: la riesecuzione NON coincide con l'archivio, vedi sopra"

dillo "FINE"
