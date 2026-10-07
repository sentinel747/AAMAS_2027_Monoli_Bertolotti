#!/usr/bin/env bash
# Ripetizioni sul lato ACCESO dei due interruttori sperimentali.
#
# **La domanda.** Con lo strato ambientale acceso e con la redistribuzione
# accesa, l'effetto appaiato della coppia sulla popolazione e' negativo in tutti
# e quattro i casi misurati (-416 e -43 il primo, -168 e -106 la seconda),
# mentre a interruttori spenti e' positivo in cinque esecuzioni su sei. Due
# interruttori indipendenti che danno lo stesso ribaltamento sono o un
# risultato — il vantaggio apparente del governo linguistico sulla popolazione
# dipende dal regime del mondo — o un caso, perche' sul lato acceso esiste UNA
# sola esecuzione per seme e fra esecuzioni identiche lo scarto vale 327 e 338
# coloni, cioe' piu' dei valori misurati.
#
# **Il disegno.** Due esecuzioni in piu' per ogni coppia (interruttore, seme),
# cosi' da averne tre per cella esattamente come sul lato spento, dove le tre
# esecuzioni del riferimento sono gia' in archivio. Le baseline NON si ripetono:
# non interrogano alcun modello e la simulazione e' deterministica fra processi,
# quindi una seconda esecuzione darebbe lo stesso numero.
#
# **L'ordine e' per costo, non per interesse.** La redistribuzione costa 19-30
# minuti a run e lo strato ambientale 63-71: le quattro run della
# redistribuzione danno una risposta completa su un interruttore in meno di due
# ore, e se quella risposta e' netta cambia il valore atteso delle altre.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_interruttori.sh \
#         >> runs/prove/coda_interruttori.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
COPRIFUOCO=${COPRIFUOCO:-23:45}
SEMI=${SEMI:-"3 4"}
COMUNI="--steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --wait 120 --map-profile balanced --administrators --thinking off"
COPPIA="--arms llm --provider gpu_farm --model gpt-oss:20b"
mkdir -p runs/prove

dillo() { echo "[interruttori] $(date -Is) $*"; }

# Come nelle altre code: `pgrep` non vede i processi Windows; il filtro sul nome
# dell'eseguibile impedisce alla ricerca di trovare se stessa; e i processi che
# hanno finito ma che Windows non riesce a terminare vanno esclusi a mano (vedi
# `_esci` in scripts/run_governor_experiment.py).
BLOCCATI=${BLOCCATI:-$(grep -v '^#' runs/.pid_bloccati 2>/dev/null | grep -E '^[0-9]+$' | paste -sd, -)}

occupata() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "\$relitti = @($BLOCCATI); if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_governor_experiment*' -and \$relitti -notcontains \$_.ProcessId }) { 'si' } else { 'no' }" \
    2>/dev/null | tr -d '\r' | grep -q '^si$'
}

minuti_al_coprifuoco() {
  local fine ora
  fine=$(date -d "today $COPRIFUOCO" +%s)
  ora=$(date +%s)
  [ "$fine" -le "$ora" ] && fine=$(date -d "tomorrow $COPRIFUOCO" +%s)
  echo $(( (fine - ora) / 60 ))
}

# $1 nome leggibile, $2 minuti stimati, $3 cartella, $4 seme, $5.. il comando.
#
# Il salto va deciso sul singolo seme e non sul numero di righe del registro:
# una coda ripresa dopo il coprifuoco trova una cartella con una riga sola e,
# contando le righe, rifarebbe il seme gia' fatto invece del seme mancante.
prova() {
  local nome=$1 stima=$2 fuori=$3 seme=$4; shift 4
  if grep -q "\"seed\": $seme," "$fuori/results.jsonl" 2>/dev/null; then
    dillo "$nome gia' fatta, salto."
    return 0
  fi
  if [ "$(minuti_al_coprifuoco)" -lt "$stima" ]; then
    dillo "COPRIFUOCO: $nome richiede ~$stima min e ne restano $(minuti_al_coprifuoco)."
    dillo "Riprendi con lo stesso comando: niente parte a meta'."
    exit 0
  fi
  dillo "$nome (stima $stima min, restano $(minuti_al_coprifuoco))"
  "$@" --out "$fuori" || dillo "ERRORE in $nome"
}

dillo "aspetto che la macchina sia libera"
while occupata; do sleep 120; done
dillo "macchina libera; coprifuoco alle $COPRIFUOCO (restano $(minuti_al_coprifuoco) min)"

# Prima la redistribuzione: costa un terzo e chiude un interruttore per intero.
for rep in rep2 rep3; do
  for seme in $SEMI; do
    prova "redistribuzione $rep seme $seme" 35 "runs/prove/redistr_llm_amm_$rep" "$seme" \
      python scripts/run_governor_experiment.py $COPPIA --seeds "$seme" $COMUNI --redistribuzione
  done
done
for rep in rep2 rep3; do
  for seme in $SEMI; do
    prova "strato ambientale $rep seme $seme" 75 "runs/prove/ambiente_llm_amm_$rep" "$seme" \
      python scripts/run_governor_experiment.py $COPPIA --seeds "$seme" $COMUNI --strato-ambientale
  done
done
dillo "INTERRUTTORI COMPLETI"

# Finito questo, si torna alle repliche dei mondi da dove si erano fermate.
dillo "torno alla terza esecuzione dei mondi"
COPRIFUOCO="$COPRIFUOCO" bash scripts/coda_weekend.sh
dillo "FINE"
