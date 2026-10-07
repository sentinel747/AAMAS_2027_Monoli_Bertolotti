#!/usr/bin/env bash
# La coda unica: tutto il lavoro che resta, in un piano solo.
#
# **Perche' una sola.** Fino a ieri c'erano tre code in attesa della stessa
# macchina, ognuna con il suo piano e il suo log, e per sapere a che punto era il
# lavoro bisognava guardare in tre posti e sommare a mente. Peggio: due code in
# attesa possono vedere la macchina libera nello stesso istante e partire
# insieme. Qui il piano e' uno, l'ordine e' esplicito, e
# `python scripts/avanzamento.py` mostra tutto.
#
# **Si ferma alla prima run non comparabile, e questo e' il punto.** Una run che
# perde tornate del governatore o che ha chiamate fallite finisce lo stesso,
# scrive i suoi artifact e ha l'aria di tutte le altre: una campagna di venti ore
# puo' riempirsi di righe inutilizzabili senza che nessuno se ne accorga fino
# all'analisi. Dopo ogni run `scripts/qualita_run.py` legge il registro e, se
# trova qualcosa, la coda si ferma e lo dice. Fermarsi costa qualche ora di
# macchina; non fermarsi costa la campagna.
#
# **Il piano e' un file TSV**, una riga per run:
#     nome<TAB>cartella<TAB>seme<TAB>argomenti
# Il nome serve a leggerlo, la cartella e il seme a sapere se e' gia' fatta, gli
# argomenti a eseguirla. Aggiungere lavoro vuol dire aggiungere righe qui sotto.
#
# **Niente coprifuoco.** La macchina lavora anche di notte. Riprendere e' sicuro:
# il runner salta i semi gia' presenti in `results.jsonl`.
#
# Uso:
#   MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda.sh >> runs/coda.log 2>&1 &
#   python scripts/avanzamento.py
#
# Variabili: PIANO, SEMI, DOPO_PIANO (aspetta che quel piano sia finito prima).
set -u
cd "$(dirname "$0")/.."

SEMI=${SEMI:-"3 4 5 6 7"}
PIANO=${PIANO:-runs/piano.txt}
TAB=$'\t'

# Copiati dal braccio 0 (`runs/campagna_scarsa_v3/llm_completo_amm`), che e' il
# braccio con cui tutte queste caselle vanno confrontate. Ogni scostamento
# renderebbe il confronto un confronto fra due cose invece che fra una.
COMUNI="--steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 \
--snapshot-interval 25 --wait 120 --map-profile balanced"
MODELLO="--arms llm --provider gpu_farm --model gpt-oss:20b"

mkdir -p runs
dillo() { echo "[coda] $(date -Is) $*"; }

. "$(dirname "$0")/lucchetto_macchina.sh"

BLOCCATI=${BLOCCATI:-$(grep -v '^#' runs/.pid_bloccati 2>/dev/null | grep -E '^[0-9]+$' | paste -sd, -)}
occupata() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "\$relitti = @($BLOCCATI); if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_governor_experiment*' -and \$relitti -notcontains \$_.ProcessId }) { 'si' } else { 'no' }" \
    2>/dev/null | tr -d '\r' | grep -q '^si$'
}

riga() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$PIANO"; }

: > "$PIANO"

# 1. Le tre obiezioni al prompt, in italiano. Gia' in corso.
for v in A B; do
  for s in $SEMI; do
    riga "prompt $v (it)" "runs/prompt/variante_$v" "$s" \
      "$MODELLO --variante-prompt $v --administrators $COMUNI"
  done
done

# 2. Le stesse, in inglese. F traduce il braccio 0, C cambia anche la parola,
#    D aggiunge la gerarchia: ogni casella dista dalla precedente per una cosa.
for v in F C D; do
  for s in $SEMI; do
    riga "prompt $v (en)" "runs/prompt/variante_$v" "$s" \
      "$MODELLO --variante-prompt $v --administrators $COMUNI"
  done
done

# 3. Il gradino cieco applicato a TUTTI E DUE i livelli.
#    La tesi afferma che la prudenza non scompare quando scompare il dominio, ma
#    finora il gradino cieco toglieva il dominio al solo governatore: gli
#    amministratori continuavano a sapere che si tratta di Marte, e sono loro a
#    riscrivere la legge nella meta' delle tornate. L'affermazione regge solo se
#    regge anche qui.
for s in $SEMI; do
  riga "cieco a due livelli" "runs/livelli/cieco_due_livelli" "$s" \
    "$MODELLO --context-levels cieco --administrators --livello-amministratori cieco $COMUNI"
done

# 4. Le baseline del mondo di riferimento con le istantanee.
#    Non interrogano alcun modello e la simulazione e' deterministica fra
#    processi: la riesecuzione da' gli stessi numeri e aggiunge solo le mappe,
#    che servono all'indice di dispersione. `confronta_baseline.py` lo verifica.
for s in $SEMI; do
  riga "baseline riferimento (mappe)" "runs/riferimento_snapshot/ctrl_none" "$s" \
    "--arms none $COMUNI"
done

dillo "piano scritto in $PIANO: $(wc -l < "$PIANO") run"

[ -n "${DOPO_PIANO:-}" ] && aspetta_piano "$DOPO_PIANO" "coda"
dillo "aspetto che la macchina sia libera"
while occupata; do sleep 120; done
prendi_lucchetto "coda"
trap lascia_lucchetto EXIT INT TERM
dillo "si parte. Nessun coprifuoco."

while IFS="$TAB" read -r nome fuori seme argomenti; do
  [ -z "${nome:-}" ] && continue
  if grep -q "\"seed\": $seme," "$fuori/results.jsonl" 2>/dev/null; then
    dillo "$nome seme $seme gia' fatta, salto."
    continue
  fi
  dillo "$nome seme $seme"
  python scripts/run_governor_experiment.py $argomenti --seeds "$seme" --out "$fuori" \
    || dillo "ERRORE in $nome seme $seme"

  # **Il controllo sta qui e non alla fine.** Una run non comparabile fra le
  # prime avvelena tutto cio' che viene dopo, e l'unico momento in cui fermarsi
  # costa poco e' subito.
  if ! python scripts/qualita_run.py "$fuori" --seme "$seme"; then
    dillo "FERMO LA CODA: $nome seme $seme non e' comparabile (vedi sopra)."
    dillo "Il resto del piano resta in $PIANO e riparte con lo stesso comando"
    dillo "una volta capito il perche'. La run va rifatta, non tenuta."
    exit 1
  fi
done < "$PIANO"

dillo "PIANO COMPLETO"

# Le baseline rieseguite devono dare gli stessi numeri di quelle in archivio.
if [ -f runs/riferimento_snapshot/ctrl_none/results.jsonl ]; then
  dillo "verifica che le baseline rieseguite coincidano con l'archivio"
  python scripts/confronta_baseline.py --nuova runs/riferimento_snapshot/ctrl_none \
    --archivio runs/campagna_scarsa/ctrl_none \
    || dillo "ATTENZIONE: la riesecuzione NON coincide con l'archivio, vedi sopra"
fi

dillo "FINE"
