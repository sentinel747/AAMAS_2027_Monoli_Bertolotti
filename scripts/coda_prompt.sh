#!/usr/bin/env bash
# La campagna incrociata sui prompt: la parola, la gerarchia, la lingua.
#
# **La domanda.** Ci sono tre obiezioni possibili al prompt con cui tutte
# le campagne dell'archivio sono state misurate: «pilastro» non e' la parola
# giusta per cio' che la politica pesa e il testo per giunta ne usa due (dice
# anche «leva»); il governatore non sa che sotto di lui ci sono amministratori
# che possono riscrivere la sua legge, mentre loro sanno di lui; e tutto il
# prompt e' in italiano mentre i modelli sono addestrati in prevalenza su
# inglese.
#
# **Il disegno e' incrociato e non a catena.** Sei caselle, ognuna diversa dalla
# precedente per UNA cosa sola, cosi' che l'effetto della parola si possa
# leggere dentro l'italiano E dentro l'inglese:
#
#     0   it  pilastro   senza gerarchia   <- gia' in archivio (campagna_scarsa_v3)
#     A   it  categoria  senza gerarchia   <- 0 -> A isola la PAROLA in italiano
#     B   it  categoria  CON gerarchia     <- A -> B isola la GERARCHIA in italiano
#     F   en  pilastro   senza gerarchia   <- 0 -> F isola la LINGUA
#     C   en  categoria  senza gerarchia   <- F -> C isola la PAROLA in inglese
#     D   en  categoria  CON gerarchia     <- C -> D isola la GERARCHIA in inglese
#
# **I parametri sono copiati, non scelti.** Sono quelli di
# `runs/campagna_scarsa_v3/llm_completo_amm`, cioe' del braccio 0 con cui queste
# caselle vanno confrontate, `temperature` e `thinking` compresi: quel braccio
# registra `temperature: null` e `thinking: ""`, quindi qui non si passa ne'
# `--temperature` ne' `--thinking`. `thinking: ""` e `thinking: "off"`
# producono la stessa richiesta (`normalizza(None) -> "off"`), ma il registro
# deve dire la stessa cosa, altrimenti il controllo di parita' degli ingressi
# (`scripts/audit_registri.py`) segnala una differenza che non c'e'.
#
# **Niente coprifuoco.** La macchina lavora anche di notte. Una run interrotta a
# meta' lascia una cartella di seme parziale, ma la ripresa e' sicura: il runner
# salta i semi gia' presenti in `results.jsonl`.
#
# Uso:
#   MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_prompt.sh \
#     >> runs/prompt/coda_prompt.log 2>&1 &
#
# Per vedere a che punto e':
#   python scripts/avanzamento.py
#
# Variabili: VARIANTI="A B" (quali caselle), SEMI="3 4 5 6 7",
# PIANO (un file di piano tutto suo, se un'altra coda sta girando),
# DOPO_PIANO (aspetta che quel piano sia finito prima di cominciare).
set -u
cd "$(dirname "$0")/.."

VARIANTI=${VARIANTI:-"A B"}
SEMI=${SEMI:-"3 4 5 6 7"}
FUORI=${FUORI:-runs/prompt}

# Copiati dal braccio 0. Ogni scostamento renderebbe il confronto un confronto
# fra due cose invece che fra una.
COMUNI="--steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 \
--snapshot-interval 25 --wait 120 --map-profile balanced --administrators"
MODELLO="--arms llm --provider gpu_farm --model gpt-oss:20b"

mkdir -p "$FUORI"
dillo() { echo "[prompt] $(date -Is) $*"; }

# Il lucchetto: due code in attesa della stessa macchina possono vedersi
# libere nello stesso istante e partire insieme.
. "$(dirname "$0")/lucchetto_macchina.sh"

# `pgrep` non vede i processi Windows, e i processi che hanno finito ma che
# Windows non riesce a terminare vanno esclusi a mano (vedi `_esci` in
# scripts/run_governor_experiment.py).
BLOCCATI=${BLOCCATI:-$(grep -v '^#' runs/.pid_bloccati 2>/dev/null | grep -E '^[0-9]+$' | paste -sd, -)}
occupata() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "\$relitti = @($BLOCCATI); if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_governor_experiment*' -and \$relitti -notcontains \$_.ProcessId }) { 'si' } else { 'no' }" \
    2>/dev/null | tr -d '\r' | grep -q '^si$'
}

# **Il piano si scrive prima di eseguirlo**, e non solo per ordine: e' il file da
# cui `scripts/avanzamento.py` sa quante run mancano. Senza, la barra puo' dire
# la percentuale di cio' che ha gia' visto, non quella del lavoro.
PIANO=${PIANO:-$FUORI/piano.txt}
: > "$PIANO"
for v in $VARIANTI; do
  for s in $SEMI; do
    echo "$v $s $FUORI/variante_$v" >> "$PIANO"
  done
done
dillo "piano scritto in $PIANO: $(wc -l < "$PIANO") run"

[ -n "${DOPO_PIANO:-}" ] && aspetta_piano "$DOPO_PIANO" "prompt"
dillo "aspetto che la macchina sia libera"
while occupata; do sleep 120; done
prendi_lucchetto "prompt"
trap lascia_lucchetto EXIT INT TERM
dillo "macchina libera, si parte. Nessun coprifuoco: si lavora anche di notte."

while read -r variante seme fuori; do
  [ -z "${variante:-}" ] && continue
  if grep -q "\"seed\": $seme," "$fuori/results.jsonl" 2>/dev/null; then
    dillo "variante $variante seme $seme gia' fatta, salto."
    continue
  fi
  dillo "variante $variante seme $seme"
  python scripts/run_governor_experiment.py $MODELLO --variante-prompt "$variante" \
    --seeds "$seme" $COMUNI --out "$fuori" \
    || dillo "ERRORE in variante $variante seme $seme"
done < "$PIANO"

dillo "CAMPAGNA PROMPT COMPLETA"
