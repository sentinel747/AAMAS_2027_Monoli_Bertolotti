#!/usr/bin/env bash
# Il pilota della base pulita: due domande, quattro ore, prima di impegnarne cinquanta.
#
# **Perche' esiste.** La campagna pulita e' 55 run per circa cinquanta ore. Una
# configurazione sbagliata scoperta alla run quaranta costa due giorni, e in
# questo progetto e' gia' successo. Il pilota risponde a due domande, in
# quest'ordine, perche' la prima e' quella che puo' fermare tutto:
#
#   1. **Ogni braccio regge?** Gli undici bracci della campagna, un seme solo e
#      un quarto della lunghezza. Non producono risultati: producono la prova
#      che scrivono artifact validi, che il modello risponde in tutte le
#      varianti di prompt, e che il cancello di qualita' li promuove.
#
#   2. **Quanto vale il pavimento dei cantieri?** E' l'unico parametro della
#      campagna su cui nessuno ha una risposta giusta da dare. 1,1 e' il valore
#      con cui e' stato misurato tutto l'archivio ed e' sopra il massimo di ogni
#      altra priorita' di lavoro: con cinque cantieri aperti il menu di cella
#      contiene solo costruzioni. 0,95 sta appena sotto quel massimo, quindi
#      tiene il cantiere in cima fra le costruzioni senza scacciare il
#      sostentamento. 0 lo spegne del tutto. Tre valori, tre semi, baseline
#      senza modello: la differenza si legge senza rumore di campionamento.
#
# La baseline e' deterministica fra processi, quindi le nove run del secondo
# blocco non interrogano nessun modello e costano solo tempo di CPU.
#
# Uso:
#   MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/pilota.sh >> runs/pilota.log 2>&1 &
#   python scripts/avanzamento.py --piano runs/piano_pilota.txt --log runs/pilota.log
set -u
cd "$(dirname "$0")/.."

PIANO=${PIANO:-runs/piano_pilota.txt}
FUORI=${FUORI:-runs/pilota}
TAB=$'\t'

# Il regime della base pulita: strato ambientale acceso (il mondo non e' piu'
# congelato), redistribuzione spenta (accesa la colonia muore: misurato, 0 vivi
# su un seme), capienze corrette (l'ossigeno fantasma degli habitat entrava nel
# quadro che il governatore legge).
REGIME="--strato-ambientale --correzioni-motore"
COMUNI="--agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 \
--snapshot-interval 25 --wait 120 --map-profile balanced"
MODELLO="--arms llm --provider gpu_farm --model gpt-oss:20b"

mkdir -p runs
dillo() { echo "[pilota] $(date -Is) $*"; }
. "$(dirname "$0")/lucchetto_macchina.sh"

BLOCCATI=${BLOCCATI:-$(grep -v '^#' runs/.pid_bloccati 2>/dev/null | grep -E '^[0-9]+$' | paste -sd, -)}
occupata() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "\$relitti = @($BLOCCATI); if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_governor_experiment*' -and \$relitti -notcontains \$_.ProcessId }) { 'si' } else { 'no' }" \
    2>/dev/null | tr -d '\r' | grep -q '^si$'
}

riga() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$PIANO"; }

: > "$PIANO"

# ---------------------------------------------------------------------------
# 1. Tenuta degli undici bracci: un seme, 250 passi.
# ---------------------------------------------------------------------------
SEME_PROVA=3
CORTO="--steps 250 $COMUNI"

riga "prova: baseline" "$FUORI/ctrl_none" "$SEME_PROVA" \
  "--arms none $REGIME $CORTO"

for v in 0 A B F C D; do
  riga "prova: prompt $v" "$FUORI/variante_$v" "$SEME_PROVA" \
    "$MODELLO --variante-prompt $v --administrators $REGIME $CORTO"
done

riga "prova: cieco due livelli" "$FUORI/cieco_due_livelli" "$SEME_PROVA" \
  "$MODELLO --variante-prompt D --administrators --context-levels cieco \
--livello-amministratori cieco $REGIME $CORTO"

riga "prova: solo governatore" "$FUORI/solo_governatore" "$SEME_PROVA" \
  "$MODELLO --variante-prompt D $REGIME $CORTO"

riga "prova: solo amministratori" "$FUORI/solo_amministratori" "$SEME_PROVA" \
  "--arms none --administrators --variante-prompt D --provider gpu_farm \
--model gpt-oss:20b $REGIME $CORTO"

# ---------------------------------------------------------------------------
# 2. Il pavimento dei cantieri: tre valori, tre semi, senza modello.
# ---------------------------------------------------------------------------
for p in 1.1 0.95 0.0; do
  etichetta=$(echo "$p" | tr '.' '_')
  for s in 3 4 5; do
    riga "pavimento $p" "$FUORI/pavimento_$etichetta" "$s" \
      "--arms none --pavimento-cantieri $p $REGIME --steps 1000 $COMUNI"
  done
done

dillo "piano del pilota in $PIANO: $(wc -l < "$PIANO") run"

dillo "aspetto che la macchina sia libera"
while occupata; do sleep 60; done
prendi_lucchetto "pilota"
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

  # **Nel pilota il cancello non ferma, segnala.** Fermarsi al primo intoppo
  # avrebbe senso in campagna, dove una run avvelena le successive; qui serve
  # l'elenco COMPLETO di cio' che non va, per correggerlo in una volta sola.
  python scripts/qualita_run.py "$fuori" --seme "$seme" \
    || dillo "DA GUARDARE: $nome seme $seme non passa il cancello (vedi sopra)."
done < "$PIANO"

dillo "PILOTA COMPLETO"
