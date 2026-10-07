#!/usr/bin/env bash
# Aspetta che il fornitore torni libero, poi riprende la campagna da sola.
#
# **Perche' esiste.** Il 16 settembre la farm Ollama si e' saturata a meta'
# campagna e ci e' rimasta per piu' di ventiquattro ore. Rilanciare a mano ogni
# tanto significa o controllare di continuo, o perdere ore fra il momento in cui
# il fornitore si libera e il momento in cui ce ne accorgiamo. Rilanciare alla
# cieca e' peggio: ogni run fallita costa un quarto d'ora, perche' la
# simulazione macina comunque i suoi mille passi prima che la guardia scopra che
# nessuna chiamata e' passata.
#
# Questa sentinella sonda il fornitore con UNA chiamata ogni tanto --- il costo
# di un sondaggio contro il costo di una run e' mille a uno --- e fa ripartire
# il recupero solo quando la chiamata riesce davvero.
#
# **Si fida del codice d'uscita, non del testo.** `check_provider.py` esce con 1
# quando nessuna chiamata e' riuscita: il ripiego deterministico produce un JSON
# valido, quindi leggere «risposte JSON: 1/1» direbbe il contrario del vero.
#
# Uso:
#   MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/sentinella_fornitore.sh \
#     >> runs/sentinella.log 2>&1 &
#
# Variabili: FORNITORE, MODELLO_LLM, CAMPAGNA, INTERVALLO (secondi), ORE_MAX.
set -u
cd "$(dirname "$0")/.."

FORNITORE=${FORNITORE:-gpu_farm}
MODELLO_LLM=${MODELLO_LLM:-gpt-oss:20b}
CAMPAGNA=${CAMPAGNA:-runs/base_pulita}
ETICHETTA=${ETICHETTA:-$(basename "$CAMPAGNA")}
# Dieci minuti: abbastanza raro da non pesare sulla farm che stiamo aspettando,
# abbastanza fitto da non perdere mezza notte di finestra libera.
INTERVALLO=${INTERVALLO:-600}
ORE_MAX=${ORE_MAX:-48}
PAVIMENTO=${PAVIMENTO:-1.1}
LAVORATORI=${LAVORATORI:-1}

dillo() { echo "[sentinella] $(date -Is) $*"; }

scadenza=$(( $(date +%s) + ORE_MAX * 3600 ))
tentativo=0

dillo "aspetto $FORNITORE/$MODELLO_LLM per $CAMPAGNA; sondo ogni ${INTERVALLO}s, al massimo per ${ORE_MAX}h."

while [ "$(date +%s)" -lt "$scadenza" ]; do
  tentativo=$((tentativo + 1))
  # 0 = risponde; 1 = nessuna chiamata riuscita, cioe' occupato; 2 = non e' un
  # problema di carico ma di configurazione, e aspettare non lo risolve.
  motivo=$(python scripts/check_provider.py --provider "$FORNITORE" \
             --model "$MODELLO_LLM" 2>&1 >/dev/null)
  sonda=$?
  if [ "$sonda" -eq 2 ]; then
    dillo "FERMO: il sondaggio non e' nemmeno partito. Non e' il fornitore"
    dillo "  occupato, e' la configurazione: aspettare non serve."
    dillo "  $(echo "$motivo" | head -3 | tr '\n' ' ')"
    exit 2
  fi
  if [ "$sonda" -eq 0 ]; then
    dillo "il fornitore risponde (sondaggio $tentativo). Preparo il recupero."

    if ! python scripts/piano_recupero.py "$CAMPAGNA"; then
      dillo "non manca nulla da recuperare: ho finito."
      exit 0
    fi

    piano="runs/piano_recupero_${ETICHETTA}.txt"
    dillo "riparto con $piano, $LAVORATORI lavoratore/i."
    PIANO_PRONTO=1 PIANO="$piano" PAVIMENTO="$PAVIMENTO" \
      ETICHETTA="$ETICHETTA" FORNITORE="$FORNITORE" MODELLO_LLM="$MODELLO_LLM" \
      LAVORATORI="$LAVORATORI" ASPETTA_MACCHINA=0 MARSABM_ALLOW_LLM_CALLS=1 \
      bash scripts/coda_base_pulita.sh >> "runs/${ETICHETTA}.log" 2>&1
    esito=$?
    dillo "la coda e' uscita con $esito."
    # 0 = piano finito senza buchi. 2 = la coda si e' fermata perche' il
    # fornitore e' caduto di nuovo. 3 = piano finito ma con run saltate. Negli
    # ultimi due casi c'e' ancora lavoro: si torna ad aspettare invece di
    # dichiarare completato cio' che non lo e'.
    if [ "$esito" -eq 0 ]; then
      dillo "recupero completato."
      exit 0
    elif [ "$esito" -eq 3 ]; then
      dillo "il piano e' finito ma con run saltate: riprovo al prossimo giro."
    else
      dillo "la finestra si e' richiusa: torno ad aspettare."
    fi
  else
    dillo "ancora occupato (sondaggio $tentativo)."
  fi
  sleep "$INTERVALLO"
done

dillo "SCADUTO: $ORE_MAX ore e il fornitore non si e' mai liberato."
dillo "  Le run mancanti sono elencate in runs/saltate_${ETICHETTA}.txt"
dillo "  e il piano per rifarle si rigenera con:"
dillo "  python scripts/piano_recupero.py $CAMPAGNA"
exit 1
