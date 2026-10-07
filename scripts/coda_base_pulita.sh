#!/usr/bin/env bash
# La base pulita: undici bracci, cinque semi, un mondo solo.
#
# **Perche' si riparte da zero.** Le campagne in archivio sono state misurate
# con un simulatore che aveva difetti trovati dopo: il prompt inglese conteneva
# italiano, il gradino cieco perdeva la cecita' dalla seconda tornata, una
# risposta fuori schema abrogava la legge in vigore, le capienze che il
# governatore legge dichiaravano l'ossigeno di impianti che non lo producono.
# Ognuno preso da solo sposta poco; tutti insieme rendono impossibile dire «so
# esattamente che cosa ho misurato», che e' la sola cosa che una tesi deve poter
# dire. Qui il simulatore e' corretto, il regime e' dichiarato, e ogni braccio
# dista dal riferimento per UNA cosa sola.
#
# **Il disegno.** Non un incrocio completo --- lingua x parola x gerarchia x
# contesto x configurazione sono 49 bracci e nove giorni di macchina --- ma un
# riferimento e un braccio per ogni variabile, che e' quanto basta a leggere
# ogni effetto per differenza:
#
#   0, A, B, F, C, D   le sei caselle del prompt. La parola si legge in
#                      italiano (A-0) e in inglese (C-F); la gerarchia in
#                      italiano (B-A) e in inglese (D-C); la lingua in tre
#                      punti (C-A, F-0, D-B). Nessuna casella nuova serve.
#   cieco due livelli  il dominio tolto a governo E amministratori.
#   solo governatore   serve il livello locale?
#   solo amministratori serve il centro?
#   D ripetuto         la varianza di campionamento del modello, senza la
#                      quale un effetto e una pescata diversa si somigliano.
#   baseline           il metro di tutto.
#
# **Il regime, dichiarato una volta sola.** Strato ambientale ACCESO: il mondo
# non e' piu' congelato, e l'obiezione «il vostro clima e' fermo» non si puo'
# piu' fare; il confronto resta pulito perche' e' appaiato per seme, quindi i
# due bracci vedono lo stesso meteo. Redistribuzione SPENTA: accesa la colonia
# crolla da ~1500 a ~600 abitanti e in un caso muore del tutto, e mettere in
# comune il surplus in automatico toglie al governo proprio il problema che deve
# risolvere. Capienze corrette ACCESE. Pavimento dei cantieri: lo decide il
# pilota, e va passato in PAVIMENTO.
#
# Uso:
#   PAVIMENTO=0.95 MARSABM_ALLOW_LLM_CALLS=1 \
#     nohup bash scripts/coda_base_pulita.sh >> runs/base_pulita.log 2>&1 &
#   python scripts/avanzamento.py --piano runs/piano_base_pulita.txt --log runs/base_pulita.log
set -u
cd "$(dirname "$0")/.."

SEMI=${SEMI:-"3 4 5 6 7"}
# L'etichetta distingue due istanze della stessa coda su GPU diverse: da lei
# nascono cartella, piano, log, lucchetto e sentinella. Senza, la seconda coda
# aspetterebbe il lucchetto della prima e si fermerebbe quando si ferma l'altra.
ETICHETTA=${ETICHETTA:-base_pulita}
PIANO=${PIANO:-runs/piano_$ETICHETTA.txt}
FUORI=${FUORI:-runs/$ETICHETTA}
FORNITORE=${FORNITORE:-gpu_farm}
MODELLO_LLM=${MODELLO_LLM:-gpt-oss:20b}
PAVIMENTO=${PAVIMENTO:?serve PAVIMENTO=<valore scelto dal pilota>}
# Quanti processi in parallelo SULLA STESSA GPU. Il valore predefinito e' uno, e
# il motivo e' misurato: la resa di un fornitore e' fissa --- 3,7 chiamate al
# secondo su Ollama, 6,3 su vLLM --- quindi due code sulla stessa GPU se la
# dividono e guadagnano solo la sovrapposizione fra calcolo e attesa (1,58x,
# misurato). Il parallelismo che rende davvero e' fra GPU diverse: due istanze di
# questa coda con ETICHETTA e FORNITORE diversi.
LAVORATORI=${LAVORATORI:-1}
SALTATE=runs/saltate_$ETICHETTA.txt
# **Quante run di fila possono fallire prima di concludere che il fornitore e'
# giu'.** Sotto questa soglia si riprova; raggiunta, si smette: insistere
# costerebbe un quarto d'ora per run e riempirebbe l'elenco delle saltate di
# run che non hanno nulla che non va.
ROTTE_DI_FILA_MAX=${ROTTE_DI_FILA_MAX:-2}
rotte_di_fila=0

# **Due thread per simulazione, non venticinque.** Misurato: stessa run, 100s
# con due thread contro 104s a thread liberi, e risultati identici alla cifra.
# I thread in piu' non lavoravano, occupavano posto; limitarli e' cio' che
# permette di far girare cinque simulazioni dove ne giravano due.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
TAB=$'\t'

# **Attesa a 300s, non 120.** Misurato con cinque simulazioni in parallelo: la
# latenza mediana di una tornata sta sui 40s, ma la coda arriva a 118s su Ollama
# --- e ogni volta che supera il tetto il governatore perde la tornata e il
# cancello boccia tutta la run. Alzare il tetto non allenta il cancello: una run
# ammessa non lo tocca mai, quindi il valore puo' cambiare solo l'esito di quelle
# che oggi si buttano. L'ammissione resta zero tornate perse.
REGIME="--strato-ambientale --correzioni-motore --pavimento-cantieri $PAVIMENTO"
COMUNI="--steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 \
--snapshot-interval 25 --wait 300 --map-profile balanced"
MODELLO="--arms llm --provider $FORNITORE --model $MODELLO_LLM"

mkdir -p runs
dillo() { echo "[${ETICHETTA//[^a-z_]/_}] $(date -Is) $*"; }
. "$(dirname "$0")/lucchetto_macchina.sh"

BLOCCATI=${BLOCCATI:-$(grep -v '^#' runs/.pid_bloccati 2>/dev/null | grep -E '^[0-9]+$' | paste -sd, -)}
occupata() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "\$relitti = @($BLOCCATI); if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_governor_experiment*' -and \$relitti -notcontains \$_.ProcessId }) { 'si' } else { 'no' }" \
    2>/dev/null | tr -d '\r' | grep -q '^si$'
}

riga() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$PIANO"; }

# **Un piano gia' scritto si usa, non si riscrive.** Normalmente la coda
# rigenera il piano a ogni avvio, cosi' che corrisponda sempre agli undici
# bracci dichiarati e nessuno esegua per sbaglio una campagna diversa. Ma
# quando restano poche run sparse da recuperare, l'ordine in cui si rifanno
# conta --- il braccio che misura il rumore va per primo, perche' senza quello
# nessun altro numero si legge --- e allora il piano lo si scrive a mano.
if [ "${PIANO_PRONTO:-0}" != "1" ]; then

: > "$PIANO"

# 1. La baseline per prima: e' il metro, e senza di lei nessun confronto esiste.
for s in $SEMI; do
  riga "baseline" "$FUORI/ctrl_none" "$s" "--arms none $REGIME $COMUNI"
done

# 2. Il riferimento: inglese, «categoria», con la gerarchia.
for s in $SEMI; do
  riga "RIF: D (en, categoria, gerarchia)" "$FUORI/variante_D" "$s" \
    "$MODELLO --variante-prompt D --administrators $REGIME $COMUNI"
done

# 3. Le altre cinque caselle del prompt.
for v in C F B A 0; do
  for s in $SEMI; do
    riga "prompt $v" "$FUORI/variante_$v" "$s" \
      "$MODELLO --variante-prompt $v --administrators $REGIME $COMUNI"
  done
done

# 4. Il dominio tolto a tutti e due i livelli.
for s in $SEMI; do
  riga "cieco a due livelli" "$FUORI/cieco_due_livelli" "$s" \
    "$MODELLO --variante-prompt D --administrators --context-levels cieco \
--livello-amministratori cieco $REGIME $COMUNI"
done

# 5. I due livelli presi da soli.
for s in $SEMI; do
  riga "solo governatore" "$FUORI/solo_governatore" "$s" \
    "$MODELLO --variante-prompt D $REGIME $COMUNI"
done
for s in $SEMI; do
  # **`--admin-arm llm` non e' un dettaglio: senza, il braccio e' inerte.**
  # Gli amministratori seguono per difetto il braccio del governo, che qui e'
  # «nessuno»: ricevono i distretti, non intervengono mai, e nessun contatore
  # protesta perche' non c'e' niente da contare. Misurato sul pilota:
  # 0 interventi senza il flag, 23 con.
  riga "solo amministratori" "$FUORI/solo_amministratori" "$s" \
    "--arms none --administrators --admin-arm llm --variante-prompt D \
--provider $FORNITORE --model $MODELLO_LLM $REGIME $COMUNI"
done

# 6. Il riferimento una seconda volta: e' la misura di quanto il modello pesca
#    diverso a parita' di tutto, ed e' la sola difesa contro lo scambiare una
#    pescata per un effetto.
for s in $SEMI; do
  riga "RIF ripetuto" "$FUORI/variante_D_rep2" "$s" \
    "$MODELLO --variante-prompt D --administrators $REGIME $COMUNI"
done

fi

dillo "piano in $PIANO: $(wc -l < "$PIANO") run, $FORNITORE/$MODELLO_LLM, pavimento $PAVIMENTO"

if [ "${ASPETTA_MACCHINA:-1}" = "1" ]; then
  dillo "aspetto che la macchina sia libera"
  while occupata; do sleep 120; done
else
  dillo "non aspetto: questa coda gira apposta insieme all'altra, su un'altra GPU."
fi
prendi_lucchetto "$ETICHETTA"
trap lascia_lucchetto EXIT INT TERM
dillo "si parte con $LAVORATORI lavoratori per GPU. Nessun coprifuoco."
# **L'elenco non si azzera quando si sta recuperando.** Con un piano gia'
# scritto la coda e' un secondo passaggio, e le run saltate dal primo sono la
# memoria di cio' che manca: la sentinella rilancia con la stessa etichetta,
# quindi troncare qui cancellerebbe quella memoria a ogni ripartenza.
[ "${PIANO_PRONTO:-0}" = "1" ] || : > "$SALTATE"

# Una run, con una riprova. Restituisce 0 se e' buona, 1 se va saltata.
#
# **Perche' due tentativi e non uno.** I guasti che il cancello coglie ---
# fornitore caduto, risposta non interpretabile, tornata persa --- sono quasi
# sempre passeggeri: la stessa run rifatta subito dopo di solito passa. Ma se
# fallisce due volte non e' piu' sfortuna, e insistere significherebbe bloccare
# la coda su un braccio rotto mentre gli altri dieci aspettano.
una_run() {  # io, nome, fuori, seme, argomenti
  local io="$1" nome="$2" fuori="$3" seme="$4" argomenti="$5"
  local tentativo=1
  while [ "$tentativo" -le 2 ]; do
    [ "$tentativo" -gt 1 ] && dillo "[$io] $nome seme $seme: tentativo $tentativo"
    python scripts/run_governor_experiment.py $argomenti --seeds "$seme" --out "$fuori" \
      || dillo "[$io] ERRORE nell'esecuzione di $nome seme $seme"
    if python scripts/qualita_run.py "$fuori" --seme "$seme"; then
      rotte_di_fila=0
      return 0
    fi
    dillo "[$io] $nome seme $seme NON COMPARABILE (vedi sopra): la metto da parte."
    python scripts/scarta_run.py "$fuori" "$seme" \
      --motivo "cancello di qualita', tentativo $tentativo" >/dev/null 2>&1 \
      || dillo "[$io] non sono riuscito a scartarla: guardala a mano."
    tentativo=$((tentativo + 1))
  done
  dillo "[$io] SALTO $nome seme $seme dopo due tentativi falliti."
  printf '%s\tseme %s\t%s\n' "$nome" "$seme" "$fuori" >> "$SALTATE"
  rotte_di_fila=$((rotte_di_fila + 1))
  if [ "$rotte_di_fila" -ge "$ROTTE_DI_FILA_MAX" ]; then
    dillo "FERMO: $rotte_di_fila run di fila non sono riuscite. Il fornitore"
    dillo "  non risponde: ogni altra run costerebbe un quarto d'ora per nulla"
    dillo "  e finirebbe fra le saltate senza averlo meritato."
    dillo "  Verificare con: MARSABM_ALLOW_LLM_CALLS=1 python scripts/check_provider.py"
    dillo "  --provider $FORNITORE --model $MODELLO_LLM"
    dillo "  Le run gia' fatte restano: la coda riparte da dove si e' fermata."
    return 2
  fi
  return 1
}

lavora() {  # piano, etichetta
  local piano="$1" io="$2"
  while IFS="$TAB" read -r nome fuori seme argomenti; do
    [ -z "${nome:-}" ] && continue
    # Un piano scritto su Windows finisce con \r: senza toglierlo, l'ultimo
    # argomento arriva alla simulazione con un ritorno a capo attaccato.
    argomenti=${argomenti%$'\r'}
    seme=${seme%$'\r'}
    if grep -q "\"seed\": $seme," "$fuori/results.jsonl" 2>/dev/null; then
      dillo "[$io] $nome seme $seme gia' fatta, salto."
      continue
    fi
    dillo "[$io] $nome seme $seme"
    una_run "$io" "$nome" "$fuori" "$seme" "$argomenti"
    # 2 = il fornitore e' giu': non ha senso passare alla prossima riga.
    [ "$?" -eq 2 ] && { dillo "[$io] mi fermo qui."; return 2; }
  done < "$piano"
  dillo "[$io] piano finito"
}

# **L'esito dei lavoratori si raccoglie.** Senza, il codice d'uscita dello
# script sarebbe quello dell'ultima stampa --- cioe' sempre zero --- e chi
# rilancia la coda in automatico crederebbe a un piano finito bene anche quando
# si e' fermata a meta'.
fermato=0
if [ "$LAVORATORI" -le 1 ]; then
  lavora "$PIANO" "solo" || [ "$?" -ne 2 ] || fermato=1
else
  python scripts/dividi_piano.py "$PIANO" "$LAVORATORI"
  pids=""
  i=1
  while [ "$i" -le "$LAVORATORI" ]; do
    lavora "$PIANO.$i" "$i" &
    pids="$pids $!"
    i=$((i + 1))
    # Sfasati: due run che partono nello stesso istante si contendono il
    # caricamento del modello sulla farm e la lettura dei dati climatici.
    sleep 45
  done
  # Basta un lavoratore che si e' arreso per dire che il fornitore e' caduto.
  for p in $pids; do wait "$p" || [ "$?" -ne 2 ] || fermato=1; done
fi

# `grep -c` su un file vuoto stampa gia' 0 ed esce 1: senza il `|| true` il
# ripiego aggiungerebbe una seconda riga e il confronto numerico fallirebbe.
saltate=$(grep -c . "$SALTATE" 2>/dev/null || true)
saltate=${saltate:-0}

if [ "$fermato" -eq 1 ]; then
  dillo "INTERROTTA: $ETICHETTA. Il fornitore non risponde; il piano non e'"
  dillo "  stato finito. Le run gia' fatte restano e la coda riparte da li'."
  exit 2
fi
if [ "$saltate" -gt 0 ]; then
  dillo "PIANO FINITO CON $saltate RUN SALTATE: $ETICHETTA"
  dillo "elenco in $SALTATE --- vanno rifatte a mano dopo aver capito il perche'."
  cat "$SALTATE"
  exit 3
fi
dillo "COMPLETA: $ETICHETTA"
