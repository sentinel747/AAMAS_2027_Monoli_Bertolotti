#!/usr/bin/env bash
# Un lucchetto per la macchina, condiviso da tutte le code.
#
# **Perche' il controllo «e' occupata?» non basta.** Le code guardano se esiste
# un processo di simulazione e, se non c'e', partono. Con una coda sola va bene.
# Con due code in attesa della stessa macchina --- ed e' la situazione normale
# adesso, che la campagna sui prompt e le baseline aspettano entrambe --- fra il
# controllo e l'avvio c'e' una finestra in cui tutte e due vedono la macchina
# libera e tutte e due partono. Due simulazioni insieme non danno risultati
# sbagliati, ma si rubano la CPU e la farm a vicenda, e i tempi misurati
# diventano incomparabili con quelli dell'archivio.
#
# `mkdir` e' atomico su ogni filesystem: o crea la cartella, o fallisce. E' il
# lucchetto piu' semplice che funzioni davvero.
#
# Il lucchetto porta dentro il PID di chi lo tiene: un lucchetto abbandonato da
# un processo morto --- una coda uccisa, un riavvio --- viene riconosciuto e
# tolto, invece di bloccare la macchina per sempre.
#
# Uso, da dentro un'altra coda:
#   . scripts/lucchetto_macchina.sh
#   prendi_lucchetto "prompt"      # aspetta finche' non e' suo
#   ...esegui la run...
#   lascia_lucchetto

# Le code che caricano questo file hanno tutte `set -u`; lo si ripete qui perche'
# una variabile non definita in un lucchetto condiviso fra code e' il modo piu'
# silenzioso di cancellare la cartella sbagliata con `rm -rf`.
set -u

LUCCHETTO=${LUCCHETTO:-runs/.lucchetto_macchina}

_vivo() {
  # Un PID di bash o python ancora in vita? `kill -0` non manda segnali, chiede
  # solo se il processo esiste.
  kill -0 "$1" 2>/dev/null
}

prendi_lucchetto() {
  local chi=${1:-coda} atteso=0
  while ! mkdir "$LUCCHETTO" 2>/dev/null; do
    local proprietario
    proprietario=$(cat "$LUCCHETTO/pid" 2>/dev/null || echo "")
    if [ -n "$proprietario" ] && ! _vivo "$proprietario"; then
      echo "[$chi] $(date -Is) lucchetto abbandonato dal pid $proprietario, lo tolgo"
      rm -rf "$LUCCHETTO"
      continue
    fi
    if [ "$atteso" -eq 0 ]; then
      echo "[$chi] $(date -Is) la macchina e' presa da $(cat "$LUCCHETTO/chi" 2>/dev/null || echo 'un altro'), aspetto"
      atteso=1
    fi
    sleep 60
  done
  echo "$$" > "$LUCCHETTO/pid"
  echo "$chi" > "$LUCCHETTO/chi"
}

lascia_lucchetto() {
  rm -rf "$LUCCHETTO"
}

# Aspetta che un piano di coda sia finito, non solo che la macchina sia libera.
#
# **Perche' serve oltre al lucchetto.** Una coda avviata prima che il lucchetto
# esistesse non lo prende, e fra una sua run e la successiva la macchina sembra
# libera per qualche secondo. Una coda nuova che guardasse solo quello si
# infilerebbe in mezzo a una campagna altrui. Qui si aspetta la condizione
# giusta: che ogni riga del piano abbia il suo risultato.
#
# Il piano ha una riga per run: "<variante> <seme> <cartella>".
aspetta_piano() {
  local piano=$1 chi=${2:-coda} detto=0
  [ -f "$piano" ] || return 0
  while true; do
    local mancanti=0
    while read -r _variante seme fuori; do
      [ -z "${seme:-}" ] && continue
      grep -q "\"seed\": $seme," "$fuori/results.jsonl" 2>/dev/null || mancanti=$((mancanti + 1))
    done < "$piano"
    [ "$mancanti" -eq 0 ] && break
    if [ "$detto" -eq 0 ]; then
      echo "[$chi] $(date -Is) aspetto che finisca $piano: mancano $mancanti run"
      detto=1
    fi
    sleep 300
  done
  echo "[$chi] $(date -Is) $piano e' completo"
}
