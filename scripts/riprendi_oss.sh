#!/usr/bin/env bash
# Le otto run che mancano a OSS-55, e nient'altro.
#
# **Perche' un piano gia' scritto e non la coda normale.** La coda, rilanciata,
# rigenera il piano completo e ricontrolla cinquantacinque righe per scoprire
# che quarantasette sono gia' fatte. Funziona, ma il piano nuovo porterebbe gli
# argomenti di oggi, non quelli con cui le altre run sono state eseguite: dentro
# la stessa campagna sarebbe una differenza di configurazione, cioe' il difetto
# che questo progetto passa il tempo a evitare. `PIANO_PRONTO=1` gli dice di
# usare il piano di recupero cosi' com'e', copiato riga per riga dall'originale.
#
# **L'ordine non e' l'ordine del piano.** Le prime quattro righe sono il braccio
# ripetuto, quello da cui esce il metro del rumore. Oggi la campagna OSS ne ha
# una coppia sola su cinque, e con una coppia sola `esiti_campagna.py` si
# rifiuta di emettere qualunque verdetto --- giustamente, perche' una singola
# estrazione non e' una stima della varianza del modello. Finche' quelle quattro
# non ci sono, le altre quarantasette non si possono leggere: sono le uniche che
# non ci si puo' permettere di perdere due volte.
#
# **Quanti lavoratori.** Predefinito uno, perche' la farm di Ollama e'
# condivisa e cinque richieste in parallelo sono cio' che ha prodotto la coda di
# latenze da cui e' nato il timeout a 120 secondi. Si alza con
# `LAVORATORI=2 bash scripts/riprendi_oss.sh`, e conviene farlo solo dopo aver
# visto `check_provider.py` rispondere in fretta: col container appena riavviato
# la latenza mediana e' scesa da 37 a 12 secondi, e a quel punto due richieste
# insieme la farm le regge.
#
# La divisione e' per BRACCIO, mai per riga (`dividi_piano.py`): due processi
# che appendono allo stesso `results.jsonl` lo corrompono senza dare errore. Su
# queste otto run il taglio viene bene da solo --- il braccio del rumore, che
# ha quattro run, finisce tutto su un lavoratore e le altre quattro sull'altro.
#
# Il piano di recupero porta `--wait 300`.
#
# Prima di lanciare, verificare che il fornitore risponda davvero:
#   MARSABM_ALLOW_LLM_CALLS=1 python scripts/check_provider.py \
#     --provider gpu_farm --model gpt-oss:20b
set -u
cd "$(dirname "$0")/.."

PIANO=runs/piano_recupero_base_pulita.txt
if [ ! -s "$PIANO" ]; then
  echo "manca $PIANO: rigenerarlo con python scripts/piano_recupero.py" >&2
  exit 1
fi

export MARSABM_ALLOW_LLM_CALLS=1
# `PAVIMENTO` e' obbligatorio nella coda. Col piano gia' scritto non entra
# in nessuna riga --- le otto righe portano gia' il proprio
# `--pavimento-cantieri 1.1` --- ma senza di lui la coda esce subito.
PIANO_PRONTO=1 PIANO="$PIANO" LAVORATORI="${LAVORATORI:-1}" PAVIMENTO=1.1 \
  bash scripts/coda_base_pulita.sh "$@"
