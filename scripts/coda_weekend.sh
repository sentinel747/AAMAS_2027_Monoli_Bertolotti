#!/usr/bin/env bash
# Coda del fine settimana (venerdi' 11 -> domenica 13 settembre 2026).
#
# **Il coprifuoco e' la ragione per cui questo script esiste.** La macchina si
# spegne a mezzanotte, e una run uccisa a meta' lascia una cartella di seme
# parziale che va poi cancellata a mano (successo l'8 settembre con qwen3.6).
# Qui ogni seme parte SOLO se il tempo stimato per quel mondo lo fa finire
# entro il coprifuoco: quando non ci sta, la coda si ferma pulita e stampa il
# comando per riprenderla. Riprendere e' sicuro perche' il runner salta i semi
# gia' presenti in results.jsonl: nessun lavoro viene rifatto.
#
# **L'ordine e' per valore, non per comodita'.**
#   1. campagna mista (quattro modelli nella stessa colonia): e' l'unica
#      configurazione qualitativamente nuova, ma dipende da gpu_farm4, che
#      l'11 settembre era giu'. Si sonda l'endpoint PRIMA di ogni blocco: se
#      risponde, la mista passa avanti a tutto il resto.
#   2. repliche dei quattro mondi non di riferimento: oggi hanno UNA sola
#      esecuzione per seme, quindi l'effetto si puo' leggere per direzione ma
#      non per grandezza. Una seconda esecuzione da' lo scarto fra esecuzioni
#      identiche anche li'. ~16 h in tutto.
#   3. terza esecuzione degli stessi mondi, se avanza tempo.
#
# Uso (identico ogni giorno, anche per riprendere):
#   MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_weekend.sh \
#     >> runs/weekend/coda_weekend.log 2>&1 &
#
# Variabili: COPRIFUOCO=23:45 (ora locale), MARGINE=1.25 (moltiplicatore sulla
# durata massima gia' osservata per quel mondo).
set -u
cd "$(dirname "$0")/.."
COPRIFUOCO=${COPRIFUOCO:-23:45}
MARGINE=${MARGINE:-1.25}
SEMI=${SEMI:-"3 4 5 6 7"}
COMUNI="--steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --wait 120 --administrators --thinking off"
mkdir -p runs/weekend

dillo() { echo "[weekend] $(date -Is) $*"; }

# Minuti da adesso al coprifuoco di oggi; negativo (o zero) se e' passato.
minuti_al_coprifuoco() {
  # Un orario gia' passato significa la notte prossima, non quella trascorsa:
  # cosi' COPRIFUOCO=01:00 vuol dire l'una di notte e non l'una di stamattina.
  local ora_fine ora_ora
  ora_fine=$(date -d "today $COPRIFUOCO" +%s 2>/dev/null) || return 1
  ora_ora=$(date +%s)
  if [ "$ora_fine" -le "$ora_ora" ]; then
    ora_fine=$(date -d "tomorrow $COPRIFUOCO" +%s 2>/dev/null) || return 1
  fi
  echo $(( (ora_fine - ora_ora) / 60 ))
}

# Vero se un lavoro di $1 minuti stimati ci sta prima del coprifuoco.
ci_sta() {
  local stima=$1 restano
  restano=$(minuti_al_coprifuoco)
  [ "$restano" -ge "$stima" ]
}

ferma_per_oggi() {
  dillo "COPRIFUOCO: $1 richiede ~$2 min e ne restano $(minuti_al_coprifuoco)."
  dillo "Niente parte a meta'. Riprendi domani con lo stesso comando:"
  dillo "  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_weekend.sh >> runs/weekend/coda_weekend.log 2>&1 &"
  exit 0
}

# Un seme alla volta: il coprifuoco si puo' far valere solo a questa grana.
# $1 nome leggibile, $2 minuti stimati per seme, $3.. il comando.
esegui_semi() {
  local nome=$1 stima=$2; shift 2
  for seme in $SEMI; do
    ci_sta "$stima" || ferma_per_oggi "$nome seme $seme" "$stima"
    dillo "$nome seme $seme (stima $stima min, restano $(minuti_al_coprifuoco))"
    "$@" --seeds "$seme" || dillo "ERRORE in $nome seme $seme"
  done
}

# gpu_farm4 risponde? La campagna mista non ha senso senza il quarto modello.
farm4_su() {
  # **Nessun consenso, nessuna chiamata.** La chiave qui finisce in un'intestazione
  # HTTP verso lo stesso endpoint che tutto il livello Python protegge: senza
  # questa riga sarebbe l'unico punto del repository in cui una credenziale
  # raggiunge la rete fuori dall'interlock.
  [ "${MARSABM_ALLOW_LLM_CALLS:-}" = "1" ] || return 1
  curl -s -m 20 -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${GPU_FARM_KEY4:-}" \
    http://llm-server.example:8000/v1/models 2>/dev/null | grep -q '^2'
}

campagna_mista() {
  [ -f runs/misto/amm_misto/COMPLETA ] && return 0
  if ! farm4_su; then
    dillo "gpu_farm4 ancora giu': la campagna mista resta in attesa."
    return 0
  fi
  dillo "gpu_farm4 risponde: la campagna mista passa avanti."
  esegui_semi "misto" 120 \
    python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
      --admin-models gpu_farm:gpt-oss:20b gpu_farm:gpt-oss:120b gpu_farm4:Qwen/Qwen3.8-27B-FP8 gpu_farm:qwen3.6:27b \
      --map-profile balanced $COMUNI --out runs/misto/amm_misto
  # Cinque righe = campagna chiusa: la prossima ripresa non la rifa'.
  [ "$(wc -l < runs/misto/amm_misto/results.jsonl 2>/dev/null || echo 0)" -ge 5 ] \
    && touch runs/misto/amm_misto/COMPLETA && dillo "CAMPAGNA MISTA COMPLETA"
}

# mondo|profilo|cartella della coppia|minuti stimati per seme (massimo gia'
# osservato su quel mondo, moltiplicato per MARGINE e arrotondato).
MONDI="ice_rich|ice_rich|llm_completo_amm|66
fragmented|fragmented|llm_completo_amm|79
high_hazard|high_hazard|llm_completo_amm|94
scarce_resources|scarce_resources|llm_completo_amm_v2|50"

repliche() {
  local suffisso=$1
  while IFS='|' read -r mondo profilo coppia stima; do
    [ -z "$mondo" ] && continue
    local out="runs/mondi/$mondo/${coppia}_${suffisso}"
    if [ "$(wc -l < "$out/results.jsonl" 2>/dev/null || echo 0)" -ge 5 ]; then
      dillo "$mondo $suffisso gia' completa, salto."
      continue
    fi
    campagna_mista
    esegui_semi "$mondo $suffisso" "$stima" \
      python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
        --map-profile "$profilo" $COMUNI --out "$out"
  done <<< "$MONDI"
}

# **Il freno a mano.** Questa coda e' anche la coda a cui altre code passano la
# macchina quando finiscono. Quando il lavoro piu' importante e' un altro, il
# passaggio di consegne va potuto fermare senza uccidere processi a mano e senza
# toccare gli script che stanno girando: basta creare il file.
if [ -f runs/.coda_sospesa ]; then
  dillo "SOSPESA dal freno a mano (runs/.coda_sospesa). Motivo:"
  sed 's/^/  /' runs/.coda_sospesa
  dillo "Per riprendere: rm runs/.coda_sospesa e rilancia lo stesso comando."
  exit 0
fi

dillo "coda del fine settimana avviata; coprifuoco alle $COPRIFUOCO (restano $(minuti_al_coprifuoco) min)"
campagna_mista
repliche rep2
repliche rep3
dillo "CODA WEEKEND COMPLETA"
