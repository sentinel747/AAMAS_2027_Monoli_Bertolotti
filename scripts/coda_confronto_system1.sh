#!/usr/bin/env bash
# Confronto System 1 / LLM a parita' di modello e di impostazione (2026-09-28).
#
# Domanda: le differenze fra il governo System 1 (sceglie in due livelli fra
# regole valide) e il governatore LLM (scrive la legge) dipendono dal FORMATO,
# dal MODELLO o dall'IMPOSTAZIONE della run? Finora il confronto mescolava tutte
# e tre (tesi: 1000 passi, cadenza 25, strato ambientale, dotazione 0,6;
# System 1 v4: 2000 passi, cadenza 20, niente strato). Qui il modello e' fisso
# (Qwen3.8-27B-FP8, stesso server vLLM della tesi) e si riempiono le due caselle
# mancanti, piu' Jev per l'effetto del modello a formato fisso:
#
#   coda A (impostazione tesi, confronto con runs/base_qwen/ctrl_none e con il
#   braccio della tesi con indicatore corretto, runs/controllo_copertura_20260926):
#     - Qwen SCEGLIE (System 1 v4 via gateway locale 8008), semi 3-7 + seme 3 ripetuto
#     - Jev SCEGLIE (System 1 v4, TypeSafe, tetto cumulativo 4,6 USD), semi 3-7
#   coda B (impostazione System 1 v4, confronto con campaign_v2h none e con Qwen v4):
#     - Qwen SCRIVE (LLM, prompt D, amministratori), semi 3-7, due esecuzioni
#
# Tutti con --copertura-elettrica-vera (il braccio della tesi di riferimento e'
# quello con l'indicatore corretto; il System 1 v4 l'indicatore non lo usa) e
# con attesa 600 s, perche' nessuna tornata vada persa.
#
# Richiede il gateway acceso:  VLLM_BASE_URL=http://llm-server.example:8000/v1
#   VLLM_API_KEY=$GPU_FARM_KEY4 VLLM_MODEL=Qwen/Qwen3.8-27B-FP8
#   VLLM_DECISION_API=chat SEMIF_PORT=8008 SEMIF_DECISION_TIMEOUT_SECONDS=60
#   python scripts/run_gateway.py   (in Desktop/JEVlocal/semif-runtime)
# Con l'attesa predefinita del gateway (10 s) e la coda B in parallelo sullo
# stesso vLLM, il 28/09 il 2-15% delle decisioni e' finito in 502 -> ripiego.
# Per questo: gateway a 60 s e coda A DA SOLA sul server Qwen, prima della B.
# Ogni run System 1 valida deve avere zero fallback_reasons http_status_* e
# circuit_open in semantic_telemetry.json.
#
# Riprendibile: salta le run con results.json. Stato: python scripts/avanzamento_mondi.py
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 bash scripts/coda_confronto_system1.sh A
#       MARSABM_ALLOW_LLM_CALLS=1 bash scripts/coda_confronto_system1.sh B
set -u
cd "$(dirname "$0")/.."
QUALE=${1:?specificare la coda: A oppure B}
OUT=runs/confronto_system1
mkdir -p "$OUT/tesi" "$OUT/system1"

TESI="--steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --strato-ambientale --correzioni-motore --pavimento-cantieri 1.1 --map-profile balanced --log-interval 250 --snapshot-interval 25 --copertura-elettrica-vera --administrators --wait 600"
S1="--arms semif --semantic-candidate-profile v4 --semantic-timeout 120"
QWEN_S1="--semantic-provider rest --jev-endpoint http://127.0.0.1:8008"
JEV_S1="--semantic-provider typesafe --typesafe-budget-usd 4.6"
V4="--steps 2000 --agents 300 --cadence 20 --administrators --wait 600 --copertura-elettrica-vera"
QWEN_LLM="--arms llm --provider gpu_farm4 --model Qwen/Qwen3.8-27B-FP8 --variante-prompt D"

LISTA="$OUT/coda_$QUALE.txt"
: > "$LISTA"
aggiungi() {  # $1 cartella, $2 seme, resto: opzioni
  local dir=$1 seme=$2; shift 2
  [ -f "$dir/results.json" ] && return
  rm -rf "$dir"
  echo "python scripts/run_governor_experiment.py $* --seeds $seme --out $dir" >> "$LISTA"
}
case "$QUALE" in
  A)
    PARALLELE=${PARALLELE:-2}
    for s in 3 4 5 6 7; do aggiungi "$OUT/tesi/qwen_s1_s$s" $s $TESI $S1 $QWEN_S1; done
    aggiungi "$OUT/tesi/qwen_s1_rep2_s3" 3 $TESI $S1 $QWEN_S1
    # Jev solo su richiesta (CON_JEV=1): il 28/09 l'account TypeSafe ha risposto
    # 402 a ogni decisione (credito esaurito) e le run sono finite tutte in ripiego.
    if [ "${CON_JEV:-0}" = "1" ]; then
      for s in 3 4 5 6 7; do aggiungi "$OUT/tesi/jev_s1_s$s" $s $TESI $S1 $JEV_S1; done
    fi
    ;;
  B)
    PARALLELE=${PARALLELE:-3}
    for s in 3 4 5 6 7; do
      aggiungi "$OUT/system1/qwen_llm_s$s" $s $V4 $QWEN_LLM
      aggiungi "$OUT/system1/qwen_llm_rep2_s$s" $s $V4 $QWEN_LLM
    done
    ;;
  *) echo "coda sconosciuta: $QUALE" >&2; exit 2 ;;
esac
echo "[$QUALE] $(date -Is) $(wc -l < "$LISTA") run da fare" >> "$OUT/coda.log"
xargs -P "$PARALLELE" -I CMD sh -c 'CMD > /dev/null 2>&1; echo "['"$QUALE"'] $(date -Is) fine: $(echo CMD | grep -o "\-\-out [^ ]*")"' < "$LISTA" >> "$OUT/coda.log" 2>&1
echo "[$QUALE] $(date -Is) coda esaurita" >> "$OUT/coda.log"
