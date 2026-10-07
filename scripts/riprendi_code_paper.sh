#!/usr/bin/env bash
# Riprende le due code del paper dopo una pausa (es. il coprifuoco).
#
# Stessi parametri del lancio del 01/10. PIANO_PRONTO=1 e' essenziale: senza,
# la coda riscriverebbe il piano dai suoi undici bracci, perdendo
# `--copertura-elettrica-vera` e le dieci run dei modelli. Le run gia' nei
# results.jsonl si saltano; quella interrotta a meta' (se c'e') si rifa' da capo.
#
# Uso:  bash scripts/riprendi_code_paper.sh
set -u
cd "$(dirname "$0")/.."
export MARSABM_ALLOW_LLM_CALLS=1 PAVIMENTO=1.1 PIANO_PRONTO=1 ASPETTA_MACCHINA=0

LUCCHETTO=runs/.lucchetto_paper_qwen ETICHETTA=paper_qwen PIANO=runs/piano_paper_qwen.txt \
  FUORI=runs/paper_qwen FORNITORE=gpu_farm4 MODELLO_LLM=Qwen/Qwen3.8-27B-FP8 LAVORATORI=3 \
  nohup bash scripts/coda_base_pulita.sh >> runs/paper_qwen.log 2>&1 &
sleep 3
LUCCHETTO=runs/.lucchetto_paper_oss ETICHETTA=paper_oss PIANO=runs/piano_paper_oss.txt \
  FUORI=runs/paper_oss FORNITORE=gpu_farm MODELLO_LLM=gpt-oss:20b LAVORATORI=2 \
  nohup bash scripts/coda_base_pulita.sh >> runs/paper_oss.log 2>&1 &
echo "code ripartite; avanzamento: python scripts/avanzamento_due.py --paper"
