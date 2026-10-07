#!/usr/bin/env bash
# Copia di sicurezza sul NAS, durante la campagna e non alla fine.
#
# **Perche' durante.** Cinquantacinque run per venti ore su un portatile senza
# copia: se quel disco parte a meta' notte non c'e' niente da recuperare, c'e'
# tutto da rifare. Copiando mentre la campagna va, un guasto lascia comunque
# tutto quello che era stato fatto fino a quel momento.
#
# **Non cancella mai niente.** Nessun `/MIR`, nessun `/PURGE`, nessun `/MOVE`:
# solo `/E /XO`, che copia i file nuovi o piu' recenti e lascia stare tutto il
# resto. Se un file sparisce dal portatile, la sua copia sul NAS resta. E' la
# differenza fra una copia di sicurezza e uno specchio, ed e' tutta la
# differenza che conta: uno specchio propaga anche le cancellazioni.
#
# **Che cosa NON va sul NAS.** `PrisonerStreamlit/` non esce da questa macchina,
# per istruzione esplicita. `data/` sono quindici gigabyte di dati climatici di
# terze parti, riscaricabili. `outputs/` ne sono altri sessantotto, di run vecchie
# gia' superate. Copiarli intaserebbe la rete per ore senza proteggere niente di
# insostituibile, e la copia deve stare dietro alla campagna, non davanti.
#
# Uso:
#   nohup bash scripts/copia_di_sicurezza.sh >> runs/copia.log 2>&1 &
#   bash scripts/copia_di_sicurezza.sh --una-volta      # una passata sola
set -u
cd "$(dirname "$0")/.."

DESTINAZIONE=${DESTINAZIONE:-'Z:\BackupRunTesi'}
OGNI=${OGNI:-600}          # secondi fra una passata e l'altra
UNA_VOLTA=${1:-}

# Il percorso in forma Windows, con le barre rovesce: robocopy le vuole cosi'.
QUI=$(pwd -W 2>/dev/null || pwd | sed 's|^/\([a-z]\)|\U\1:|')
QUI=$(printf '%s' "$QUI" | tr '/' '\\')

dillo() { echo "[copia] $(date -Is) $*"; }

raggiungibile() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "if (Test-Path '$DESTINAZIONE') { 'si' } else { 'no' }" 2>/dev/null \
    | tr -d '\r' | grep -q '^si$'
}

# robocopy considera "successo" ogni codice sotto 8; da 8 in su sono errori veri.
copia() {  # sorgente, sottocartella di destinazione, esclusioni...
  local da="$1" a="$2"; shift 2
  powershell.exe -NoProfile -NonInteractive -Command \
    "robocopy '$da' '$DESTINAZIONE\\$a' /E /XO /R:2 /W:5 /NFL /NDL /NJH /NJS /NP $* | Out-Null; if (\$LASTEXITCODE -ge 8) { exit 1 } else { exit 0 }" \
    >/dev/null 2>&1
}

passata() {
  if ! raggiungibile; then
    dillo "NAS non raggiungibile ($DESTINAZIONE): salto questa passata."
    return 0
  fi
  local inizio=$(date +%s)

  # 1. La campagna: e' la cosa insostituibile.
  if [ -d runs/base_pulita ]; then
    copia "$(pwd -W 2>/dev/null || pwd)\\runs\\base_pulita" 'runs\base_pulita' \
      || dillo "ERRORE copiando runs/base_pulita"
  fi

  # 2. Piani e registri della coda: senza di loro le run non si sanno leggere.
  for f in runs/piano_base_pulita.txt runs/base_pulita.log runs/piano_pilota.txt runs/pilota.log; do
    [ -f "$f" ] && powershell.exe -NoProfile -NonInteractive -Command \
      "New-Item -ItemType Directory -Force '$DESTINAZIONE\\runs' | Out-Null; Copy-Item '$(pwd -W 2>/dev/null || pwd)\\$(echo "$f" | tr '/' '\\')' '$DESTINAZIONE\\runs\\' -Force" \
      >/dev/null 2>&1
  done

  # 3. Il progetto senza i dati di terze parti e senza cio' che non esce di qui.
  copia "$QUI" 'progetto' \
    '/XD runs outputs data PrisonerStreamlit node_modules __pycache__ .venv target'

  dillo "passata finita in $(( $(date +%s) - inizio ))s"
}

if [ "$UNA_VOLTA" = "--una-volta" ]; then
  passata
  exit 0
fi

dillo "copia di sicurezza avviata verso $DESTINAZIONE, una passata ogni ${OGNI}s"
while true; do
  passata
  # Si ferma da sola quando la campagna e' finita: un'ultima passata e via.
  if grep -qE "BASE PULITA (COMPLETA|FERMATA)" runs/base_pulita.log 2>/dev/null; then
    dillo "campagna conclusa: ultima passata e poi mi fermo."
    passata
    dillo "copia di sicurezza conclusa."
    exit 0
  fi
  sleep "$OGNI"
done
