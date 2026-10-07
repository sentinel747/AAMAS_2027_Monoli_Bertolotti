#!/usr/bin/env bash
# Guardia delle code di controllo e di confronto (2026-09-29).
#
# Ogni due minuti, per ogni run in corso (cartella senza results.json):
#   - se il governatore ha perso una tornata ("missed": true), oppure
#   - se una decisione System 1 e' finita in ripiego per un errore del fornitore
#     (fallback_reason http_status_* o circuit_open; NON selected_fallback, che e'
#     il modello che sceglie di aspettare),
# ferma il processo, cancella la cartella (la coda la rifa' al rilancio) e lo
# scrive nel log. La tesi ha zero tornate perse: una run che ne perde anche una
# non ripete il protocollo.
#
# Segnala anche la RAM libera sotto 800 MB. Esce quando non ci sono piu' ne'
# simulazioni ne' motori di coda per due controlli di fila, e stampa quali
# cartelle ha cancellato: sono quelle da rifare rilanciando le code.
#
# Uso:  bash scripts/guardia_code.sh   (in background, accanto alle code)
set -u
cd "$(dirname "$0")/.."
LOG=runs/guardia_code.log
RADICI="runs/confronto_system1/system1/* runs/confronto_system1/tesi/* runs/controllo_copertura_mondi/*/llm_amm_*"
vuoti=0
echo "[guardia] $(date -Is) avvio" >> "$LOG"

ps_conta() {  # numero di processi il cui comando contiene $1
  powershell -NoProfile -c "(Get-CimInstance Win32_Process | ? { \$_.CommandLine -match '$1' -and \$_.Name -ne 'powershell.exe' }).Count" 2>/dev/null | tail -1 | tr -d '\r'
}

while true; do
  for d in $RADICI; do
    [ -d "$d" ] || continue
    [ -f "$d/results.json" ] && continue
    sotto=$(ls -d "$d"/*_seed* 2>/dev/null | head -1)
    [ -n "$sotto" ] || continue
    motivo=""
    # semantic_decisions.jsonl si scrive solo a fine run: durante la run le
    # decisioni System 1 (con il loro fallback_reason) stanno nei registri di
    # governatore e amministratori, scritti tornata per tornata (29/09: un 520
    # di Jev e' passato inosservato perche' si guardava solo il primo file).
    if [ -f "$sotto/governor_decisions.jsonl" ] && grep -q '"missed": true' "$sotto/governor_decisions.jsonl"; then
      motivo="tornata persa"
    elif cat "$sotto/governor_decisions.jsonl" "$sotto/administrator_decisions.jsonl" "$sotto/semantic_decisions.jsonl" 2>/dev/null \
         | grep -qE '"fallback_reason": "(http_status_[0-9]+|circuit_open)'; then
      motivo="errore del fornitore (System 1)"
    elif [ -f "$sotto/administrator_decisions.jsonl" ] && grep -qE '"provider_failures": [1-9]' "$sotto/administrator_decisions.jsonl"; then
      motivo="chiamata fallita di un amministratore"
    fi
    if [ -n "$motivo" ]; then
      powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { \$_.CommandLine -match '$d' } | % { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
      sleep 2
      rm -rf "$d"
      echo "[guardia] $(date -Is) $motivo: fermata e cancellata, da rifare: $d" >> "$LOG"
    fi
  done
  ram=$(powershell -NoProfile -c "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" 2>/dev/null | tail -1 | tr -d '\r')
  [ "${ram:-99999}" -lt 800 ] && echo "[guardia] $(date -Is) RAM libera bassa: ${ram} MB" >> "$LOG"
  sim=$(ps_conta 'run_governor_experiment')
  code=$(ps_conta 'xargs')
  if [ "${sim:-0}" = "0" ] && [ "${code:-0}" = "0" ]; then vuoti=$((vuoti + 1)); else vuoti=0; fi
  [ $vuoti -ge 2 ] && break
  sleep 120
done
echo "[guardia] $(date -Is) nessuna run e nessuna coda attiva: fine" >> "$LOG"
grep "da rifare" "$LOG" | sed -n "/$(date +%Y-%m-%d)/p"
