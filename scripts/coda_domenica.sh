#!/usr/bin/env bash
# Domenica 13 settembre: aggancia la terza esecuzione dei mondi alla coda delle
# prove, senza che le due si contendano la macchina.
#
# **Perche' una terza esecuzione.** Con due esecuzioni per seme sappiamo quanto
# vale lo scarto fra run identiche (da 10 a 427 coloni), ma non sappiamo se un
# cambio di segno sulla popolazione sia rumore o una direzione vera: due punti
# non distinguono le due cose. Il terzo punto sullo stesso seme dice se il
# comportamento e' riproducibile o se cambia a ogni giro. Le due grandezze che
# gia' tengono 5 mondi su 5 (celle occupate e morti per nascita) ne escono
# confermate o smentite, e questo e' il vero motivo per spendere le ore.
#
# Le prove NON vengono rilanciate qui: stanno gia' girando. Questo script
# aspetta che finiscano e poi passa il testimone a coda_weekend.sh, che
# riprende rep3 dal seme mancante (ice_rich ha gia' il seme 3).
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_domenica.sh \
#         >> runs/weekend/coda_domenica.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
COPRIFUOCO=${COPRIFUOCO:-23:45}

dillo() { echo "[domenica] $(date -Is) $*"; }

# Come nelle altre code: `pgrep` non vede i processi Windows, e il filtro sul
# nome dell'eseguibile impedisce alla ricerca di trovare se stessa.
#
# **E i processi che non sanno morire vanno esclusi a mano.** Una run che ha
# usato il Mars Climate Database resta viva dopo aver finito, bloccata nel
# distacco delle DLL, e non la si puo' terminare (vedi `_esci` in
# scripts/run_governor_experiment.py, che corregge il difetto per le run
# future). Senza questo elenco la coda aspetta per sempre un lavoro gia' finito.
BLOCCATI=${BLOCCATI:-$(grep -v '^#' runs/.pid_bloccati 2>/dev/null | grep -E '^[0-9]+$' | paste -sd, -)}

gira() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "\$relitti = @($BLOCCATI); if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq '$1' -and \$_.CommandLine -like '*$2*' -and \$relitti -notcontains \$_.ProcessId }) { 'si' } else { 'no' }" \
    2>/dev/null | tr -d '\r' | grep -q '^si$'
}

dillo "aspetto che le prove finiscano"
while gira python.exe run_governor_experiment || gira bash.exe coda_prove.sh; do
  sleep 300
done
dillo "prove finite: parte la terza esecuzione dei mondi (coprifuoco $COPRIFUOCO)"

COPRIFUOCO="$COPRIFUOCO" bash scripts/coda_weekend.sh
dillo "FINE"
