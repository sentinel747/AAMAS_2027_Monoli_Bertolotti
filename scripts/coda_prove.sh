#!/usr/bin/env bash
# Coda "prove": due interruttori che tutte le campagne tengono spenti, accesi
# una volta sola sugli STESSI semi gia' eseguiti, per misurare quanto pesa
# tenerli spenti invece di limitarsi a dichiararlo.
#
#   A. strato ambientale acceso — clima dal Mars Climate Database e modello
#      planetario dinamico. Nessuna delle 595 run dell'archivio lo ha mai
#      usato: il database e' installato (15 GB, MCD 6.1) e la configurazione lo
#      chiede, ma l'unico codice che lo interroga vive nel `PlanetaryCoupler`,
#      che non viene costruito quando lo strato e' spento. Il mondo gira allora
#      su costanti marziane ferme (media -55 C, opacita' 0,18, stabilita'
#      dell'acqua liquida a zero), con la variazione spaziale per cella e gli
#      eventi estremi, ma senza stagioni ne' ciclo giorno-notte.
#
#   B. redistribuzione automatica di cella accesa — deposito del surplus nel
#      magazzino, prelievo di chi sta sotto soglia, flusso fra celle vicine.
#      Spenta, la scarsita' resta locale, ed e' su quella scarsita' che poggia
#      meta' dei risultati.
#
# Due semi per braccio invece di cinque: qui non si cerca un effetto, si cerca
# se l'interruttore sposta l'ordine di grandezza. Se lo sposta, la campagna va
# ripensata; se non lo sposta, la scelta e' dichiarabile in una riga.
#
# Confronto: runs/campagna_scarsa/ctrl_none e runs/campagna_scarsa_v3/llm_completo_amm,
# stessi semi, stessa configurazione, interruttori spenti.
#
# Uso:  MARSABM_ALLOW_LLM_CALLS=1 nohup bash scripts/coda_prove.sh \
#         >> runs/prove/coda_prove.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
ATTESA=${ATTESA:-"runs/weekend/coda_weekend.log"}
SEMI=${SEMI:-"3 4"}
COMUNI="--steps 1000 --agents 300 --cadence 25 --dotazione 0.6 --log-interval 250 --snapshot-interval 25 --wait 120 --map-profile balanced"
mkdir -p runs/prove

# **`pgrep` non vede i processi Windows**: da Git Bash restituisce sempre
# vuoto, e la prima versione di questa coda e' partita subito, contendendo la
# CPU alla coda del fine settimana. Il controllo passa percio' da PowerShell,
# che i processi Windows li vede. Due simulazioni sulla stessa macchina si
# rallentano a vicenda: una per volta, sempre.
# **E il filtro sul nome dell'eseguibile non e' un di piu'.** La riga di
# comando di questo stesso controllo contiene la parola che cerca, quindi
# senza `Name -eq 'python.exe'` la ricerca trova se stessa e la coda aspetta
# per sempre. E' successo.
# **E i processi che non sanno morire vanno esclusi a mano.** Una run che ha
# usato il Mars Climate Database resta viva dopo aver finito, bloccata nel
# distacco delle DLL, e non la si puo' terminare (vedi `_esci` in
# scripts/run_governor_experiment.py, che corregge il difetto per le run
# future). Senza questo elenco la coda scambia quei relitti per simulazioni in
# corso e aspetta per sempre un lavoro gia' finito. E' successo.
BLOCCATI=${BLOCCATI:-$(grep -v '^#' runs/.pid_bloccati 2>/dev/null | grep -E '^[0-9]+$' | paste -sd, -)}

occupata() {
  powershell.exe -NoProfile -NonInteractive -Command     "\$relitti = @($BLOCCATI); if (Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_governor_experiment*' -and \$relitti -notcontains \$_.ProcessId }) { 'si' } else { 'no' }"     2>/dev/null | tr -d '' | grep -q '^si$'
}

# Lo stesso coprifuoco della coda del fine settimana: la macchina si spegne
# a mezzanotte e una run uccisa a meta' lascia una cartella parziale da
# togliere a mano. Nessuna prova parte se non finisce prima.
COPRIFUOCO=${COPRIFUOCO:-23:45}
STIMA=${STIMA:-60}

minuti_al_coprifuoco() {
  # Un orario gia' passato significa la notte prossima, non quella trascorsa:
  # cosi' COPRIFUOCO=01:00 vuol dire l'una di notte e non l'una di stamattina.
  local fine ora
  fine=$(date -d "today $COPRIFUOCO" +%s)
  ora=$(date +%s)
  if [ "$fine" -le "$ora" ]; then
    fine=$(date -d "tomorrow $COPRIFUOCO" +%s)
  fi
  echo $(( (fine - ora) / 60 ))
}

prova() {
  local nome=$1; shift
  if [ "$(minuti_al_coprifuoco)" -lt "$STIMA" ]; then
    echo "[prove] $(date -Is) COPRIFUOCO: $nome richiede ~$STIMA min e ne restano $(minuti_al_coprifuoco)."
    echo "[prove] $(date -Is) Riprendi con lo stesso comando."
    exit 0
  fi
  echo "[prove] $(date -Is) $nome (restano $(minuti_al_coprifuoco) min)"
  "$@" || echo "[prove] ERRORE $nome"
}

echo "[prove] $(date -Is) attendo che la macchina sia libera"
while occupata; do sleep 300; done
echo "[prove] $(date -Is) macchina libera, partono le prove"

for seme in $SEMI; do
  prova "A/baseline strato ambientale seme $seme" \
    python scripts/run_governor_experiment.py --arms none --seeds "$seme" $COMUNI \
      --strato-ambientale --out runs/prove/ambiente_none

  prova "A/coppia strato ambientale seme $seme" \
    python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
      --administrators --thinking off --seeds "$seme" $COMUNI \
      --strato-ambientale --out runs/prove/ambiente_llm_amm

  prova "B/baseline redistribuzione seme $seme" \
    python scripts/run_governor_experiment.py --arms none --seeds "$seme" $COMUNI \
      --redistribuzione --out runs/prove/redistr_none

  prova "B/coppia redistribuzione seme $seme" \
    python scripts/run_governor_experiment.py --arms llm --provider gpu_farm --model gpt-oss:20b \
      --administrators --thinking off --seeds "$seme" $COMUNI \
      --redistribuzione --out runs/prove/redistr_llm_amm
done
echo "[prove] $(date -Is) PROVE COMPLETE"
