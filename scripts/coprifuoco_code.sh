#!/usr/bin/env bash
# Coprifuoco: a un'ora data nessuna run nuova parte, quelle in volo finiscono.
#
# **Come.** Ogni lavoratore della coda legge il suo pezzo di piano
# (`<piano>.1`, `.2`, ...) una riga alla volta. Svuotare quel file all'ora del
# taglio fa trovare al lavoratore la fine del piano alla prossima lettura: la
# run in corso arriva in fondo, passa il cancello di qualita', e poi il
# lavoratore si ferma. Il piano intero (`<piano>`) non si tocca: la coda si
# riprende con `scripts/riprendi_code_paper.sh`, che salta le run gia' fatte.
#
# **Il taglio e' per coda**, perche' le run durano diverso: gpt-oss ~1h25,
# Qwen ~1h50 (misurato il 01/10). Taglio = ora di fine meno il margine.
#
# Limite noto: una run bocciata dal cancello viene ritentata subito dentro la
# coda, anche dopo il taglio. Raro; il controllo finale lo segnala.
#
# Uso:
#   nohup bash scripts/coprifuoco_code.sh 23:30 > runs/coprifuoco.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
FINE=${1:?serve l ora di fine, es. 23:30}
# etichetta  margine_minuti
CODE="paper_oss 100
paper_qwen 120"

secondi_a() { echo $(( $(date -d "$1" +%s) - $(date +%s) )); }
dillo() { echo "[coprifuoco] $(date -Is) $*"; }

fine_s=$(date -d "$FINE" +%s)
while read -r etichetta margine; do
  taglio=$(date -d "@$((fine_s - margine * 60))" +%H:%M)
  dillo "$etichetta: nessuna run nuova dopo le $taglio"
done <<< "$CODE"

# Un taglio alla volta, dal piu' vicino.
echo "$CODE" | while read -r etichetta margine; do echo "$((fine_s - margine * 60)) $etichetta"; done \
  | sort -n | while read -r t etichetta; do
  attesa=$(( t - $(date +%s) ))
  [ "$attesa" -gt 0 ] && sleep "$attesa"
  for f in runs/piano_$etichetta.txt.[0-9]*; do
    [ -f "$f" ] && : > "$f"
  done
  msg="COPRIFUOCO: niente run nuove, quelle in corso finiscono (riprendere con scripts/riprendi_code_paper.sh)"
  echo "[$etichetta] $(date -Is) $msg" >> "runs/$etichetta.log"
  dillo "$etichetta: piano dei lavoratori svuotato"
done

# All'ora di fine: chi gira ancora?
attesa=$(secondi_a "$FINE"); [ "$attesa" -gt 0 ] && sleep "$attesa"
vive=$(powershell.exe -NoProfile -NonInteractive -Command \
  "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*run_governor_experiment*' }).Count" \
  2>/dev/null | tr -d '\r')
dillo "alle $FINE simulazioni ancora in corso: ${vive:-?}"
