# -*- coding: utf-8 -*-
"""I quattro prompt del governatore, dallo stesso identico quadro di colonia.

**A che serve.** La campagna sui gradini di contesto toglie informazione al
prompt del governatore un pezzo alla volta, e il risultato e' che la prudenza
resta anche quando nel testo non compare piu' ne' la parola Marte ne' la parola
colonia. Per mostrarlo bisogna far vedere i prompt, non descriverli: questo
script li produce tutti e quattro a partire dallo STESSO quadro, cosi' che
l'unica differenza visibile sia il contesto e non lo stato del mondo.

**Perche' si simula invece di leggere un registro.** Il registro conserva
l'impronta digitale del quadro, non il quadro: serve a verificare che due run
abbiano visto lo stesso stato, non a ricostruire il testo. Il prompt si
rigenera percio' da una colonia vera, fatta girare qui per un centinaio di
passi con la stessa configurazione delle campagne.

Uso:
    python scripts/genera_prompt_esempio.py --passi 120 --seme 3 \
        --out docs/nota_professore/prompt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.parity_harness import realistic_config  # noqa: E402
from src.governors.context import LIVELLI  # noqa: E402
from src.governors.llm_arm import build_governor_prompt  # noqa: E402
from src.governors.observation import build_picture  # noqa: E402
from src.governors.policy import Bounds  # noqa: E402
from src.simulation.agent_coupled_runner import AgentCoupledRunner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passi", type=int, default=120)
    ap.add_argument("--agenti", type=int, default=300)
    ap.add_argument("--seme", type=int, default=3)
    ap.add_argument("--dotazione", type=float, default=0.6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    config = realistic_config(args.agenti, args.passi, args.seme,
                              dotazione=args.dotazione)
    runner = AgentCoupledRunner(config)
    runner.run()

    quadro = build_picture(
        args.passi,
        runner._last_metrics,
        runner.core.cells,
        runner.core.agents,
        runner.core.agents.alive_rows(),
    )
    limiti = Bounds(weight_min=0.5, weight_max=3.0)

    args.out.mkdir(parents=True, exist_ok=True)
    prompt: dict[str, list[str]] = {}
    for livello in LIVELLI:
        testo = build_governor_prompt(quadro, limiti, livello)
        prompt[livello] = testo.splitlines()
        f = args.out / f"prompt_{livello}.txt"
        f.write_text(testo, encoding="utf-8")
        print(f"{livello:12s} {len(testo):6d} caratteri, "
              f"{len(testo.splitlines()):4d} righe  -> {f}")

    # Gli ESTRATTI da mettere nella nota. Si ritagliano qui e non nel documento
    # LaTeX per due ragioni: il prompt intero e' lungo trecento righe e nella
    # nota non ci sta, e il testo completo contiene un trattato lungo (U+2014)
    # che `listings` non sa comporre. Gli estratti sono di sola ASCII.
    def ritaglia(nome: str, righe: list[str]) -> None:
        f = args.out / f"{nome}.txt"
        f.write_text("\n".join(righe).rstrip() + "\n", encoding="utf-8", errors="strict")
        print(f"  estratto {nome}: {len(righe)} righe")

    # L'apertura: e' li' che i quattro gradini differiscono.
    fine_apertura = {"completo": 7, "senza_aiuti": 3, "nomi_veri": 1, "cieco": 1}
    for livello, fino_a in fine_apertura.items():
        ritaglia(f"apertura_{livello}", prompt[livello][:fino_a])

    # Il vocabolario: identico nella sostanza, rinominato nel gradino cieco.
    for livello in ("nomi_veri", "cieco"):
        righe = prompt[livello]
        inizio = next(i for i, r in enumerate(righe) if r.startswith("Indicatori ("))
        fine = next(i for i, r in enumerate(righe[inizio:], inizio)
                    if r.startswith("Leve pesabili:")) + 1
        ritaglia(f"vocabolario_{livello}", righe[inizio:fine])


if __name__ == "__main__":
    main()
