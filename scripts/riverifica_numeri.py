# -*- coding: utf-8 -*-
"""Ricalcola dai registri i numeri portanti della tesi e li confronta con il testo.

Non si fida di cio' che la tesi dice: rilegge le run, rifa' i conti con le stesse
funzioni che producono le figure, e stampa accanto il valore scritto. Una riga
che non torna e' un errore da correggere, non una sfumatura.
"""
from __future__ import annotations

import io
import statistics
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "scripts"))

import figure_confronto as fc  # noqa: E402
from esiti_campagna import leggi  # noqa: E402

esito_globale = 0


def riga(nome: str, atteso, ottenuto, tolleranza: float = 0.0) -> None:
    global esito_globale
    if isinstance(atteso, (int, float)) and isinstance(ottenuto, (int, float)):
        ok = abs(atteso - ottenuto) <= tolleranza
    else:
        ok = str(atteso) == str(ottenuto)
    if not ok:
        esito_globale = 1
    print(f"  [{'ok' if ok else 'NO'}] {nome:<52} in tesi: {atteso}   ricalcolato: {ottenuto}")


def main() -> int:
    print("== campagna sui cinque mondi: effetto sulle celle ==")
    mondi = sorted(fc.scopri_mondi(RADICE / "runs/mondi"),
                   key=lambda m: (m.D if m.D is not None else 9.0, m.nome))
    attesi = {"ricco di ghiaccio": -23.2, "riferimento": -14.5, "frammentato": -52.5,
              "rischio alto": -19.5, "risorse scarse": -4.4}
    tot_neg = tot = 0
    semi_pieni = semi_tot = 0
    for m in mondi:
        eff = m.effetti("celle")
        tutti = [v for s in m.semi for v in eff[s]]
        tot += len(tutti)
        tot_neg += sum(1 for v in tutti if v < 0)
        for s in m.semi:
            semi_tot += 1
            if eff[s] and all(v < 0 for v in eff[s]):
                semi_pieni += 1
        per_seme = [statistics.fmean(eff[s]) for s in m.semi if eff[s]]
        riga(f"effetto medio per seme, {m.etichetta_in('it')}",
             attesi[m.etichetta_in("it")], round(statistics.fmean(per_seme), 1), 0.06)
    riga("esecuzioni sotto la baseline", "50 su 59", f"{tot_neg} su {tot}")
    riga("semi concordi in ogni esecuzione", "18 su 25", f"{semi_pieni} su {semi_tot}")

    print("\n== campagna per famiglie: vivi e celle ==")
    serie = fc.scopri_serie(RADICE / "runs/modelli", "gptoss20b")
    attesi_celle = {"gpt-oss 20B": 92, "gpt-oss 120B": 59, "Qwen3.8 27B": 61, "Qwen3.6 27B": 44}
    attesi_vivi = {"gpt-oss 20B": 1555, "gpt-oss 120B": 1453, "Qwen3.8 27B": 1619, "Qwen3.6 27B": 1696}
    for m in serie:
        et = m.etichetta_in("it")
        if et not in attesi_celle:
            continue
        celle = statistics.fmean(e.celle for s in m.semi for e in m.coppia[s])
        vivi = statistics.fmean(e.vivi for s in m.semi for e in m.coppia[s])
        riga(f"celle, {et}", attesi_celle[et], round(celle), 0.6)
        riga(f"vivi, {et}", attesi_vivi[et], round(vivi), 0.6)
    base = serie[0].baseline
    riga("baseline: celle", 107, round(statistics.fmean(e.celle for e in base.values())), 0.6)
    riga("baseline: vivi", 1562, round(statistics.fmean(e.vivi for e in base.values())), 0.6)

    print("\n== le due campagne a protocollo intero ==")
    for campagna, atteso_gov, atteso_amm, atteso_senzagov in (
            ("runs/base_qwen", -78.4, 46.0, -29.0),
            ("runs/base_pulita", -63.6, 26.6, 36.6)):
        b = leggi(str(RADICE / campagna), "ctrl_none")
        d = leggi(str(RADICE / campagna), "variante_D")
        gov = leggi(str(RADICE / campagna), "solo_governatore")
        amm = leggi(str(RADICE / campagna), "solo_amministratori")
        nome = Path(campagna).name

        def media(a, b_, campo="territorio"):
            semi = sorted(set(a) & set(b_))
            return statistics.fmean(a[s][campo] - b_[s][campo] for s in semi), len(semi)

        m1, n1 = media(d, b)
        m2, n2 = media(gov, d)
        m3, n3 = media(amm, d)
        riga(f"{nome}: governare comprime ({n1} semi)", atteso_gov, round(m1, 1), 0.06)
        riga(f"{nome}: senza amministratori allarga ({n2} semi)", atteso_amm, round(m2, 1), 0.06)
        riga(f"{nome}: senza governatore ({n3} semi)", atteso_senzagov, round(m3, 1), 0.06)

        rep = leggi(str(RADICE / campagna), "variante_D_rep2")
        semi = sorted(set(rep) & set(d))
        rumore = statistics.fmean(abs(rep[s]["territorio"] - d[s]["territorio"]) for s in semi)
        atteso_rumore = 28.0 if nome == "base_qwen" else 16.4
        riga(f"{nome}: rumore sulle celle ({len(semi)} semi)", atteso_rumore, round(rumore, 1), 0.06)

    print("\n== baseline identiche fra le due campagne ==")
    bq = leggi(str(RADICE / "runs/base_qwen"), "ctrl_none")
    bp = leggi(str(RADICE / "runs/base_pulita"), "ctrl_none")
    uguali = all(bq[s]["vivi"] == bp[s]["vivi"] and bq[s]["territorio"] == bp[s]["territorio"]
                 for s in sorted(set(bq) & set(bp)))
    riga("baseline identiche cifra per cifra", True, uguali)
    riga("baseline, vivi per seme", "1746 1598 1788 1749 1621",
         " ".join(str(bp[s]["vivi"]) for s in sorted(bp)))
    riga("baseline, celle per seme", "141 127 158 172 175",
         " ".join(str(bp[s]["territorio"]) for s in sorted(bp)))

    print("\nESITO:", "tutto torna" if esito_globale == 0 else "CI SONO RIGHE CHE NON TORNANO")
    return esito_globale


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
