# -*- coding: utf-8 -*-
"""Ogni singola run che entra nei risultati, per mondo, braccio e seme.

**Perche' esiste.** «Quali run stiamo considerando» e' la domanda a cui un
lavoro sperimentale deve saper rispondere in una pagina, e finora la risposta
stava sparsa fra le tabelle di cinque rapporti diversi. Qui c'e' l'elenco
intero: un rigo per esecuzione, con il mondo, il braccio, il seme, l'esito e le
due condizioni di ammissione (tornate del governatore perse, chiamate fallite).

**L'insieme e' dichiarato, non dedotto da un glob.** `runs/` contiene anche
archivi superati --- serie di modelli poi rifatte, campagne con il parser
vecchio --- e pescarli con un glob li farebbe entrare nei risultati senza che
nessuno lo decida. L'elenco delle campagne riportate sta in
`scripts/audit_registri.py` ed e' lo stesso da cui il controllo di ammissione
legge: se le due liste divergessero, ci sarebbero run controllate e non
riportate, o peggio riportate e non controllate.

Uso:
    python scripts/inventario_run.py
    python scripts/inventario_run.py --out docs/benchmarks/inventario.md
    python scripts/inventario_run.py --latex        # per la nota al professore
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from scripts.audit_registri import CAMPAGNE  # noqa: E402

#: Nome leggibile del profilo di mappa. Il mondo di riferimento e' `balanced`:
#: la cartella che lo contiene si chiama pero' `campagna_scarsa`, perche' quando
#: fu creata «scarsa» voleva dire «balanced con dotazione 0,6». Il nome della
#: cartella e' rimasto e trae in inganno: il mondo e' il riferimento, non un
#: mondo a risorse scarse, che e' un profilo diverso (`scarce_resources`).
MONDI = {
    "balanced": "riferimento (balanced)",
    "ice_rich": "ricco di ghiaccio (ice_rich)",
    "fragmented": "frammentato (fragmented)",
    "high_hazard": "rischio alto (high_hazard)",
    "scarce_resources": "risorse scarse (scarce_resources)",
    "mineral_rich": "ricco di minerali (mineral_rich)",
}

#: L'ordine in cui i mondi si presentano: il riferimento per primo, poi per
#: difficolta' crescente.
ORDINE_MONDI = ["balanced", "ice_rich", "fragmented", "high_hazard",
                "scarce_resources", "mineral_rich"]


def _righe(percorso: Path) -> list[dict]:
    fuori = []
    if not percorso.exists():
        return fuori
    for riga in io.open(percorso, encoding="utf-8", errors="replace"):
        riga = riga.strip()
        if riga:
            try:
                fuori.append(json.loads(riga))
            except json.JSONDecodeError:
                continue
    return fuori


def _fallite(registro: Path, seme) -> int | None:
    for cartella in sorted(registro.parent.glob(f"*seed{seme}")):
        f = cartella / "api_usage.json"
        if f.exists():
            try:
                return int(json.loads(f.read_text(encoding="utf-8")).get("failed_calls", 0))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                return None
    return None


def _registri(radice: Path) -> list[Path]:
    """Tutti i `results.jsonl` sotto una radice dichiarata."""
    if (radice / "results.jsonl").exists():
        return [radice / "results.jsonl"]
    return sorted(radice.glob("**/results.jsonl"))


def raccogli() -> list[dict]:
    """Una voce per esecuzione, con tutto cio' che serve a giudicarla."""
    voci = []
    for campagna, radici in CAMPAGNE.items():
        for relativo in radici:
            for registro in _registri(ROOT / relativo):
                gruppo = registro.parent.relative_to(ROOT).as_posix()
                for r in _righe(registro):
                    cfg = r.get("config") or {}
                    api = r.get("api") or {}
                    fallite = api.get("failed_calls")
                    if fallite is None:
                        fallite = _fallite(registro, r.get("seed"))
                    voci.append({
                        "campagna": campagna,
                        "cartella": gruppo,
                        "mondo": cfg.get("map_profile") or "balanced",
                        "dotazione": cfg.get("dotazione"),
                        "braccio": r.get("braccio") or r.get("arm"),
                        "modello": cfg.get("model") or "",
                        "seme": r.get("seed"),
                        "vivi": r.get("population"),
                        "celle": (r.get("espansione") or {}).get("celle_totali"),
                        "morti": r.get("deaths"),
                        "tornate_perse": r.get("governor_misses"),
                        "chiamate_fallite": fallite,
                    })
    return voci


def _etichetta_braccio(voce: dict) -> str:
    """Il braccio con il nuovo vocabolario, e senza confondere i controlli.

    Il registro scrive il braccio in una sintassi compatta (`llm:cieco+amm`,
    `scripted+amm`, `none+amm:llm`) in cui la parola «amm» compare in tre casi
    diversi: amministratori linguistici, amministratori a regole fisse e
    amministratori casuali. Chiamarli tutti LLM (Amm) farebbe passare per
    linguistici due bracci di CONTROLLO, che esistono apposta per non esserlo.
    """
    b = str(voce["braccio"] or "")
    governo, _, amministrazione = b.partition("+")
    gov_llm = governo.startswith("llm")
    livello = governo.split(":", 1)[1] if ":" in governo else ""
    variante = ""
    if "-p" in livello:
        livello, variante = livello.split("-p", 1)

    def _coda() -> str:
        pezzi = []
        if livello and livello != "completo":
            pezzi.append(livello.replace("_", " "))
        if variante:
            pezzi.append(f"prompt {variante}")
        return f" [{', '.join(pezzi)}]" if pezzi else ""

    if governo in ("none", "") and not amministrazione:
        return "baseline"
    if governo == "random":
        return "controllo casuale" + ("+amm" if amministrazione else "")
    if governo == "scripted":
        return "controllo a regole fisse" + ("+amm" if amministrazione else "")
    # `none+amm:llm` e' il decentramento puro: nessun governo, amministratori
    # linguistici. E' la casella che isola il livello locale.
    if governo == "none" and amministrazione.endswith("llm"):
        return "LLM (Amm)"
    if gov_llm and amministrazione:
        return "LLM (Gov+Amm)" + _coda()
    if gov_llm:
        return "LLM (Gov)" + _coda()
    return b


def per_mondo(voci: list[dict]) -> dict:
    """mondo -> braccio -> cartella -> {seme: voce}."""
    fuori: dict = {}
    for v in voci:
        fuori.setdefault(v["mondo"], {}).setdefault(_etichetta_braccio(v), {}) \
            .setdefault(v["cartella"], {})[v["seme"]] = v
    return fuori


def markdown(voci: list[dict]) -> list[str]:
    out = ["# Inventario: ogni run che entra nei risultati\n"]
    out.append(
        "Generato da `scripts/inventario_run.py`. Le campagne sono quelle "
        "dichiarate in `scripts/audit_registri.py`, cioe' le stesse su cui gira "
        "il controllo delle condizioni di ammissione.\n"
    )
    out.append(
        "«Tornate perse» sono le deliberazioni del governatore che non sono "
        "arrivate; «fallite» le chiamate al modello andate in errore. Una run "
        "e' comparabile solo se valgono entrambe zero.\n"
    )
    totale = 0
    mappa = per_mondo(voci)
    for mondo in ORDINE_MONDI:
        if mondo not in mappa:
            continue
        out.append(f"\n## {MONDI.get(mondo, mondo)}\n")
        for braccio in sorted(mappa[mondo]):
            for cartella in sorted(mappa[mondo][braccio]):
                per_seme = mappa[mondo][braccio][cartella]
                semi = sorted(per_seme)
                totale += len(semi)
                modelli = {per_seme[s]["modello"] for s in semi} - {""}
                dotazioni = {per_seme[s]["dotazione"] for s in semi}
                perse = sum(int(per_seme[s]["tornate_perse"] or 0) for s in semi)
                fallite = sum(int(per_seme[s]["chiamate_fallite"] or 0) for s in semi)
                out.append(f"**{braccio}** — `{cartella}`"
                           + (f", modello {'/'.join(sorted(modelli))}" if modelli else "")
                           + (f", dotazione {sorted(dotazioni)[0]}" if len(dotazioni) == 1 else "")
                           + f" — {len(semi)} run"
                           + ("" if perse == 0 and fallite == 0
                              else f" — **{perse} tornate perse, {fallite} chiamate fallite**")
                           + "\n")
                out.append("| seme | vivi | celle | morti |")
                out.append("|---:|---:|---:|---:|")
                for s in semi:
                    v = per_seme[s]
                    out.append(f"| {s} | {v['vivi']} | {v['celle']} | {v['morti']} |")
                out.append("")
    out.append(f"\n**Totale: {totale} esecuzioni.**\n")
    return out


def _tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def latex(voci: list[dict]) -> list[str]:
    """Le stesse righe, per la nota al professore.

    La cartella entra in tabella e non e' un dettaglio da nascondere: senza, due
    righe con lo stesso braccio e lo stesso modello sono indistinguibili, e sono
    proprio quelle che vanno distinte --- una e' la prima esecuzione, l'altra la
    ripetizione, e mediarle o sommarle sarebbero errori diversi.
    """
    out = []
    mappa = per_mondo(voci)
    totale = 0
    for mondo in ORDINE_MONDI:
        if mondo not in mappa:
            continue
        out.append(r"\subsection*{" + _tex(MONDI.get(mondo, mondo)) + "}")
        out.append(r"{\footnotesize")
        out.append(r"\begin{tabular}{@{}p{3.6cm}p{4.1cm}cc p{5.2cm}@{}}")
        out.append(r"\toprule")
        out.append(r"braccio & dove sta & dot. & semi & esiti, vivi/celle \\")
        out.append(r"\midrule")
        for braccio in sorted(mappa[mondo]):
            for cartella in sorted(mappa[mondo][braccio]):
                per_seme = mappa[mondo][braccio][cartella]
                semi = sorted(per_seme)
                totale += len(semi)
                modelli = {per_seme[s]["modello"] for s in semi} - {""}
                dotazioni = sorted({per_seme[s]["dotazione"] for s in semi})
                esiti = ", ".join(f"{per_seme[s]['vivi']}/{per_seme[s]['celle']}" for s in semi)
                nome = braccio + (
                    "\\newline \\textit{" + _tex("/".join(sorted(modelli))) + "}" if modelli else ""
                )
                out.append(
                    "%s & \\path{%s} & %s & %s & %s \\\\" % (
                        _tex(nome).replace(r"\newline", r"\newline"),
                        cartella.replace("runs/", ""),
                        dotazioni[0] if len(dotazioni) == 1 else "misto",
                        ",".join(str(s) for s in semi),
                        esiti,
                    )
                )
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}}")
        out.append("")
    out.append(r"\textbf{Totale: %d esecuzioni.}" % totale)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs/benchmarks/2026-09-14-inventario-run.md")
    ap.add_argument("--latex", type=Path, default=None,
                    help="scrive anche il frammento LaTeX per la nota")
    args = ap.parse_args()

    voci = raccogli()
    testo = "\n".join(markdown(voci)) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(testo, encoding="utf-8")
    print(f"scritto {args.out}  ({len(voci)} esecuzioni)")

    if args.latex:
        args.latex.write_text("\n".join(latex(voci)) + "\n", encoding="utf-8")
        print(f"scritto {args.latex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
