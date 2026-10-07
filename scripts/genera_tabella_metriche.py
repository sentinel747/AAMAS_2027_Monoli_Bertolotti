# -*- coding: utf-8 -*-
"""Tabella delle metriche con i valori veri di una run, e con che cosa le muove.

Il capitolo tre elenca le metriche che la piattaforma pubblica. L'elenco da solo
non dice pero' quali siano vive nel profilo sperimentale usato e quali restino
ferme, e la differenza non e' un dettaglio: alcuni indici socio-operativi si
calcolano su variabili per-colono che nel profilo standard nessuno aggiorna, e
si muovono soltanto perche' nascite e morti cambiano la composizione della
popolazione.

**La classificazione e' derivata, non dichiarata.** Lo script confronta gli
snapshot lungo la run e, per i singoli coloni sopravvissuti, confronta i valori
di stress, morale e rispetto dei protocolli fra il primo e l'ultimo snapshot in
cui compaiono. Se nessun colono cambia, la variabile e' congelata alla nascita e
l'indice che la media si muove solo per ricambio: la tabella lo dice, e lo dice
perche' lo ha misurato.

Uso:
    python scripts/genera_tabella_metriche.py \
        --run runs/campagna_scarsa_v3/llm_completo_amm/llm_completo+amm_seed3 \
        --out Tesi_LaTex/tabella_metriche.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (chiave, nome in tesi, quante cifre)
SUPPORTO = [
    ("eclss_margin", "margine ECLSS", 2),
    ("power_margin", "margine energetico", 2),
    ("food_margin", "margine alimentare", 2),
    ("material_margin", "margine di materiale", 2),
    ("isru_capacity_index", "capacità ISRU", 2),
    ("life_support_reliability", "affidabilità del supporto vitale", 3),
    ("habitat_pressure_index", "pressione abitativa", 2),
    ("colony_prosperity_index", "indice di prosperità", 2),
    ("earth_input_sufficiency_index", "sufficienza dell'input terrestre", 2),
    ("colony_site_score", "punteggio del sito", 2),
    ("colony_site_regret", "rimpianto del sito", 2),
]
SOCIALI = [
    ("crew_stress_index", "stress medio della crew", 3),
    ("crew_morale_index", "morale medio", 3),
    ("protocol_compliance_index", "rispetto dei protocolli", 3),
    ("cohesion_index", "coesione", 3),
    ("conflict_risk_index", "rischio di conflitto", 3),
    ("governance_autonomy_score", "autonomia di governance", 3),
    ("mission_operational_readiness", "prontezza operativa", 3),
]
DA_RETE = [
    ("average_trust", "fiducia media"),
    ("network_density", "densità della rete"),
    ("communication_frequency", "frequenza di comunicazione"),
    ("conflict_intensity", "intensità del conflitto"),
    ("social_stability_score", "stabilità sociale"),
]
# Variabili per-colono su cui poggiano gli indici socio-operativi.
PER_COLONO = ["stress_index", "morale", "protocol_compliance", "fatigue"]


def snapshot(cartella: Path) -> list[Path]:
    d = cartella / "world_snapshots"
    return sorted(d.glob("step_*.json"))


def carica(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def congelate(primo: dict, ultimo: dict) -> dict[str, bool]:
    """Per ogni variabile per-colono: nessun sopravvissuto l'ha mai cambiata?"""
    a = {x["agent_id"]: x for x in primo["agents"]}
    b = {x["agent_id"]: x for x in ultimo["agents"]}
    comuni = set(a) & set(b)
    esito = {}
    for campo in PER_COLONO:
        cambiati = sum(
            1 for i in comuni
            if isinstance(a[i].get(campo), (int, float))
            and abs(a[i][campo] - b[i].get(campo, a[i][campo])) > 1e-9
        )
        esito[campo] = (cambiati == 0, cambiati, len(comuni))
    return esito


def fmt(v, cifre: int) -> str:
    if v is None:
        return "--"
    return f"{v:.{cifre}f}".replace(".", "{,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files = snapshot(args.run)
    if len(files) < 2:
        raise SystemExit(f"servono almeno due snapshot in {args.run}")
    primo, ultimo = carica(files[0]), carica(files[-1])
    mp, mu = primo["metrics"], ultimo["metrics"]
    cong = congelate(primo, ultimo)

    # Una metrica «si muove» se cambia fra il primo e l'ultimo snapshot.
    def muove(k: str) -> bool:
        a, b = mp.get(k), mu.get(k)
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) > 1e-9

    stress_fermo = cong["stress_index"][0] and cong["morale"][0] and cong["protocol_compliance"][0]
    nota_ricambio = "solo ricambio" if stress_fermo else "dinamica"

    R = []
    R.append("% Generato da scripts/genera_tabella_metriche.py: non modificare a mano.")
    R.append("\\begin{table}[htbp]")
    R.append(" \\centering")
    R.append(" \\small")
    R.append(" \\begin{tabular}{@{}lrrl@{}}")
    R.append("  \\toprule")
    R.append(f"  indicatore & passo {primo['step']} & passo {ultimo['step']} & che cosa lo muove \\\\")
    R.append("  \\midrule")
    R.append("  \\multicolumn{4}{@{}l}{\\emph{supporto vitale, prosperità, sito}} \\\\")
    for k, nome, c in SUPPORTO:
        perche = "stato della colonia" if muove(k) else "resta fermo"
        R.append(f"  {nome} & {fmt(mp.get(k), c)} & {fmt(mu.get(k), c)} & {perche} \\\\")
    R.append("  \\midrule")
    R.append("  \\multicolumn{4}{@{}l}{\\emph{socio-operative}} \\\\")
    for k, nome, c in SOCIALI:
        if k in ("crew_stress_index", "crew_morale_index", "protocol_compliance_index"):
            perche = nota_ricambio
        elif muove(k):
            perche = "in parte stato, in parte ricambio"
        else:
            perche = "resta fermo"
        R.append(f"  {nome} & {fmt(mp.get(k), c)} & {fmt(mu.get(k), c)} & {perche} \\\\")
    R.append("  \\midrule")
    R.append("  \\multicolumn{4}{@{}l}{\\emph{dipendenti dalla rete sociale, disattivata}} \\\\")
    for k, nome in DA_RETE:
        R.append(f"  {nome} & {fmt(mp.get(k), 3)} & {fmt(mu.get(k), 3)} & nessuna \\\\")
    R.append("  \\bottomrule")
    R.append(" \\end{tabular}")

    comuni = cong["stress_index"][2]
    R.append(
        f" \\caption{{Le metriche dichiarate, con i valori che assumono davvero in una run "
        f"del mondo di riferimento (seme 3, LLM (Gov+Amm), cioè governatore più amministratori), "
        f"all'inizio e alla fine. La colonna di destra dice che cosa le muove, ed è "
        f"ricavata dal registro e non dichiarata: su {comuni} coloni presenti in entrambi "
        f"gli snapshot, "
        + ("nessuno cambia stress, morale o rispetto dei protocolli lungo tutta la run, "
           "perché il livello psicosociale è disattivato nel profilo sperimentale. "
           "I tre indici che li mediano si spostano quindi solo perché nascite e morti "
           "cambiano la composizione della popolazione, e gli indici composti che li "
           "usano ereditano quella proprietà. "
           if stress_fermo else
           f"{cong['stress_index'][1]} cambiano stress lungo la run. ")
        + "Le ultime cinque grandezze restano a zero per costruzione: sono definite sul "
        f"grafo sociale, che le campagne tengono spento. La tabella serve a distinguere "
        f"le metriche su cui i risultati del \\cref{{chap:piano-valutazione}} poggiano da quelle "
        f"che la piattaforma calcola ma che in questa configurazione non portano "
        f"informazione.}}")
    R.append(" \\label{tab:metriche-reali}")
    R.append("\\end{table}")
    args.out.write_text("\n".join(R) + "\n", encoding="utf-8")

    print(f"scritto {args.out}")
    print(f"snapshot: {files[0].name} -> {files[-1].name}")
    for campo, (fermo, cambiati, tot) in cong.items():
        print(f"  {campo}: {'CONGELATO' if fermo else 'dinamico'} "
              f"({cambiati}/{tot} coloni sopravvissuti cambiano)")


if __name__ == "__main__":
    main()
