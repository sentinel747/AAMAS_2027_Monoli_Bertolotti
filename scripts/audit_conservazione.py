# -*- coding: utf-8 -*-
"""Bilancio di massa sistematico: chi crea e chi distrugge, risorsa per risorsa.

**Perche' esiste.** Due difetti gravi di questo simulatore non si sono visti
guardando gli esiti — la colonia sopravviveva, le metriche erano plausibili, i
test passavano. Il primo era un'unita' di misura sbagliata (supporto estensivo
letto come dose pro capite); il secondo una legge di conservazione violata (il
rifornimento d'acqua aggiungeva all'inventario senza sottrarre da nulla, e la
colonia accumulava 199.774 unita' d'acqua in ottocento passi). Il secondo e'
stato trovato facendo a mano il bilancio di massa di UNA risorsa. Questo
strumento lo fa di tutte, e lo fa attribuendo ogni variazione a chi l'ha
prodotta.

**Come funziona.** Lo stock di una risorsa e' la somma di cio' che sta nelle
celle, negli inventari dei vivi e nei serbatoi ambientali (ghiaccio di
superficie, acqua liquida, biomassa). Ad ogni passo:

    delta_totale = somma dei delta attribuiti + RESIDUO

Le fasi del passo vengono avvolte una per una — ogni singola azione per tipo,
i vitali, la biologia, gli eventi, la redistribuzione — e cio' che avanza
finisce nel residuo. **Il residuo e' la garanzia di copertura**: non serve
sapere in anticipo dove sono i punti di mutazione, perche' quelli non avvolti
si presentano da soli come massa non attribuita.

**Come si legge.** Una riga con flusso netto positivo e' una SORGENTE, una con
flusso negativo un POZZO. Nessuno dei due e' un difetto in se': una serra
produce cibo, un colono lo mangia. Il difetto e' la sorgente non pagata —
massa che compare senza che nulla diminuisca — e si riconosce perche' l'azione
che la produce e' un TRASFERIMENTO dichiarato (raccogliere, rifornirsi,
attingere) e non una produzione. Per questo la tabella per tipo d'azione e'
separata da quella per fase: un'azione di trasporto che non chiude a zero e'
un difetto per costruzione.

Uso:
    python scripts/audit_conservazione.py [--agenti 120] [--passi 200] [--seme 0]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

sys.path.insert(0, ".")

import numpy as np

from src.core import constants as C

#: Serbatoi ambientali: non sono in `cell_res` ma partecipano al bilancio,
#: perche' raccogliere ghiaccio sposta massa da qui a un inventario. Ometterli
#: farebbe apparire ogni estrazione come una creazione dal nulla.
SERBATOI = {
    "water_ice": "ice",
    "liquid_water": "water",
    "vegetation": "biomass",
}

#: Le risorse che partecipano al ciclo dell'acqua sono fungibili fra loro
#: (ghiaccio fuso, acqua ricongelata): il bilancio ha senso sulla somma.
AGGREGATI = {
    "ACQUA (tutte le forme)": ("ice", "water"),
}


def stock(state) -> dict[str, float]:
    """Massa totale per risorsa: celle + inventari dei vivi + serbatoi."""
    cells, agents = state.cells, state.agents
    vivi = agents.alive_rows()
    tot = {}
    per_cella = cells.cell_res.sum(axis=(0, 1))
    per_inv = agents.inv[vivi].sum(axis=0) if vivi.size else np.zeros(C.NR)
    for nome, idx in C.R.items():
        tot[nome] = float(per_cella[idx]) + float(per_inv[idx])
    for attr, risorsa in SERBATOI.items():
        tot[risorsa] = tot.get(risorsa, 0.0) + float(getattr(cells, attr).sum())
    return tot


def _delta(prima: dict, dopo: dict) -> dict[str, float]:
    return {k: dopo[k] - prima[k] for k in dopo}


class Bilancio:
    """Registro delle variazioni di massa, per causa.

    Non e' un contatore di eventi ma di MASSA: due azioni che si compensano
    lasciano zero, ed e' esattamente cio' che un trasferimento deve fare.
    """

    def __init__(self, state):
        self.state = state
        self.per_causa: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.conteggi: dict[str, int] = defaultdict(int)
        self.serie: list[dict[str, float]] = []
        #: Flussi cumulati a fine di ogni passo. Servono a misurare il margine
        #: su una FINESTRA: calcolarlo sull'intera run lo sporca con la fase di
        #: crescita, in cui la colonia non ha ancora costruito cio' che il
        #: piano prevede e la produzione e' per forza sotto il consumo.
        self.flussi: list[dict[str, dict[str, float]]] = []

    def registra(self, causa: str, prima: dict, dopo: dict) -> None:
        riga = self.per_causa[causa]
        for k, v in _delta(prima, dopo).items():
            if abs(v) > 1e-12:
                riga[k] += v
        self.conteggi[causa] += 1

    def avvolgi(self, causa: str, fn):
        """Misura la massa prima e dopo una chiamata, e la attribuisce."""
        def avvolta(*a, **kw):
            prima = stock(self.state)
            out = fn(*a, **kw)
            self.registra(causa, prima, stock(self.state))
            return out
        return avvolta


def esegui(agenti: int, passi: int, seme: int, dotazione: float = 1.0, redistribuzione: bool = False) -> Bilancio:
    """Esegue una run avvolgendo ogni fase, e restituisce il registro.

    **`dotazione` serve a far morire qualcuno (2026-08-28).** Fino a quel
    giorno il bilancio era stato eseguito soltanto su scenari che non uccidono
    nessuno, e il percorso del decesso non era quindi MAI stato messo alla
    prova: cancellava l'inventario del morto, e la massa spariva. Il residuo
    garantisce la copertura di cio' che accade nella run, non di cio' che la
    run non fa succedere — quindi la scelta dello scenario e' parte
    dell'esperimento, non un dettaglio del comando. Con `--dotazione 0.25` la
    colonia perde coloni e il travaso viene verificato davvero.
    """
    from scripts.parity_harness import realistic_config
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    opzioni = {} if dotazione == 1.0 else {"dotazione": dotazione}
    config = realistic_config(agenti, passi, seme, **opzioni)
    # La fase e' avvolta dal bilancio, ma con la configurazione predefinita
    # e' un'operazione nulla: senza accenderla il percorso non viene provato.
    config.setdefault("redistribution", {})["enabled"] = bool(redistribuzione)
    runner = AgentCoupledRunner(config)
    stato = runner.core

    import src.core.kernel as kernel
    from src.simulation.extreme_events import ExtremeEventEngine

    bil = Bilancio(stato)

    originali = {
        "execute_action": kernel.execute_action,
        "tick_vitals": kernel.tick_vitals,
        "update_cells": kernel.update_cells,
        "redistribute": kernel.redistribute,
        "advance": ExtremeEventEngine.advance,
    }

    def azione_avvolta(av, views, wv, request):
        prima = stock(stato)
        out = originali["execute_action"](av, views, wv, request)
        # Solo le azioni ACCETTATE muovono massa; attribuire anche i rifiuti
        # gonfierebbe i conteggi senza cambiare i totali, e renderebbe
        # illeggibile la colonna "per azione".
        if getattr(out, "accepted", False):
            bil.registra(f"azione:{request.action.value}", prima, stock(stato))
        return out

    kernel.execute_action = azione_avvolta
    kernel.tick_vitals = bil.avvolgi("fase:vitali", originali["tick_vitals"])
    kernel.update_cells = bil.avvolgi("fase:biologia", originali["update_cells"])
    kernel.redistribute = bil.avvolgi("fase:redistribuzione", originali["redistribute"])
    ExtremeEventEngine.advance = bil.avvolgi("fase:eventi", originali["advance"])

    # **Il residuo copre l'intervallo fra due passi, non il solo passo.** Nella
    # prima stesura misurava soltanto dentro `kernel.step`, e le nascite —
    # che avvengono nel runner, e portano con se' un inventario iniziale —
    # sparivano dal bilancio: 42 kit medici comparivano dal nulla senza che
    # nessuna riga li rivendicasse. Confrontare con lo stock di FINE passo
    # precedente rende impossibile che una mutazione sfugga, ovunque sia.
    passo_originale = kernel.step
    ultimo = {"stock": None, "attribuito": {}}

    def _somma_attribuita() -> dict:
        somma = defaultdict(float)
        for causa, riga in bil.per_causa.items():
            for k, v in riga.items():
                somma[k] += v
        return somma

    def passo_avvolto(state, step_index, dt_days, config_, rng):
        prima = stock(state)
        if ultimo["stock"] is not None:
            fuori = {k: prima[k] - ultimo["stock"][k] for k in prima}
            riga_f = bil.per_causa["fuori dal passo (nascite, runner)"]
            for k, v in fuori.items():
                if abs(v) > 1e-9:
                    riga_f[k] += v
        att_prima = _somma_attribuita()
        out = passo_originale(state, step_index, dt_days, config_, rng)
        dopo = stock(state)
        att_dopo = _somma_attribuita()
        residuo = {
            k: (dopo[k] - prima[k]) - (att_dopo[k] - att_prima.get(k, 0.0))
            for k in dopo
        }
        riga_res = bil.per_causa["RESIDUO (non attribuito)"]
        for k, v in residuo.items():
            if abs(v) > 1e-9:
                riga_res[k] += v
        ultimo["stock"] = dopo
        bil.serie.append(dict(dopo))
        bil.flussi.append({c: dict(v) for c, v in bil.per_causa.items()})
        return out

    import src.simulation.agent_coupled_runner as acr
    acr._core_step = passo_avvolto

    try:
        runner.run(days=passi, output_dir=None)
    finally:
        kernel.execute_action = originali["execute_action"]
        kernel.tick_vitals = originali["tick_vitals"]
        kernel.update_cells = originali["update_cells"]
        kernel.redistribute = originali["redistribute"]
        ExtremeEventEngine.advance = originali["advance"]
        acr._core_step = passo_originale

    bil.popolazione = int(stato.agents.alive_rows().size)
    bil.morti = len(runner.dead_agents)
    return bil


def rapporto(bil: Bilancio, soglia: float = 1e-6) -> None:
    """Stampa il bilancio: per causa, e l'andamento degli stock nel tempo."""
    risorse = [r for r in bil.serie[-1] if any(
        abs(v.get(r, 0.0)) > soglia for v in bil.per_causa.values()
    )] if bil.serie else []

    print("\n" + "=" * 78)
    print("FLUSSO NETTO PER CAUSA  (positivo = sorgente, negativo = pozzo)")
    print("=" * 78)
    cause = sorted(bil.per_causa, key=lambda c: (not c.startswith("azione:"), c))
    for causa in cause:
        riga = {k: v for k, v in bil.per_causa[causa].items() if abs(v) > soglia}
        if not riga:
            continue
        n = bil.conteggi.get(causa, 0)
        etichetta = f"{causa} ({n} volte)" if n else causa
        voci = "  ".join(f"{k}{v:+.1f}" for k, v in sorted(
            riga.items(), key=lambda kv: -abs(kv[1])))
        print(f"  {etichetta:<46s} {voci}")

    print("\n" + "=" * 78)
    print("ANDAMENTO DELLO STOCK  (crescita illimitata = sospetto)")
    print("=" * 78)
    tappe = [0, len(bil.serie) // 4, len(bil.serie) // 2, len(bil.serie) - 1]
    print(f"  {'risorsa':<24s}" + "".join(f"{'passo '+str(t+1):>13s}" for t in tappe))
    for r in sorted(risorse):
        valori = "".join(f"{bil.serie[t].get(r, 0.0):13.1f}" for t in tappe)
        print(f"  {r:<24s}{valori}")
    for nome, membri in AGGREGATI.items():
        valori = "".join(
            f"{sum(bil.serie[t].get(m, 0.0) for m in membri):13.1f}" for t in tappe)
        print(f"  {nome:<24s}{valori}")

    print("\n" + "=" * 78)
    print("PRODUZIONE NETTA / CONSUMO  (ultimo quarto della run)")
    print("=" * 78)
    print("  NETTA del tetto di magazzino: il surplus oltre la capienza viene")
    print("  scartato, quindi a magazzino pieno la produzione appare pari al")
    print("  consumo anche quando gli impianti potrebbero fare molto di piu'.")
    print("  Si legge INSIEME alla giacenza qui sopra:")
    print("    ~1,0 e giacenza stabile in alto  -> strozzata dalla capienza, sana;")
    print("    <1,0 e giacenza che scende       -> scarsita' vera, prima o poi morde.")
    n = len(bil.flussi)
    da, a = max(0, n - n // 4 - 1), n - 1
    if a > da:
        def finestra(causa: str, res: str) -> float:
            return (bil.flussi[a].get(causa, {}).get(res, 0.0)
                    - bil.flussi[da].get(causa, {}).get(res, 0.0))
        print(f"\n  {'risorsa':<14s}{'prod. netta':>12s}{'consumo':>12s}"
              f"{'rapporto':>10s}{'giacenza +/-':>14s}")
        for res in ("food", "water", "oxygen"):
            prod = finestra("fase:biologia", res)
            cons = -finestra("fase:vitali", res)
            marg = prod / cons if cons > 1e-9 else float("inf")
            scorta = bil.serie[a].get(res, 0.0) - bil.serie[da].get(res, 0.0)
            print(f"  {res:<14s}{prod:12.1f}{cons:12.1f}{marg:10.2f}{scorta:14.1f}")

    print(f"\n  popolazione finale {bil.popolazione}, morti {bil.morti}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agenti", type=int, default=120)
    p.add_argument("--passi", type=int, default=200)
    p.add_argument("--seme", type=int, default=0)
    p.add_argument("--dotazione", type=float, default=1.0,
                   help="fattore sulla dotazione iniziale; sotto 1 la colonia perde "
                        "coloni, ed e' l'unico modo di provare il percorso del decesso")
    p.add_argument("--redistribuzione", action="store_true",
                   help="accende deposito, prelievo e flusso fra celle: la fase"
                        " che la configurazione predefinita tiene spenta e che il"
                        " bilancio non ha quindi mai attraversato")
    a = p.parse_args()
    print(f"Bilancio di massa: {a.agenti} coloni, {a.passi} passi, seme {a.seme}, "
          f"dotazione x{a.dotazione:g}, "
          f"redistribuzione {'accesa' if a.redistribuzione else 'spenta'}.")
    rapporto(esegui(a.agenti, a.passi, a.seme, a.dotazione, a.redistribuzione))


if __name__ == "__main__":
    main()
