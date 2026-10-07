# -*- coding: utf-8 -*-
"""Le figure del confronto appaiato baseline / governo+amministratori, per configurazione.

**Che cosa disegna.** Per ogni mondo (profilo di mappa, dotazione) il protocollo
esegue la baseline `ctrl_none` e LLM (Gov+Amm) --- governatore piu' amministratori
--- sugli STESSI semi, e ripete LLM (Gov+Amm) sullo stesso seme quando il modello e'
campionato. Le figure mettono queste esecuzioni una accanto all'altra senza
mediare cio' che non va mediato:

1. `confronto_esiti`: per mondo (ordinato per punteggio di difficolta' D) e
   per seme, l'esito della baseline (segno vuoto) e di ogni esecuzione di
   LLM (Gov+Amm) (punti pieni), su vivi e celle occupate a fine run. Le repliche
   dello stesso seme restano visibili come punti distinti: la loro distanza
   e' la varianza del modello, e ogni effetto va letto contro quella.
2. `confronto_traiettorie`: la popolazione passo per passo, baseline
   tratteggiata e LLM (Gov+Amm) continua, e sotto la differenza appaiata
   (LLM (Gov+Amm) - baseline) sullo stesso seme, che mostra QUANDO le due
   traiettorie si separano e se la separazione ha un segno stabile.
3. `effetto_vs_difficolta`: l'effetto per seme (LLM (Gov+Amm) - baseline) contro il
   punteggio D della configurazione, con la fascia della varianza fra
   esecuzioni identiche misurata sulle repliche. E' la figura del paper.

**Che cosa NON fa.** Nessuna media fra semi e' presentata come risultato;
le medie compaiono solo nella tabella di riepilogo, accanto ai segni per seme.
I mondi entrano solo se hanno baseline e LLM (Gov+Amm) sullo stesso seme.

Uso:
    python scripts/figure_confronto.py [--mondi-radice runs/mondi] [--out-tesi Tesi_LaTex/figures]
                                       [--out-paper Paper_LaTex/AAMAS_Format/figures] [--rapporto file.md]
    python scripts/figure_confronto.py --mondo "balanced=runs/campagna_scarsa/ctrl_none:runs/campagna_scarsa_v3/llm_completo_amm,..."
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analisi_decentramento import _cartelle_run, _risultati  # noqa: E402
from scripts.score_difficolta import RAGGIO_DEFAULT, componenti  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Il mondo di riferimento: baseline della campagna del 2 settembre e le tre
#: esecuzioni di LLM (Gov+Amm) sugli stessi semi (campagna v3, 5-6 settembre).
RIFERIMENTO_DEFAULT = (
    "balanced",
    ROOT / "runs/campagna_scarsa/ctrl_none",
    [
        ROOT / "runs/campagna_scarsa_v3/llm_completo_amm",
        ROOT / "runs/campagna_scarsa_v3/llm_completo_amm_rep2",
        ROOT / "runs/campagna_scarsa_v3/llm_completo_amm_rep3",
    ],
)
#: Nei mondi della coda, l'esecuzione da usare e' la `_v2` quando esiste (parser
#: corretto); la prima esecuzione resta come misura del difetto.
COPPIA_PREFERITA = ("llm_completo_amm_v2", "llm_completo_amm")
ETICHETTE = {
    "balanced": "riferimento",
    "scarce_resources": "risorse scarse",
    "high_hazard": "rischio alto",
    "ice_rich": "ricco di ghiaccio",
    "mineral_rich": "ricco di minerali",
    "fragmented": "frammentato",
    # serie per modello (runs/modelli/<slug>) e per livello (runs/livelli/<livello>)
    # I nomi veri sono `gpt-oss:20b` su Ollama e `Qwen/Qwen3.8-27B-FP8` su vLLM:
    # due convenzioni diverse perche' sono due fornitori diversi. In figura
    # contano il modello e la taglia, non la sintassi del fornitore, quindi le
    # etichette sono normalizzate e gli identificatori esatti stanno nel testo.
    "gptoss20b": "gpt-oss 20B", "gptoss120b": "gpt-oss 120B",
    "qwen38_27b": "Qwen3.8 27B", "qwen36_27b": "Qwen3.6 27B",
    "completo": "completo", "senza_aiuti": "senza aiuti", "nomi_veri": "nomi veri", "cieco": "cieco",
    # serie «governo» (runs/governo/<braccio>): accentrato / decentrato / solo locale
    "llm_completo": "LLM (Gov)", "coppia": "LLM (Gov+Amm)", "amm_soli": "LLM (Amm)",
}
ETICHETTE_EN = {
    "balanced": "reference",
    "scarce_resources": "scarce resources",
    "high_hazard": "high hazard",
    "ice_rich": "ice rich",
    "mineral_rich": "mineral rich",
    "fragmented": "fragmented",
    "gptoss20b": "gpt-oss 20B", "gptoss120b": "gpt-oss 120B",
    "qwen38_27b": "Qwen3.8 27B", "qwen36_27b": "Qwen3.6 27B",
    "completo": "full context", "senza_aiuti": "no aids", "nomi_veri": "real names only", "cieco": "blind",
    "llm_completo": "LLM (Gov)", "coppia": "LLM (Gov+Adm)", "amm_soli": "LLM (Adm)",
}
#: L'ordine con cui le serie si presentano (il resto in coda, alfabetico).
ORDINE_SERIE = ("gptoss20b", "gptoss120b", "qwen38_27b", "qwen36_27b", "completo", "senza_aiuti", "nomi_veri", "cieco",
                "llm_completo", "coppia", "amm_soli")
#: Le scritte delle figure nelle due lingue (tesi in italiano, paper in inglese).
TESTI = {
    "it": {"vivi": "coloni vivi", "celle": "celle occupate", "diff": "LLM (Gov+Amm) − baseline",
           "D": "difficoltà $D$ della configurazione", "fascia": "scarto massimo fra esecuzioni identiche (±{f:.0f})",
           "fascia_media": "scarto medio (±{f:.0f})",
           "fascia_nuda": "scarto fra esecuzioni identiche",
           "nota_effetto": "ogni punto è una differenza appaiata: un'esecuzione di LLM (Gov+Amm) "
                           "meno la baseline dello STESSO seme. I mondi sono ordinati per difficoltà.",
           # Le serie per modello, per livello di contesto e per forma di governo non
           # sono mondi: la clausola sull'ordinamento per difficoltà lì è falsa.
           "nota_effetto_bracci": "ogni punto è una differenza appaiata rispetto alla baseline dello "
                                  "STESSO seme; la fascia grigia è lo scarto fra esecuzioni identiche.",
           "seme": "seme", "coppia": "LLM (Gov+Amm)", "baseline": "baseline",
           "vivi_fine": "coloni vivi a fine run", "celle_fine": "celle occupate a fine run",
           "nota_box": "ogni linea grigia unisce le due esecuzioni dello STESSO seme; "
                       "i punti arancioni sono le esecuzioni ripetute di LLM (Gov+Amm)"},
    "en": {"vivi": "colonists alive", "celle": "occupied cells", "diff": "LLM (Gov+Adm) − baseline",
           "D": "difficulty $D$ of the configuration", "fascia": "largest spread between identical runs (±{f:.0f})",
           "fascia_media": "mean spread (±{f:.0f})",
           "fascia_nuda": "spread between identical runs",
           "nota_effetto": "each dot is a paired difference: one run of LLM (Gov+Adm) minus the "
                           "baseline of the SAME seed. Worlds are ordered by difficulty.",
           "nota_effetto_bracci": "each dot is a paired difference against the baseline of the SAME "
                                  "seed; the grey band is the spread between identical runs.",
           "seme": "seed", "coppia": "LLM (Gov+Adm)", "baseline": "baseline",
           "vivi_fine": "colonists alive at the end", "celle_fine": "occupied cells at the end",
           "nota_box": "each grey line joins the two runs of the SAME seed; "
                       "orange dots are the repeated runs of LLM (Gov+Adm)"},
}
#: Le scritte delle slide di laurea: le parole del resto della presentazione
#: («senza governo», «con governo LLM»), etichette corte perche' la figura si
#: legge da lontano, e niente D (la presentazione non lo introduce).
TESTI["pres"] = dict(TESTI["it"], **{
    "diff": "con governo − senza", "seme": "seme", "coppia": "con LLM", "baseline": "senza",
    "fascia_nuda": "rumore fra esecuzioni identiche",
    "nota_effetto": "ogni punto: un'esecuzione con governo meno la colonia senza governo dello stesso seme",
    "nota_box": "ogni linea grigia unisce le due colonie dello stesso seme; "
                "i punti arancioni sono le esecuzioni ripetute con governo",
})
FONDATORI_DEFAULT = 300


@dataclass
class Esito:
    vivi: int
    morti: int
    celle: int
    nascite: int
    cartella: Path | None = None
    D: float | None = None


@dataclass
class Mondo:
    nome: str
    baseline: dict[int, Esito]                 # seme -> esito
    coppia: dict[int, list[Esito]]             # seme -> una voce per esecuzione
    semi: list[int] = field(default_factory=list)

    @property
    def etichetta(self) -> str:
        return ETICHETTE.get(self.nome, self.nome)

    def etichetta_in(self, lingua: str) -> str:
        return (ETICHETTE if lingua in ("it", "pres") else ETICHETTE_EN).get(self.nome, self.nome)

    @property
    def D(self) -> float | None:
        v = [e.D for e in self.baseline.values() if e.D is not None]
        return round(statistics.fmean(v), 3) if v else None

    def effetti(self, campo: str = "vivi") -> dict[int, list[int]]:
        """seme -> [coppia_k - baseline] per ogni esecuzione k di LLM (Gov+Amm)."""
        return {
            s: [getattr(e, campo) - getattr(self.baseline[s], campo) for e in self.coppia[s]]
            for s in self.semi
        }


# --------------------------------------------------------------------------- dati

def _esito(riga: dict, cartella: Path | None, fondatori: int) -> Esito:
    vivi = int(riga.get("population", 0) or 0)
    morti = int(riga.get("deaths", 0) or 0)
    return Esito(
        vivi=vivi,
        morti=morti,
        celle=int((riga.get("espansione") or {}).get("celle_totali", 0) or 0),
        nascite=vivi + morti - int(riga.get("agents", fondatori) or fondatori),
        cartella=cartella,
    )


def _leggi_braccio(braccio: Path, con_D: bool, raggio: int) -> dict[int, Esito]:
    righe = _risultati(braccio)
    cartelle = _cartelle_run(braccio)
    fuori = {}
    for seme, riga in righe.items():
        e = _esito(riga, cartelle.get(seme), FONDATORI_DEFAULT)
        if con_D and e.cartella and (e.cartella / "world_static_base.json").exists():
            base = json.loads((e.cartella / "world_static_base.json").read_text(encoding="utf-8", errors="replace"))
            dot = float((riga.get("config") or {}).get("dotazione", 1.0) or 1.0)
            e.D = componenti(base, dot, raggio)["D"]
        fuori[seme] = e
    return fuori


def carica_mondo(nome: str, baseline: Path, coppie: list[Path], raggio: int = RAGGIO_DEFAULT) -> Mondo:
    """Un mondo con i soli semi che hanno la baseline E almeno una esecuzione governata."""
    base = _leggi_braccio(baseline, con_D=True, raggio=raggio)
    per_seme: dict[int, list[Esito]] = {}
    for c in coppie:
        for seme, e in _leggi_braccio(c, con_D=False, raggio=raggio).items():
            per_seme.setdefault(seme, []).append(e)
    semi = sorted(s for s in base if s in per_seme)
    return Mondo(
        nome=nome,
        baseline={s: base[s] for s in semi},
        coppia={s: per_seme[s] for s in semi},
        semi=semi,
    )


def scopri_mondi(radice: Path, riferimento=RIFERIMENTO_DEFAULT, raggio: int = RAGGIO_DEFAULT) -> list[Mondo]:
    """Il riferimento piu' ogni cartella di `radice` con baseline e coppia; ordinati per D."""
    mondi = []
    nome, base, coppie = riferimento
    if base.exists() and any(c.exists() for c in coppie):
        m = carica_mondo(nome, base, [c for c in coppie if c.exists()], raggio)
        if m.semi:
            mondi.append(m)
    if radice.exists():
        for cartella in sorted(p for p in radice.iterdir() if p.is_dir()):
            base = cartella / "ctrl_none"
            coppia = next((cartella / n for n in COPPIA_PREFERITA if (cartella / n / "results.jsonl").exists()), None)
            if not (base / "results.jsonl").exists() or coppia is None:
                continue
            # Le repliche della stessa coppia (`_rep2`, `_rep3`) entrano con
            # lei, come nel riferimento: sono le esecuzioni identiche su cui si
            # misura lo scarto, e senza di loro un mondo ripetuto
            # riporterebbe solo la prima esecuzione.
            repliche = [coppia] + sorted(
                d for d in cartella.iterdir()
                if d.is_dir() and d.name.startswith(coppia.name + "_rep")
                and (d / "results.jsonl").exists()
            )
            m = carica_mondo(cartella.name, base, repliche, raggio)
            if m.semi:
                mondi.append(m)
    return sorted(mondi, key=lambda m: (m.D if m.D is not None else 9.0, m.nome))


def scopri_serie(radice: Path, nome_riferimento: str, riferimento=RIFERIMENTO_DEFAULT,
                 raggio: int = RAGGIO_DEFAULT) -> list[Mondo]:
    """Una SERIE: stesso mondo (il riferimento), coppie diverse.

    La prima voce e' LLM (Gov+Amm) del riferimento (le tre esecuzioni con gpt-oss:20b
    al livello completo) con il nome `nome_riferimento`; poi ogni sottocartella
    di `radice` che contenga un braccio con results.jsonl, confrontata con la
    STESSA baseline del riferimento. Serve per la campagna dei modelli
    (`runs/modelli/<slug>/llm_completo_amm`) e per quella dei livelli di contesto
    (`runs/livelli/<livello>/llm_<livello>_amm`). Le repliche dello stesso
    braccio (suffisso `_rep2`, `_rep3`) entrano insieme al braccio, come nel
    riferimento: servono a misurare lo scarto fra esecuzioni identiche.
    Ordine: `ORDINE_SERIE`."""
    _, base, coppie = riferimento
    fuori = []
    m = carica_mondo(nome_riferimento, base, [c for c in coppie if c.exists()], raggio)
    if m.semi:
        fuori.append(m)
    if radice.exists():
        for cartella in sorted(p for p in radice.iterdir() if p.is_dir()):
            if (cartella / "results.jsonl").exists():
                bracci = [cartella]                      # il braccio sta direttamente sotto la radice
            else:
                bracci = sorted(b for b in cartella.iterdir() if b.is_dir() and (b / "results.jsonl").exists())
            if not bracci:
                continue
            capo = bracci[0]
            repliche = [b for b in bracci if b.name in (capo.name,) or b.name.startswith(capo.name + "_rep")]
            m = carica_mondo(cartella.name, base, repliche, raggio)
            if m.semi:
                fuori.append(m)
    posto = {n: i for i, n in enumerate(ORDINE_SERIE)}
    return sorted(fuori, key=lambda m: (posto.get(m.nome, len(posto)), m.nome))


def scarto_repliche(mondo: Mondo, campo: str = "vivi") -> float | None:
    """Distanza media fra esecuzioni identiche (stesso seme, stessa coppia): la varianza del modello.

    Media, sui semi con almeno due esecuzioni, dell'intervallo (max - min) fra le
    esecuzioni; None se nessun seme e' stato ripetuto."""
    intervalli = []
    for s in mondo.semi:
        v = [getattr(e, campo) for e in mondo.coppia[s]]
        if len(v) >= 2:
            intervalli.append(max(v) - min(v))
    return round(statistics.fmean(intervalli), 1) if intervalli else None


def scarto_massimo(mondi: list[Mondo], campo: str = "vivi") -> float | None:
    """Il MASSIMO intervallo fra esecuzioni identiche, su tutti i mondi e i semi.

    **La fascia va dichiarata col massimo e non con la media.** La media per
    mondo vale fra 123 e 241 coloni, ma il singolo scarto arriva a 427 e scende
    a 9: la varianza del modello e' una proprieta' del seme, non del mondo, e
    una fascia costruita sulla media lascia fuori punti che due esecuzioni
    identiche avrebbero potuto produrre da sole. Prima che i quattro mondi non
    di riferimento avessero repliche, la fascia veniva presa dal primo mondo che
    ne aveva; ora ne hanno tutti, e sceglierne uno sarebbe arbitrario.
    """
    intervalli = []
    for m in mondi:
        for s in m.semi:
            v = [getattr(e, campo) for e in m.coppia[s]]
            if len(v) >= 2:
                intervalli.append(max(v) - min(v))
    return round(max(intervalli), 1) if intervalli else None


def serie(cartella: Path, colonna: str = "population") -> tuple[list[int], list[float]]:
    """(passi, valori) da `state_timeseries.csv`; vuote se il file manca."""
    p = cartella / "state_timeseries.csv"
    if not p.exists():
        return [], []
    passi, valori = [], []
    with p.open(encoding="utf-8", errors="replace", newline="") as f:
        for riga in csv.DictReader(f):
            try:
                passi.append(int(float(riga["step"])))
                valori.append(float(riga[colonna]))
            except (KeyError, ValueError):
                continue
    return passi, valori


def differenza_appaiata(base: tuple[list[int], list[float]], tratt: tuple[list[int], list[float]]) -> tuple[list[int], list[float]]:
    """coppia - baseline sui passi comuni."""
    b = dict(zip(*base))
    fuori_p, fuori_v = [], []
    for p, v in zip(*tratt):
        if p in b:
            fuori_p.append(p)
            fuori_v.append(v - b[p])
    return fuori_p, fuori_v


def _segni(valori: list[float]) -> str:
    return "".join("+" if v > 0 else "-" if v < 0 else "0" for v in valori)


def riepilogo(mondi: list[Mondo]) -> list[dict]:
    """Una riga per mondo: medie, effetto sulle medie per seme, segni per seme, scarto fra repliche.

    Ogni media di LLM (Gov+Amm) passa prima per il seme: un seme ripetuto tre volte
    pesa come uno, non come tre. Altrimenti il seme con piu' esecuzioni sposta
    la media rispetto all'effetto, che e' gia' calcolato per seme."""
    def per_seme(m, f):
        return statistics.fmean(statistics.fmean(f(e) for e in m.coppia[s]) for s in m.semi)

    righe = []
    for m in mondi:
        eff = m.effetti("vivi")
        eff_medio_seme = [statistics.fmean(eff[s]) for s in m.semi]
        eff_celle = m.effetti("celle")
        righe.append({
            "mondo": m.nome,
            "etichetta": m.etichetta,
            "D": m.D,
            "semi": m.semi,
            "esecuzioni_coppia": max(len(m.coppia[s]) for s in m.semi),
            "vivi_baseline": round(statistics.fmean(m.baseline[s].vivi for s in m.semi), 1),
            "vivi_coppia": round(per_seme(m, lambda e: e.vivi), 1),
            "effetto_vivi": round(statistics.fmean(eff_medio_seme), 1),
            "segni_vivi": _segni(eff_medio_seme),
            "concordi_vivi": sum(1 for v in eff_medio_seme if v > 0),
            "celle_baseline": round(statistics.fmean(m.baseline[s].celle for s in m.semi), 1),
            "celle_coppia": round(per_seme(m, lambda e: e.celle), 1),
            "effetto_celle": round(statistics.fmean(statistics.fmean(eff_celle[s]) for s in m.semi), 1),
            "morti_per_nascita_baseline": round(statistics.fmean(
                m.baseline[s].morti / m.baseline[s].nascite for s in m.semi if m.baseline[s].nascite > 0), 3),
            "morti_per_nascita_coppia": round(per_seme(
                m, lambda e: e.morti / e.nascite if e.nascite > 0 else 0.0), 3),
            "scarto_repliche_vivi": scarto_repliche(m, "vivi"),
        })
    return righe


def rapporto(mondi: list[Mondo], figure: list[Path]) -> list[str]:
    out = ["# Confronto appaiato per configurazione: baseline contro governo+amministratori", ""]
    out.append("Stessi semi, stesso mondo, stessa dotazione (0.6); 300 fondatori, 1000 passi; LLM (Gov+Amm) = governatore "
               "gpt-oss:20b + amministratori gpt-oss:20b (livello completo). D = punteggio di difficolta' "
               "(`scripts/score_difficolta.py`). L'effetto e' la media, sui semi, di (LLM (Gov+Amm) - baseline) per seme; "
               "dove LLM (Gov+Amm) e' stato eseguito piu' volte sullo stesso seme, la media delle esecuzioni. "
               "«scarto repliche» = intervallo medio fra esecuzioni identiche, la varianza del modello.")
    out.append("")
    out.append("| mondo | D | semi | esecuzioni | vivi base | vivi LLM (Gov+Amm) | effetto vivi | segni | celle base | celle LLM (Gov+Amm) | effetto celle | morti/nascita base | morti/nascita LLM (Gov+Amm) | scarto repliche |")
    out.append("|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in riepilogo(mondi):
        out.append(
            f"| {r['mondo']} | {r['D']} | {','.join(map(str, r['semi']))} | {r['esecuzioni_coppia']} | {r['vivi_baseline']:.0f} | "
            f"{r['vivi_coppia']:.0f} | {r['effetto_vivi']:+.0f} | {r['segni_vivi']} ({r['concordi_vivi']}/{len(r['semi'])}) | "
            f"{r['celle_baseline']:.0f} | {r['celle_coppia']:.0f} | {r['effetto_celle']:+.0f} | "
            f"{r['morti_per_nascita_baseline']:.2f} | {r['morti_per_nascita_coppia']:.2f} | {r['scarto_repliche_vivi'] if r['scarto_repliche_vivi'] is not None else '—'} |"
        )
    out.append("")
    out.append("## Per seme")
    out.append("")
    out.append("| mondo | seme | D | vivi base | vivi LLM (Gov+Amm), per esecuzione | celle base | celle LLM (Gov+Amm) |")
    out.append("|---|---:|---:|---:|---|---:|---|")
    for m in mondi:
        for s in m.semi:
            b = m.baseline[s]
            out.append(
                f"| {m.nome} | {s} | {b.D} | {b.vivi} | {' / '.join(str(e.vivi) for e in m.coppia[s])} | "
                f"{b.celle} | {' / '.join(str(e.celle) for e in m.coppia[s])} |"
            )
    out.append("")
    if figure:
        out.append("Figure: " + ", ".join(f"`{p.as_posix()}`" for p in figure))
        out.append("")
    return out


# --------------------------------------------------------------------------- figure

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "figure.dpi": 150,
    })
    return plt


COLORE_BASE = "#4a4a4a"
COLORE_COPPIA = "#c2410c"
COLORE_REPLICHE = ("#c2410c", "#ea8a4b", "#7a2a08")


def _titolo(m: Mondo, lingua: str = "it", con_D: bool = True) -> str:
    e = m.etichetta_in(lingua)
    if con_D and m.D is not None:
        return f"{e}\n$D={m.D:.2f}$"
    # **Le etichette lunghe vanno mandate a capo.** Senza, nella serie delle
    # forme di governo i tre titoli si toccano e si leggono come una parola
    # sola: «governatore sologovernatore + amministratoriamministratori soli».
    # Il titolo e' l'unica cosa che dice quale braccio sia quale pannello.
    if len(e) > 18 and " " in e:
        meta = len(e) // 2
        taglio = min((i for i, c in enumerate(e) if c == " "),
                     key=lambda i: abs(i - meta))
        e = e[:taglio] + "\n" + e[taglio + 1:]
    return e


def disegna_esiti(mondi: list[Mondo], out: Path, larghezza: float = 6.3, lingua: str = "it", con_D: bool = True) -> Path:
    plt = _plt()
    T = TESTI[lingua]
    n = len(mondi)
    fig, assi = plt.subplots(2, n, figsize=(larghezza, 3.4), sharey="row", squeeze=False)
    for j, m in enumerate(mondi):
        for i, (campo, nome) in enumerate((("vivi", T["vivi_fine"]), ("celle", T["celle_fine"]))):
            ax = assi[i][j]
            x = list(range(len(m.semi)))
            base = [getattr(m.baseline[s], campo) for s in m.semi]
            for k, s in enumerate(m.semi):
                for r, e in enumerate(m.coppia[s]):
                    ax.plot([k, k], [base[k], getattr(e, campo)], color=COLORE_COPPIA, lw=0.6, alpha=0.5, zorder=1)
                    ax.scatter([k], [getattr(e, campo)], s=14, color=COLORE_REPLICHE[r % len(COLORE_REPLICHE)],
                               zorder=3, label=T["coppia"] if (k == 0 and r == 0) else None)
            ax.scatter(x, base, s=34, facecolors="white", edgecolors=COLORE_BASE, lw=1.1, zorder=4,
                       label=T["baseline"])
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in m.semi])
            if i == 0:
                ax.set_title(_titolo(m, lingua, con_D))
            if i == 1:
                ax.set_xlabel(T["seme"])
            if j == 0:
                # Ogni riga e' alta meno di quattro centimetri: l'etichetta per
                # esteso e ruotata e' piu' lunga dell'asse e sconfina nell'altra
                # riga, dove le due si sovrappongono. Qui va la grandezza e
                # basta; il «a fine run» lo dicono la didascalia e la nota.
                ax.set_ylabel(nome.split(" a fine run")[0].split(" at the end")[0],
                              fontsize=8)
            ax.grid(axis="y", lw=0.3, alpha=0.4)
    h, l = assi[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


def disegna_esiti_box(mondi: list[Mondo], out: Path, larghezza: float = 6.3,
                      lingua: str = "it", con_D: bool = True, altezza: float = 3.6) -> Path:
    """Gli stessi esiti, come distribuzioni invece che seme per seme.

    **Perche' il seme non deve stare sull'asse x.** Il numero del seme e'
    un'etichetta, non una grandezza: metterlo sull'asse suggerisce un ordine che
    non esiste e invita a leggere una pendenza fra semi che non significa
    niente. Cio' che il confronto ha da dire e' come sono fatte le due
    distribuzioni, e per quello serve un box.

    **Ma il box da solo butta via l'appaiamento, che e' il metodo.** Due box
    affiancati dicono «queste due popolazioni di numeri si somigliano», mentre
    l'affermazione del lavoro e' piu' stretta: *sullo stesso seme*, la colonia
    governata occupa meno celle di quella non governata. Percio' i punti restano,
    uniti da una linea che congiunge i due valori dello stesso seme: la linea che
    scende e' il dato, il box e' il riassunto. Una figura che mostrasse solo i
    box farebbe sembrare debole un risultato che e' forte proprio perche'
    appaiato.
    """
    plt = _plt()
    T = TESTI[lingua]
    n = len(mondi)
    fig, assi = plt.subplots(2, n, figsize=(larghezza, altezza), sharey="row", squeeze=False)
    for j, m in enumerate(mondi):
        for i, (campo, nome) in enumerate((("vivi", T["vivi_fine"]), ("celle", T["celle_fine"]))):
            ax = assi[i][j]
            base = [getattr(m.baseline[s], campo) for s in m.semi]
            # Per l'appaiamento LLM (Gov+Amm) vale una volta sola per seme: la media
            # delle sue esecuzioni. Le esecuzioni restano visibili come punti.
            coppia = [statistics.fmean(getattr(e, campo) for e in m.coppia[s]) for s in m.semi]

            corpi = ax.boxplot(
                [base, coppia], positions=[0, 1], widths=0.45, showfliers=False,
                medianprops=dict(color="black", lw=1.1),
                boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                patch_artist=True,
            )
            for corpo, colore in zip(corpi["boxes"], (COLORE_BASE, COLORE_COPPIA)):
                corpo.set_facecolor(colore)
                corpo.set_alpha(0.18)
                corpo.set_edgecolor(colore)

            for k, s in enumerate(m.semi):
                ax.plot([0, 1], [base[k], coppia[k]], color="#999999", lw=0.6, alpha=0.8, zorder=2)
                for r, e in enumerate(m.coppia[s]):
                    ax.scatter([1], [getattr(e, campo)], s=9, zorder=3, alpha=0.9,
                               color=COLORE_REPLICHE[r % len(COLORE_REPLICHE)],
                               label=T["coppia"] if (j == 0 and k == 0 and r == 0) else None)
                ax.scatter([0], [base[k]], s=20, facecolors="white", edgecolors=COLORE_BASE,
                           lw=0.9, zorder=4,
                           label=T["baseline"] if (j == 0 and k == 0) else None)

            ax.set_xticks([0, 1])
            if lingua == "pres":
                ax.set_xticklabels([T["baseline"], T["coppia"]], fontsize=7)
            else:
                ax.set_xticklabels([T["baseline"], T["coppia"]], fontsize=7,
                                   rotation=45, ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.55, 1.55)
            if i == 0:
                ax.set_title(_titolo(m, lingua, con_D))
                ax.set_xticklabels([])
            if j == 0:
                # Ogni riga e' alta meno di quattro centimetri: l'etichetta per
                # esteso e ruotata e' piu' lunga dell'asse e sconfina nell'altra
                # riga, dove le due si sovrappongono. Qui va la grandezza e
                # basta; il «a fine run» lo dicono la didascalia e la nota.
                ax.set_ylabel(nome.split(" a fine run")[0].split(" at the end")[0],
                              fontsize=8)
            ax.grid(axis="y", lw=0.3, alpha=0.4)
    # Niente legenda: ripeterebbe le due scritte gia' sull'asse. Al suo posto
    # l'unica cosa che il lettore non puo' indovinare, cioe' che cosa sono le
    # linee grigie --- ed e' l'informazione che rende valido il confronto.
    fig.text(0.5, -0.01, T["nota_box"], ha="center", va="top", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if lingua == "pres":
        fig.subplots_adjust(wspace=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


def disegna_traiettorie(mondi: list[Mondo], out: Path, colonna: str = "population", larghezza: float = 6.3) -> Path:
    plt = _plt()
    n = len(mondi)
    fig, assi = plt.subplots(2, n, figsize=(larghezza, 3.6), sharex=True, sharey="row", squeeze=False)
    cmap = plt.get_cmap("tab10")
    for j, m in enumerate(mondi):
        alto, basso = assi[0][j], assi[1][j]
        for k, s in enumerate(m.semi):
            col = cmap(k % 10)
            b = serie(m.baseline[s].cartella, colonna) if m.baseline[s].cartella else ([], [])
            if b[0]:
                alto.plot(*b, color=col, lw=0.7, ls=(0, (2, 1.5)), alpha=0.9)
            for e in m.coppia[s]:
                t = serie(e.cartella, colonna) if e.cartella else ([], [])
                if not t[0]:
                    continue
                alto.plot(*t, color=col, lw=0.7, alpha=0.9)
                if b[0]:
                    basso.plot(*differenza_appaiata(b, t), color=col, lw=0.7, alpha=0.9)
        basso.axhline(0, color="black", lw=0.6)
        alto.set_title(_titolo(m))
        basso.set_xlabel("passo (settimane)")
        for ax in (alto, basso):
            ax.grid(lw=0.3, alpha=0.4)
    assi[0][0].set_ylabel("coloni vivi" if colonna == "population" else colonna)
    assi[1][0].set_ylabel("LLM (Gov+Amm) − baseline\n(stesso seme)")
    alto = assi[0][0]
    alto.plot([], [], color=COLORE_BASE, lw=0.9, ls=(0, (2, 1.5)), label="baseline (tratteggio)")
    alto.plot([], [], color=COLORE_BASE, lw=0.9, label="LLM (Gov+Amm) (continua)")
    h, l = alto.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


def disegna_effetto_box(mondi: list[Mondo], out: Path, larghezza: float = 4.4,
                        lingua: str = "it", bracci: bool = False, altezza: float = 2.6) -> Path:
    """L'effetto appaiato come distribuzione: un box per mondo, sulle DIFFERENZE.

    **Perche' questa e' la figura giusta per dati appaiati.** Il box di una
    distribuzione di valori assoluti --- baseline a sinistra, governo a destra
    --- risponde alla domanda «queste due popolazioni di numeri si somigliano?»,
    che non e' la domanda del lavoro. La domanda del lavoro e' se la differenza
    *dentro la coppia di run dello stesso seme* abbia un segno, e l'oggetto che
    la risponde e' un box solo: quello delle differenze, con lo zero disegnato.
    Un box che sta tutto sotto lo zero e' l'affermazione; due box che si
    sovrappongono non dicono niente, nemmeno quando ogni singola differenza e'
    negativa.

    Sostituisce lo scatter contro il punteggio di difficolta' per una seconda
    ragione, che si vedeva a occhio nella figura vecchia: D non separa i semi
    dentro un mondo (nel riferimento va da 0,299 a 0,305), separa i mondi. Su un
    asse continuo i punti si ammucchiavano in due grumi verticali e la figura
    suggeriva una regressione che non c'e'. Il mondo sull'asse, ordinato per D,
    dice la stessa cosa senza prometterne un'altra.
    """
    plt = _plt()
    T = TESTI[lingua]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(larghezza, altezza))
    for ax, campo, nome in ((a1, "vivi", T["vivi"]), (a2, "celle", T["celle"])):
        # La fascia va presa dalla serie che HA le esecuzioni ripetute, non dalla
        # prima in ordine: nella serie delle forme di governo la prima e'
        # `llm_completo`, con una sola esecuzione per seme, e la figura restava
        # senza fascia proprio dove serviva a leggerla.
        rumore = None
        for m in sorted(mondi, key=lambda x: -max((len(v) for v in x.coppia.values()), default=0)):
            rumore = scarto_repliche(m, campo)
            if rumore is not None:
                break
        if rumore is not None:
            ax.axhspan(-rumore, rumore, color="#999999", alpha=0.18, lw=0,
                       label=T["fascia_nuda"])
        dati, etichette = [], []
        for m in mondi:
            eff = m.effetti(campo)
            dati.append([v for s in m.semi for v in eff[s]])
            etichette.append(m.etichetta_in(lingua))
        corpi = ax.boxplot(dati, widths=0.55, showfliers=False,
                           medianprops=dict(color="black", lw=1.1),
                           boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
                           capprops=dict(lw=0.8), patch_artist=True)
        for corpo in corpi["boxes"]:
            corpo.set_facecolor(COLORE_COPPIA)
            corpo.set_alpha(0.18)
            corpo.set_edgecolor(COLORE_COPPIA)
        for i, valori in enumerate(dati, start=1):
            # Un filo di dispersione orizzontale fissa, non casuale: due run
            # con lo stesso effetto devono restare distinguibili, e la figura
            # deve venire identica a ogni rigenerazione.
            for k, v in enumerate(valori):
                dx = (k % 5 - 2) * 0.055
                ax.scatter([i + dx], [v], s=9, color=COLORE_COPPIA, alpha=0.75, zorder=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticklabels(etichette, rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel(T["diff"] + "\n" + nome, fontsize=7)
        ax.grid(axis="y", lw=0.3, alpha=0.4)
    fig.text(0.5, -0.06, T["nota_effetto_bracci" if bracci else "nota_effetto"],
             ha="center", va="top", fontsize=6.5, color="#555555")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


def disegna_effetto_vs_difficolta(mondi: list[Mondo], out: Path, larghezza: float = 3.3, lingua: str = "it") -> Path:
    """Effetto per seme (LLM (Gov+Amm) - baseline) contro D della configurazione, con la fascia della varianza."""
    plt = _plt()
    T = TESTI[lingua]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(larghezza, 2.4))
    cmap = plt.get_cmap("tab10")
    # **Una fascia sola, e senza il suo numero nella legenda.** Prima ce n'erano
    # due annidate, la media e il massimo osservato, ognuna con la propria cifra
    # scritta accanto: tre informazioni sovrapposte in una figura larga otto
    # centimetri, e la cifra che si legge per prima (il ±160 della media) e'
    # proprio quella che NON stabilisce che cosa un'esecuzione sola puo'
    # dimostrare. Resta la fascia dello scarto medio, come riferimento visivo
    # per capire se un punto e' grande o piccolo; il numero che conta --- il
    # massimo --- sta nel testo, dove si puo' spiegare che cos'e'.
    for ax, campo, nome in ((a1, "vivi", T["vivi"]), (a2, "celle", T["celle"])):
        medie = [scarto_repliche(m, campo) for m in mondi]
        f_med = round(statistics.fmean([x for x in medie if x is not None]), 1) if any(
            x is not None for x in medie) else None
        if f_med is not None:
            ax.axhspan(-f_med, f_med, color="#999999", alpha=0.20, lw=0,
                       label=T["fascia_nuda"] if ax is a1 else None)
        ax.axhline(0, color="black", lw=0.6)
        for i, m in enumerate(mondi):
            col = cmap(i % 10)
            eff = m.effetti(campo)
            xs = [m.baseline[s].D for s in m.semi for _ in m.coppia[s]]
            ys = [v for s in m.semi for v in eff[s]]
            if any(x is None for x in xs):
                continue
            ax.scatter(xs, ys, s=12, color=col, alpha=0.85, label=m.etichetta_in(lingua), zorder=3)
        ax.set_ylabel(T["diff"] + "\n" + nome, fontsize=7)
        ax.set_xlim(0.25, 0.55)
        ax.grid(lw=0.3, alpha=0.4)
    fig.tight_layout(rect=(0, 0.30, 1, 1))
    fig.supxlabel(T["D"], fontsize=8, y=0.27)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- main

def _mondo_da_argomento(spec: str) -> Mondo:
    """`nome=baseline:coppia1,coppia2`."""
    nome, resto = spec.split("=", 1)
    base, coppie = resto.split(":", 1)
    return carica_mondo(nome, Path(base), [Path(c) for c in coppie.split(",") if c])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mondi-radice", type=Path, default=ROOT / "runs/mondi")
    ap.add_argument("--mondo", action="append", default=[], help="nome=baseline:coppia1,coppia2 (sostituisce la scoperta)")
    ap.add_argument("--out-tesi", type=Path, default=ROOT / "Tesi_LaTex/figures")
    ap.add_argument("--out-paper", type=Path, default=ROOT / "Paper_LaTex/AAMAS_Format/figures")
    ap.add_argument("--rapporto", type=Path, default=ROOT / "docs/benchmarks/2026-09-07-confronto-per-configurazione.md")
    ap.add_argument("--raggio", type=int, default=RAGGIO_DEFAULT)
    ap.add_argument("--serie", choices=("modelli", "livelli", "governo"), default=None,
                    help="invece dei mondi: la serie dei modelli (runs/modelli), dei livelli di contesto (runs/livelli) "
                         "o delle forme di governo (runs/governo: governatore solo, coppia, amministratori soli), "
                         "sullo stesso mondo di riferimento; figura confronto_<serie>.pdf e rapporto dedicato")
    ap.add_argument("--presentazione", type=Path, default=None, metavar="CARTELLA",
                    help="disegna SOLO le due figure delle slide di laurea (popolazione e rumore) "
                         "in CARTELLA; non tocca figure di tesi e paper ne' il rapporto")
    args = ap.parse_args()

    if args.presentazione:
        mondi = scopri_mondi(args.mondi_radice, raggio=args.raggio)
        mondi = sorted(mondi, key=lambda m: (m.D if m.D is not None else 9.0, m.nome))
        for r in riepilogo(mondi):
            print(f"{r['mondo']}: effetto celle {r['effetto_celle']:+.1f}, effetto vivi {r['effetto_vivi']:+.1f}")
        print(f"scarto massimo sui vivi: {scarto_massimo(mondi, 'vivi')}")
        import matplotlib
        matplotlib.rcParams["savefig.dpi"] = 250
        # Disegnate piccole e salvate fitte: in proporzione le scritte vengono
        # piu' grandi, ed e' quello che serve a una figura proiettata.
        for f in (disegna_esiti_box(mondi, args.presentazione / "popolazione_vs_celle.pdf",
                                    larghezza=5.6, altezza=3.0, lingua="pres", con_D=False),
                  disegna_effetto_box(mondi, args.presentazione / "rumore_box.pdf",
                                      larghezza=5.2, altezza=2.6, lingua="pres")):
            f.unlink()  # per le slide basta il PNG
            print(f"figura {f.with_suffix('.png')}")
        return 0

    if args.serie:
        radice = ROOT / "runs" / args.serie
        nome_rif = {"modelli": "gptoss20b", "livelli": "completo", "governo": "coppia"}[args.serie]
        serie = scopri_serie(radice, nome_rif, raggio=args.raggio)
        if len(serie) < 2:
            print(f"nessuna coppia in {radice} oltre al riferimento", file=sys.stderr)
            return 1
        for m in serie:
            print(f"{m.nome}: semi={m.semi} esecuzioni={[len(m.coppia[s]) for s in m.semi]}")
        larg = 1.3 * len(serie) + 0.8
        figure = [
            # Nel corpo della tesi la domanda è appaiata: mostriamo direttamente
            # le differenze rispetto alla baseline, con lo zero esplicito.
            disegna_effetto_box(serie, args.out_tesi / f"confronto_{args.serie}.pdf", larghezza=larg, bracci=True),
            # La vista sui valori assoluti resta disponibile come figura di
            # supporto, ma non è più quella richiamata nel capitolo dei risultati.
            disegna_esiti_box(serie, args.out_tesi / f"confronto_{args.serie}_box.pdf", larghezza=larg, con_D=False),
            disegna_esiti(serie, args.out_paper / f"confronto_{args.serie}.pdf", larghezza=larg, lingua="en", con_D=False),
        ]
        rapp = ROOT / "docs/benchmarks" / f"2026-09-08-confronto-{args.serie}.md"
        rapp.write_text("\n".join(rapporto(serie, figure)) + "\n", encoding="utf-8")
        print(f"scritto {rapp}")
        for f in figure:
            print(f"figura {f}")
        return 0

    mondi = [_mondo_da_argomento(s) for s in args.mondo] if args.mondo else scopri_mondi(args.mondi_radice, raggio=args.raggio)
    mondi = sorted(mondi, key=lambda m: (m.D if m.D is not None else 9.0, m.nome))
    if not mondi:
        print("nessun mondo con baseline e coppia sugli stessi semi", file=sys.stderr)
        return 1
    for m in mondi:
        print(f"{m.nome}: D={m.D} semi={m.semi} esecuzioni={[len(m.coppia[s]) for s in m.semi]}")

    figure = [
        disegna_esiti(mondi, args.out_tesi / "confronto_esiti.pdf"),
        disegna_esiti_box(mondi, args.out_tesi / "confronto_esiti_box.pdf"),
        disegna_esiti_box(mondi, args.out_paper / "confronto_esiti_box.pdf", larghezza=7.0, lingua="en"),
        disegna_traiettorie(mondi, args.out_tesi / "confronto_traiettorie.pdf"),
        disegna_effetto_box(mondi, args.out_tesi / "effetto_box.pdf"),
        disegna_effetto_box(mondi, args.out_paper / "effetto_box.pdf", lingua="en"),
        disegna_effetto_vs_difficolta(mondi, args.out_tesi / "effetto_vs_difficolta.pdf"),
        disegna_effetto_vs_difficolta(mondi, args.out_paper / "effetto_vs_difficolta.pdf", lingua="en"),
        disegna_esiti(mondi, args.out_paper / "confronto_esiti.pdf", larghezza=7.0, lingua="en"),
    ]
    testo = "\n".join(rapporto(mondi, figure)) + "\n"
    args.rapporto.parent.mkdir(parents=True, exist_ok=True)
    args.rapporto.write_text(testo, encoding="utf-8")
    print(f"scritto {args.rapporto}")
    for f in figure:
        print(f"figura {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
