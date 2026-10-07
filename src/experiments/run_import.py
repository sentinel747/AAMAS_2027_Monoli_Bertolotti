"""Ripartire da una run gia' eseguita invece di ricomporne la configurazione.

Una run salva la propria configurazione completa in `config.yaml` (JSON, malgrado
l'estensione). Riaprirla e riconvertirla in `ManualConfigOptions` permette di
ripetere un esperimento cambiando una cosa sola, che e' il caso di gran lunga
piu' frequente: cambiare un seme, alzare i passi, spostare un braccio.

**Perche' un modulo condiviso e non due letture separate.** Il pannello della GUI
e il wizard da terminale devono importare la *stessa* configurazione dalla
*stessa* run, altrimenti "ripeti quella run" significherebbe due cose diverse a
seconda di dove lo si chiede — ed e' esattamente la classe di divergenza che
`test_the_wizard_and_the_gui_panel_cover_the_same_options` esiste per impedire.

L'ordinamento e' per data di scrittura decrescente e non alfabetico: chi vuole
ripetere una run quasi sempre vuole ripetere l'ultima, e un elenco alfabetico
mette in cima `campagna_a` di tre settimane fa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.manual_config import ManualConfigOptions, options_from_scenario_config

#: Il file che porta la configurazione completa. L'estensione dice YAML ma il
#: contenuto e' JSON: e' cosi' da prima di questo modulo e non va "corretto"
#: qui, perche' cambiarlo renderebbe illeggibili le run gia' salvate.
CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class RunSummary:
    """Quanto basta per riconoscere una run in un elenco, senza aprirla tutta."""

    run_id: str
    path: Path
    modified: float
    name: str
    seed: int
    steps: int | None
    agents: int
    map_profile: str
    governors: str

    def label(self) -> str:
        """Una riga leggibile, con i campi che distinguono davvero due run."""
        passi = "illimitati" if self.steps is None else str(self.steps)
        pezzi = [
            f"seed {self.seed}",
            f"{self.agents} agenti",
            f"{passi} passi",
            self.map_profile,
        ]
        if self.governors != "none":
            pezzi.append(f"governatori: {self.governors}")
        return f"{self.name} ({', '.join(pezzi)})"


def _run_config_path(directory: Path) -> Path | None:
    percorso = directory / CONFIG_FILENAME
    return percorso if percorso.is_file() else None


def load_run_config(directory: Path) -> dict[str, Any] | None:
    """La configurazione di una run, o `None` se non e' leggibile.

    `None` e non un'eccezione: una cartella senza configurazione — interrotta a
    meta', o prodotta da una versione precedente — non deve impedire di elencare
    e importare tutte le altre.
    """
    percorso = _run_config_path(directory)
    if percorso is None:
        return None
    try:
        dati = json.loads(percorso.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None
    return dati if isinstance(dati, dict) else None


def summarize_run(run_id: str, directory: Path) -> RunSummary | None:
    config = load_run_config(directory)
    if config is None:
        return None
    agenti = config.get("agents") if isinstance(config.get("agents"), dict) else {}
    mondo = config.get("world") if isinstance(config.get("world"), dict) else {}
    consiglio = config.get("governors") if isinstance(config.get("governors"), dict) else {}
    passi = config.get("days")
    try:
        modificata = (directory / CONFIG_FILENAME).stat().st_mtime
    except OSError:
        modificata = 0.0
    return RunSummary(
        run_id=run_id,
        path=directory,
        modified=modificata,
        name=str(config.get("name") or run_id),
        seed=int(config.get("seed", 0) or 0),
        steps=None if passi in (None, 0) else int(passi),
        agents=int(agenti.get("count", 0) or 0),
        map_profile=str(mondo.get("map_profile") or "balanced"),
        governors=str(consiglio.get("arm") or "none"),
    )


def list_importable_runs(limit: int = 40) -> list[RunSummary]:
    """Le run importabili, dalla piu' recente. Solo quelle con configurazione.

    Il tetto esiste perche' questo elenco finisce in un menu di terminale e in un
    menu a tendina: oltre qualche decina di voci nessuno dei due e' usabile, e
    chi cerca una run di sei mesi fa la apre per percorso.
    """
    # Import locale: `routes_runs` importa FastAPI, e il wizard da terminale deve
    # poter funzionare anche dove il server non e' installato.
    from src.api.routes_runs import discover_runs

    riassunti = [
        riassunto
        for run_id, directory in discover_runs().items()
        if (riassunto := summarize_run(run_id, directory)) is not None
    ]
    riassunti.sort(key=lambda r: (-r.modified, r.run_id))
    return riassunti[: max(1, limit)]


def options_from_run(run_id: str) -> ManualConfigOptions | None:
    """Le opzioni di una run passata, pronte da modificare e rilanciare.

    Il nome della run **non** viene ereditato: ripetere una run con lo stesso
    nome sovrascriverebbe o affiancherebbe artefatti indistinguibili, ed e'
    proprio la confusione che questa funzione dovrebbe far risparmiare. Viene
    lasciato vuoto, cosi' entrambe le interfacce chiedono il nome nuovo.
    """
    from src.api.routes_runs import discover_runs

    directory = discover_runs().get(run_id)
    if directory is None:
        return None
    config = load_run_config(directory)
    if config is None:
        return None
    opzioni = options_from_scenario_config(config, str(config.get("scenario_id") or ""))
    opzioni.run_name = ""
    return opzioni
