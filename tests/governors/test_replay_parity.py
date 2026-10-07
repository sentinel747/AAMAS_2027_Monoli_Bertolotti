"""Una run con governatori e' rieseguibile, e in riesecuzione da' lo stesso stato.

E' il test che restituisce al percorso governatori l'oracolo del progetto. Se
cade, il registro non contiene tutto cio' che serve a riprodurre la run, e nessun
risultato prodotto con i governatori sarebbe ri-derivabile.

Nello stesso file sta il **cancello delle prestazioni in forma strutturale**: con
`governors.count: 0` non si costruisce alcun governatore, non parte alcun thread
e `Governor.advance` non viene mai chiamata. E' piu' solido di qualunque cronometro,
perche' non dipende dal rumore della macchina — i numeri misurati col cronometro
stanno in `docs/benchmarks/README.md`, dove un cambio di carico li invecchia
senza tingere di rosso la suite.
"""

import math
import threading

import pytest

from scripts.parity_harness import realistic_config
import src.governors.arms as arms
from src.core.state_digest import step_digest
from src.governors.config import build_governor
from src.governors.record import load_records
from src.simulation.agent_coupled_runner import AgentCoupledRunner

#: Passi e coloni delle run di questo file. Dieci passi con cadenza 2 danno cinque
#: confini di tick, quindi quattro tick registrati (l'ultimo resta in volo alla
#: fine della run): abbastanza perche' la riesecuzione debba pescare piu' di una
#: riga e nell'ordine giusto, poco abbastanza da restare un test.
#: Duecento e non quaranta. La costituzione scelta il 2026-08-24 (`docs/
#: benchmarks/2026-08-24-costituzioni.md`) ha come seconda regola `occupants >
#: 150 -> explore x2`, che su una colonia di quaranta coloni non puo' scattare
#: mai — e la prima (`food < 2`) scatta solo al passo iniziale, prima che la
#: prima policy entri in vigore. Su quaranta agenti questo file confronterebbe
#: quindi **due baseline**, cioe' sarebbe verde anche col registro scollegato:
#: e' esattamente il modo di mentire che `test_the_recorded_run_is_not_
#: trivially_equal_to_an_ungoverned_one` esiste per impedire, ed e' stato quel
#: test a segnalarlo quando la costituzione e' cambiata. Misurato a duecento:
#: la regola scatta sei volte in otto passi e il digest cambia.
AGENTS = 200
#: Venti e non dieci. Con la priorita' di sviluppo resa proporzionale al
#: fabbisogno (2026-08-21) i primi passi della colonia sono occupati anche da
#: laboratori e infermerie, e a dieci passi la direttiva dello `scripted` non
#: faceva ancora in tempo a produrre uno stato distinguibile dalla baseline:
#: misurato, la differenza ricompare a venti. Il guardiano ha fatto il proprio
#: lavoro — segnalare che lo scenario era diventato troppo corto — e non va
#: silenziato abbassando l'asticella.
STEPS = 20
CADENCE = 2


def _scenario(steps: int = STEPS) -> dict:
    """Lo scenario delle run: `realistic_config` con le due correzioni del Task 12a.

    **Scostamento dal piano, con la sua ragione — e' la stessa gia' scritta in
    `scripts/run_governor_experiment.py`.** Il piano faceva registrare una run su
    `realistic_config` liscia. Misurato qui: su quella configurazione un consiglio
    `scripted` con mandato `life_support` PARLA a ogni tick (moltiplicatore 1,74 e
    quota 9 su `BUILD_GREENHOUSE`) e la run resta **bit-identica** alla baseline —
    `756fa61b3a8a3a4cf69cc25c4ed6e750` con e senza governatori. Il motivo e' quello
    che il Task 12a ha misurato: `greenhouse_per_capita = 0,25` chiede una serra
    ogni 4 coloni mentre la maschera ne concede una ogni 7, quindi la colonia
    costruisce gia' al ritmo massimo e nessuna direttiva ha dove spingere.

    Su uno scenario cosi' la parita' di riesecuzione confronterebbe **due
    baseline**: sarebbe verde anche con il registro scollegato, cioe' proprio
    quando serve. Le due correzioni portano il piano della colonia sotto il tetto
    fisico e la colonia sotto il proprio piano, ed erano gia' state adottate per
    l'esperimento a quattro bracci. Valgono per tutte le run di questo file,
    governate e non: sono lo scenario, non un trattamento.
    """
    config = realistic_config(AGENTS, steps, 0)
    config["redistribution"]["greenhouse_per_capita"] = 0.10
    config["colony"]["initial_structures"]["greenhouse"] = max(
        1, math.ceil(AGENTS * 0.05)
    )
    return config


#: La costituzione di QUESTO test, indipendente da quella in vigore.
#:
#: Pesa `build` dove la copertura elettrica e' corta: su questo scenario e' la
#: condizione che morde, perche' la colonia costruisce e la corrente e' il suo
#: vincolo. Non e' la costituzione di riferimento e non pretende di esserlo —
#: quella si sceglie con `scripts/scegli_costituzione.py` e cambia quando cambia
#: il mondo. Qui serve soltanto una politica che produca uno stato diverso dalla
#: baseline, perche' altrimenti la parita' di riesecuzione confronterebbe due
#: baseline e sarebbe verde anche col registro scollegato.
COSTITUZIONE_DI_PROVA = (
    ("power_coverage", "<", 1.0, "build", 3.0),
)


@pytest.fixture(autouse=True)
def _costituzione_di_prova(monkeypatch):
    """Sostituisce la costituzione globale per la durata di ogni test del file."""
    from src.governors.policy import Bounds, Condition, PILLAR_BY_NAME, Policy, Rule

    def politica(bounds: Bounds) -> Policy:
        return Policy(
            tuple(
                Rule(
                    Condition(indicatore, op, soglia),
                    {PILLAR_BY_NAME[pilastro]: min(max(peso, bounds.weight_min), bounds.weight_max)},
                )
                for indicatore, op, soglia, pilastro, peso in COSTITUZIONE_DI_PROVA
            ),
            "costituzione di prova",
        )

    monkeypatch.setattr(arms, "reference_policy", politica)


def _governed_config(replay_from=None) -> dict:
    """La configurazione delle due run: identica, salvo da dove arrivano le direttive.

    Il braccio del ramo di riesecuzione e' deliberatamente `llm`: con
    `replay_from` valorizzato `build_council` non deve nemmeno guardarlo, quindi
    scriverci il braccio piu' costoso e' la prova che nessun provider viene
    costruito. Se un giorno la riesecuzione smettesse di scavalcare il braccio, il
    test fallirebbe qui e non a valle con un digest diverso.
    """
    config = _scenario()
    config["governors"] = {
        "arm": "llm" if replay_from else "scripted",
        "cadence_steps": CADENCE,
    }
    if replay_from:
        config["governors"]["replay_from"] = str(replay_from)
    return config


def _directives(path) -> list:
    """Le policy registrate, tick per tick: la diagnosi quando la parita' cade."""
    return [row["policy"] for row in load_records(path)]


def test_a_replayed_run_reproduces_the_recorded_run_exactly(tmp_path):
    """**Secondo scostamento dal piano.** Il piano faceva scrivere gli artefatti
    alla sola run registrata (`output_dir=tmp_path / "live"`) e non alla
    riesecuzione (`output_dir=None`). E' gia' misurato altrove in questa cartella
    (`test_shell_wiring.test_an_active_council_changes_the_run_and_writes_its_
    record`) che salvare gli artefatti **cambia il digest**: due run identiche
    senza governatori danno `398e6f57...` senza cartella e `6e2adca3...` con
    cartella, perche' il salvataggio attraversa ogni cella del pianeta e
    materializza voci che `state_digest` include. Con il confronto del piano
    questo test sarebbe stato rosso anche con una riesecuzione perfetta, e la
    diagnosi sarebbe finita sul registro, che non c'entra. Le due run scrivono
    quindi in due cartelle diverse ma ricevono lo stesso trattamento, e l'unica
    differenza che resta e' da dove vengono le direttive.

    Effetto collaterale voluto: anche la riesecuzione lascia il proprio registro,
    quindi la diagnosi prescritta dal piano — confrontare le direttive tick per
    tick — e' disponibile senza strumentare nulla, ed e' la prima asserzione:
    quando cade, dice a che tick le due run hanno divergito prima ancora di
    guardare lo stato.
    """
    live = AgentCoupledRunner(_governed_config())
    live.run(days=STEPS, output_dir=tmp_path / "live")
    recorded = step_digest(live.core)["overall"]
    live_record = tmp_path / "live" / "governor_decisions.jsonl"

    replayed = AgentCoupledRunner(_governed_config(replay_from=live_record))
    replayed.run(days=STEPS, output_dir=tmp_path / "replay")
    replay_record = tmp_path / "replay" / "governor_decisions.jsonl"

    assert _directives(replay_record) == _directives(live_record), (
        "la riesecuzione ha applicato direttive diverse tick per tick: il "
        "registro non contiene tutto cio' che serve, oppure non viene riletto "
        "nell'ordine in cui e' stato scritto"
    )
    assert step_digest(replayed.core)["overall"] == recorded, (
        "la riesecuzione non riproduce la run registrata: il registro non "
        "contiene tutto cio' che serve"
    )


def test_the_recorded_run_is_not_trivially_equal_to_an_ungoverned_one(tmp_path):
    """Test AGGIUNTO. Non tocca quello sopra: gli toglie il modo di mentire.

    **Pericolo coperto.** La parita' di riesecuzione e' un'uguaglianza, e ogni
    uguaglianza ha un modo banale di essere vera: se il consiglio non spostasse
    nulla — un mandato che tace, un registro di righe vuote, una direttiva che non
    arriva al kernel, o uno scenario in cui non ha dove spingere — allora run
    registrata e riesecuzione sarebbero identiche perche' **entrambe** sarebbero la
    baseline, e il test qui sopra resterebbe verde mentre l'oracolo non guarda piu'
    niente. Non e' un pericolo teorico: e' esattamente cio' che accade sullo
    scenario che il piano prescriveva, ed e' il motivo per cui `_scenario` esiste.

    Nessun altro test lo copre da questo lato: `test_shell_wiring` prova che un
    consiglio **casuale** cambia lo stato, con un braccio diverso da questo e su
    un altro scenario, e non dice niente su cio' che la riesecuzione riproduce.

    Due asserzioni e non una, perche' i due modi di essere vacui sono diversi e
    hanno diagnosi diverse: un registro di direttive vuote dice che il governatore
    non ha parlato, un registro pieno con digest uguale dice che ha parlato a
    vuoto.
    """
    live = AgentCoupledRunner(_governed_config())
    live.run(days=STEPS, output_dir=tmp_path / "live")
    governed = step_digest(live.core)["overall"]
    live_record = tmp_path / "live" / "governor_decisions.jsonl"

    plain = AgentCoupledRunner(_scenario())
    plain.run(days=STEPS, output_dir=tmp_path / "plain")

    speaking = [row for row in load_records(live_record) if row["policy"]["rules"]]
    assert speaking, (
        "il registro non contiene nemmeno una policy con regole: la parita' di "
        "riesecuzione confronterebbe due baseline e non proverebbe nulla"
    )
    assert governed != step_digest(plain.core)["overall"], (
        "la run registrata e' indistinguibile da una senza governatori: la "
        "riesecuzione riprodurrebbe la baseline, non una run governata"
    )


def test_without_governors_no_council_is_built_and_no_thread_starts(monkeypatch):
    """Il cancello delle prestazioni, in forma strutturale.

    Il costo di un consiglio spento non si misura col cronometro: si dimostra
    guardando che non esista. `governors.count: 0` e' il default, quindi questo
    e' il caso di OGNI run che non chieda i governatori — le run di tesi
    comprese — e la garanzia deve valere per costruzione e non per misura. I
    numeri col cronometro restano in `docs/benchmarks/README.md`, dove invecchiano
    senza far diventare rossa la suite quando la macchina cambia carico.

    Tre cose insieme, perche' sono tre modi diversi di pagare: nessun oggetto
    (`build_council` restituisce `None`), nessun thread (il consiglio ne possiede
    uno che nasce nel costruttore), nessuna chiamata per passo (`advance`
    sostituita con una che solleva: se la shell la invocasse anche una sola volta
    la run morirebbe, invece di rallentare in silenzio).

    Il pericolo e' concreto: la sola scrittura `if self._council is not None` in
    ciascuna delle due shell separa una run di tesi dal costruire un quadro di
    colonia a ogni passo, e un refactor che la togliesse — costruendo il quadro
    sempre e usandolo solo a volte — non farebbe fallire nessun altro test.
    """
    from src.governors.governor import Governor

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "advance chiamata senza governatori: il percorso non e' spento"
        )

    monkeypatch.setattr(Governor, "advance", forbidden)

    config = realistic_config(20, 6, 0)
    assert build_governor(config, 0) is None, (
        "senza sezione `governors` non deve nascere alcun governatore"
    )
    config["governors"] = {"count": 0}
    assert build_governor(config, 0) is None, (
        "governors.count: 0 era l'interruttore spento del consiglio e resta spento"
    )

    before = {thread.name for thread in threading.enumerate()}
    runner = AgentCoupledRunner(config)
    assert runner._governor is None
    assert "governors" not in {t.name for t in threading.enumerate()} - before, (
        "il thread del consiglio e' partito comunque: il costo c'e' anche quando "
        "il consiglio non serve"
    )
    runner.run(days=6, output_dir=None)
    assert runner.core.governor_policy is None, (
        "senza governatore il kernel non deve vedere alcuna policy"
    )
