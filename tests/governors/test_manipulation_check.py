"""La verifica di manipolazione, prima di spendere una chiamata API.

Se una policy che spinge in una direzione nota NON produce lo spostamento
corrispondente, la leva non e' attaccata — e allora un risultato nullo
dell'esperimento non distinguerebbe "governare con un LLM non aiuta" da "la
leva non era collegata". Questo test scopre quel difetto a costo zero.

**Perche' la spinta e' estrema e arriva da un registro scritto a mano.** La
scelta dei coloni e' un campionamento proporzionale su sei pilastri: una
regola blanda (x2,5 su un pilastro, condizionata) sposta le frequenze meno del
rumore del softmax su un solo seme — misurato: la costituzione dello scripted
su 20 passi muove le raccolte in ENTRAMBE le direzioni a seconda dello
scenario. Una policy incondizionata che pesa `explore` x4 e gli altri pilastri
di lavoro x0,25 impone invece un rapporto 16:1 che nessun rumore copre. E
passarla per `replay_from` verifica insieme la leva E il percorso di
riesecuzione con una policy d'autore, non solo con quelle registrate live.
"""

import math

from scripts.parity_harness import realistic_config
from scripts.run_governor_experiment import run_arm
from src.agents import pillars
from src.governors.arms import GovernorProposal
from src.governors.observation import ColonyPicture
from src.governors.policy import Policy, Rule
from src.governors.record import GovernorRecorder
from src.simulation.agent_coupled_runner import AgentCoupledRunner

AGENTS, STEPS, CADENCE = 40, 20, 2

EXPLORE_EVERYTHING = Policy(
    (
        Rule(None, {
            pillars.P_EXPLORE: 4.0,
            pillars.P_RESOURCES: 0.25,
            pillars.P_BUILD: 0.25,
            pillars.P_SUSTENANCE: 0.25,
        }),
    ),
    rationale="spinta di prova: esplorare sopra ogni cosa",
)


def _scenario(replay_from=None) -> dict:
    config = realistic_config(AGENTS, STEPS, 0)
    config["redistribution"]["greenhouse_per_capita"] = 0.10
    config["colony"]["initial_structures"]["greenhouse"] = max(
        1, math.ceil(AGENTS * 0.05)
    )
    if replay_from is not None:
        config["governors"] = {
            "arm": "llm",
            "cadence_steps": CADENCE,
            "replay_from": str(replay_from),
        }
    return config


def _exploration_share(runner) -> float:
    exploring = sum(
        runner.action_counts.get(name, 0) for name in ("observe", "explore", "move")
    )
    return exploring / max(1, sum(runner.action_counts.values()))


def test_an_explore_pushing_policy_produces_more_exploration(tmp_path):
    record = tmp_path / "governor_decisions.jsonl"
    recorder = GovernorRecorder(record)
    picture = ColonyPicture(step=1, population=AGENTS)
    for tick in range((STEPS // CADENCE) + 1):
        recorder.record(
            tick=tick,
            application_step=1 + (tick + 1) * CADENCE,
            picture=picture,
            proposal=GovernorProposal(policy=EXPLORE_EVERYTHING, provider="test"),
            policy=EXPLORE_EVERYTHING,
            missed=False,
        )
    recorder.close()

    plain = AgentCoupledRunner(_scenario())
    plain.run(days=STEPS, output_dir=None)
    pushed = AgentCoupledRunner(_scenario(replay_from=record))
    pushed.run(days=STEPS, output_dir=None)

    assert _exploration_share(pushed) > _exploration_share(plain), (
        "la leva non e' attaccata: una policy che pesa explore 16 volte gli "
        "altri pilastri non ha spostato il lavoro verso l'esplorazione. Ogni "
        "risultato nullo dell'esperimento sarebbe ambiguo."
    )


def test_every_arm_runs_and_reports_the_same_metric_keys(tmp_path):
    results = {
        arm: run_arm(arm, seed=0, steps=12, agents=40, output_dir=tmp_path / arm,
                     cadence_steps=2)
        for arm in ("none", "random", "scripted")
    }
    keys = {frozenset(value) for value in results.values()}
    assert len(keys) == 1, "i bracci devono essere confrontabili campo per campo"
    for arm, value in results.items():
        assert value["arm"] == arm
        assert value["steps"] == 12
