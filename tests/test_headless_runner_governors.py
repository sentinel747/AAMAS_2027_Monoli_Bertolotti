"""Le due interfacce rimaste devono poter fare le stesse cose.

Da quando `appnogui.py` non e' piu' in uso, le run si lanciano dal frontend
(che costruisce il governatore dalla configurazione) o da
`scripts/headless_runner.py`. Il secondo non aveva **nessuna** opzione per il
governatore: da terminale l'unico modo era un blocco JSON in `--overrides`,
cioe' il modo migliore per sbagliare in silenzio `cadence_steps` — l'unico
parametro capace di rendere nullo l'intero esperimento.
"""

from scripts.headless_runner import _apply_agent_logs, _apply_governors, build_parser


def _args(*argv):
    return build_parser().parse_args(["--run-name", "prova", *argv])


def test_without_the_option_no_governor_block_is_written():
    """Assente, non `arm: none`.

    `build_governor` restituisce `None` sia con il blocco assente sia con
    `arm: "none"`, ma la baseline resta bit-exact **per costruzione** solo
    finche' il kernel non ha nemmeno una policy da applicare: scrivere il
    blocco comunque sposterebbe quella garanzia dal codice a una lettura di
    stringa.
    """
    config = {}
    _apply_governors(config, _args())
    assert "governors" not in config


def test_the_governor_options_reach_the_config():
    config = {}
    _apply_governors(config, _args(
        "--governors-arm", "llm", "--governors-cadence", "20", "--governors-wait", "60",
        "--governors-provider", "gpu_farm", "--governors-model", "gpt-oss:20b",
        "--governors-effort", "low", "--governors-temperature", "0",
    ))
    blocco = config["governors"]
    assert blocco["arm"] == "llm"
    assert blocco["cadence_steps"] == 20
    assert blocco["wait_seconds"] == 60.0
    assert "mandates" not in blocco, "i mandati non esistono piu'"
    # `thinking` compare sempre, anche a `off`: il livello di ragionamento e'
    # una condizione della run e va scritto nella configurazione salvata, non
    # lasciato implicito. Un default implicito e' esattamente cio' che cambia
    # sotto i piedi fra una versione del registro e la successiva, e una run
    # riletta l'anno prossimo direbbe di essere girata con un'altra.
    assert blocco["assignments"] == [
        {
            "provider": "gpu_farm",
            "model": "gpt-oss:20b",
            "effort": "low",
            "thinking": "off",
            "temperature": 0.0,
        }
    ]
    assert blocco["context_level"] == "completo"
    assert "administrators" not in blocco, "senza --administrators lo strato non esiste"


def test_a_control_arm_needs_no_provider():
    config = {}
    _apply_governors(config, _args("--governors-arm", "random"))
    assert config["governors"]["arm"] == "random"
    assert "assignments" not in config["governors"], "un braccio senza rete non assegna provider"


def test_temperature_zero_is_a_value_and_not_an_absence():
    """Zero e' proprio il valore che serve, ed e' falsy."""
    config = {}
    _apply_governors(config, _args(
        "--governors-arm", "llm", "--governors-provider", "gpu_farm",
        "--governors-model", "m", "--governors-temperature", "0",
    ))
    assert config["governors"]["assignments"][0]["temperature"] == 0.0


def test_the_agent_logs_are_on_unless_asked_otherwise():
    config = {}
    _apply_agent_logs(config, _args().agent_logs)
    assert config["headless"]["store_memory_logs"] is True
    _apply_agent_logs(config, _args("--no-agent-logs").agent_logs)
    assert config["headless"]["store_memory_logs"] is False
