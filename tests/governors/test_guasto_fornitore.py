"""Un fornitore giu' non e' un amministratore che accetta.

L'11 settembre 2026 la campagna con amministratori misti ha girato con
l'endpoint di uno dei quattro modelli spento. Tutte e 118 le sue chiamate sono
fallite, `api_usage.json` le ha contate come `failed_calls`, e il registro per
tornata le ha riportate come 118 accettazioni ben formate su 118: nel confronto
fra modelli quel modello risultava l'unico che non riscrive mai. Un guasto si
era travestito da risultato scientifico.

La catena era questa: il fornitore, quando la chiamata fallisce, non solleva e
non restituisce testo vuoto, ma il payload di ripiego pensato per la decisione
di un COLONO (`chosen_action`, `thought_summary`, ...). Quel JSON si legge
benissimo come dizionario, non ha la chiave `accept`, e
`grezza.get("accept", True)` vale allora `True`.
"""

import pytest

from src.governors.admin_arms import AmministratoreLLM


class _RispostaGuasta:
    """Cio' che il fornitore restituisce quando l'endpoint non risponde."""

    text = (
        '{"thought_summary": "Fallback mode: conserve resources and explore locally.",'
        ' "chosen_action": "observe", "magnitude": 1.0}'
    )
    error = "connessione rifiutata"
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0


class _FornitoreGuasto:
    provider_id = "gpu_farm4"
    model = "Qwen/Qwen3.8-27B-FP8"

    def complete_json(self, prompt):
        return _RispostaGuasta()


def test_il_ripiego_del_fornitore_non_viene_letto_come_decisione():
    """Con `error` valorizzato la risposta e' un guasto, non un'accettazione."""
    esito = AmministratoreLLM(_FornitoreGuasto(), "Qwen/Qwen3.8-27B-FP8").propose_text("x")
    assert esito.get("failed") is True, (
        "senza questo marcatore lo strato non puo' distinguere un endpoint "
        "spento da un amministratore che accetta"
    )
    assert "chosen_action" not in (esito.get("raw") or {}), (
        "il payload di ripiego di un colono non deve arrivare allo strato"
    )


def test_un_guasto_non_e_ne_accettazione_ne_intervento(tmp_path):
    """Il registro per tornata deve avere un quarto esito."""
    from src.governors.administration import DecisioneAmministratore

    d = DecisioneAmministratore(distretto=0, policy=None, guasto=True, celle=((0, 0),))
    assert d.guasto is True
    assert d.accettata is False, "un guasto non e' un'accettazione"
    assert d.to_json()["provider_failure"] is True
    assert d.to_json()["accepted_government_policy"] is False
