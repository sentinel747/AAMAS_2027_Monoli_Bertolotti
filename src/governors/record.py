from __future__ import annotations

"""Registro per tick, e riesecuzione dal registro.

E' cio' che restituisce a una run col governatore la verificabilita' che il
resto del progetto ha. Le risposte di un modello non sono deterministiche,
quindi il digest bit-exact per seme non si applica a una run LLM; registrando
quadro, proposta e policy, quella run diventa **rieseguibile**, e in
riesecuzione il digest torna a essere un oracolo.

Effetto collaterale importante e voluto: `ReplayProposer` non costruisce alcun
provider, quindi l'intero percorso — governatore, applicazione, kernel — si
verifica senza mai chiamare un'API.

**Formato v2 (2026-08-24).** Una riga per tick con UN `governor` e una
`policy`, al posto della lista `governors` e della `directive` per celle del
consiglio. Un registro del formato precedente non e' rieseguibile qui: le
direttive per cella non hanno una traduzione fedele in una policy globale, e
una traduzione approssimata produrrebbe una "riesecuzione" che non riesegue
niente. `ReplayProposer` lo dice con un errore chiaro invece di degradare.

Nel JSON i pilastri compaiono per NOME e non per indice: il registro e' letto
da persone oltre che dalla riesecuzione, e `PILLAR_BY_NAME` e' l'unica fonte
della corrispondenza nei due sensi.
"""

import json
from pathlib import Path

from src.governors.policy import (
    Condition,
    INDICATORS,
    OPS,
    PILLAR_BY_NAME,
    PILLAR_NAME_BY_INDEX,
    Policy,
    Rule,
)
from src.governors.observation import picture_digest


def _policy_to_json(policy: Policy | None) -> dict | None:
    if policy is None:
        return None
    return {
        "rationale": policy.rationale,
        "rules": [
            {
                "if": (
                    None
                    if rule.condition is None
                    else {
                        "indicator": rule.condition.indicator,
                        "op": rule.condition.op,
                        "value": rule.condition.value,
                    }
                ),
                "weights": {
                    PILLAR_NAME_BY_INDEX[pillar]: value
                    for pillar, value in sorted(rule.weights.items())
                },
            }
            for rule in policy.rules
        ],
    }


def _policy_from_json(payload) -> Policy | None:
    """Il giro inverso di `_policy_to_json`, VALIDATO.

    Il registro puo' essere stato scritto a mano (la verifica di manipolazione
    lo fa apposta), modificato, o venire da una versione futura con un
    vocabolario piu' ricco. Un indicatore fuori vocabolario che passasse da qui
    esploderebbe DENTRO `kernel.step`, al primo passo in cui la policy viene
    valutata — con un `KeyError` a meta' run e una diagnosi lontana dalla
    causa. Meglio un `ValueError` che nomina la voce, al caricamento.
    """
    if payload is None:
        return None
    rules = []
    for entry in payload.get("rules", []):
        raw_condition = entry.get("if")
        if raw_condition is None:
            condition = None
        else:
            indicator = str(raw_condition["indicator"])
            op = str(raw_condition["op"])
            if indicator not in INDICATORS:
                raise ValueError(
                    f"il registro nomina l'indicatore {indicator!r}, fuori dal "
                    f"vocabolario ({', '.join(sorted(INDICATORS))}): registro "
                    "modificato o scritto da un'altra versione"
                )
            if op not in OPS:
                raise ValueError(
                    f"il registro usa l'operatore {op!r}: valgono solo {OPS}"
                )
            condition = Condition(indicator, op, float(raw_condition["value"]))
        weights = {}
        for name, value in (entry.get("weights") or {}).items():
            pillar = PILLAR_BY_NAME.get(str(name))
            if pillar is None:
                raise ValueError(
                    f"il registro pesa il pilastro {name!r}, fuori dal "
                    f"vocabolario ({', '.join(sorted(PILLAR_BY_NAME))})"
                )
            weights[pillar] = float(value)
        rules.append(Rule(condition, weights))
    return Policy(tuple(rules), str(payload.get("rationale", "")))


class GovernorRecorder:
    """Scrive una riga JSON per tick, in ordine."""

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # **Il registro appartiene a UNA run.** Aperto in aggiunta, una run
        # rieseguita nella stessa cartella --- che e' quello che fa la ripresa
        # di una campagna dopo un guasto --- accoda le proprie righe a quelle
        # della run precedente, e il file finisce per raccontare due storie
        # sovrapposte con lo stesso numero di tick. Misurato il 2026-09-03: 78
        # tornate in un registro che ne poteva contenere 40. Le righe si
        # scrivono comunque una per tick e vengono svuotate subito, quindi una
        # run interrotta lascia il suo parziale: si perde solo il parziale
        # della run che si sta sostituendo, che e' proprio cio' che si vuole.
        self._handle = self._path.open("w", encoding="utf-8")
        self._last: dict | None = None

    def record(
        self, tick, application_step, picture, proposal, policy, missed,
        miss_reason: str = "", waited_s: float = 0.0,
    ) -> None:
        row = {
            "tick": int(tick),
            "application_step": int(application_step),
            "picture_step": int(picture.step),
            "picture_digest": picture_digest(picture),
            "missed": bool(missed),
            "governor": (
                None
                if proposal is None
                else {
                    "provider": proposal.provider,
                    "model": proposal.model,
                    "rationale": proposal.rationale,
                    "tokens_in": int(proposal.tokens_in),
                    "tokens_out": int(proposal.tokens_out),
                    "cost": float(proposal.cost),
                    "latency_s": float(proposal.latency_s),
                    "drops": vars(proposal.drops),
                    "proposal": _policy_to_json(proposal.policy),
                }
            ),
            "policy": _policy_to_json(policy),
        }
        # La risposta grezza, solo quando esiste (bracci a testo): e' l'unico
        # modo di sapere, dopo, che cosa il parser ha scartato e perche'.
        grezza = getattr(proposal, "raw_policy", None) if proposal is not None else None
        if grezza is not None:
            row["governor"]["proposal_raw"] = grezza
        semantic = getattr(proposal, "semantic_decision", None) if proposal is not None else None
        if semantic is not None:
            row["governor"]["semantic_decision"] = semantic
            row["governor"]["candidate_profile"] = str(
                getattr(proposal, "candidate_profile", "") or ""
            )
            tutte = getattr(proposal, "semantic_decisions", None)
            if tutte:
                row["governor"]["semantic_decisions"] = list(tutte)
        # Presenti solo quando dicono qualcosa: una riga riuscita non porta un
        # motivo di fallimento vuoto.
        if miss_reason:
            row["miss_reason"] = str(miss_reason)
        if waited_s:
            row["waited_s"] = float(waited_s)
        # Una riga per tick, scaricata subito: una run interrotta a meta' deve
        # lasciare rieseguibile la parte gia' decisa, non un file vuoto.
        #
        # `ensure_ascii` resta al suo default (scappa tutto in \uXXXX) e non e'
        # una scelta di gusto: la `rationale` e' testo del modello, e U+2028,
        # U+2029 e U+0085 sono terminatori di riga sopra 0x20, quindi nessuno li
        # scappa d'ufficio ma i lettori ci spezzano sopra. Uno solo di quei
        # caratteri trasformerebbe un tick in due righe monche e farebbe fallire
        # la rilettura dell'INTERO registro.
        self._handle.write(json.dumps(row) + "\n")
        self._handle.flush()
        self._last = row

    def last_summary(self) -> dict | None:
        """L'ultimo tick in forma trasportabile, per la vista live.

        `None` finche' nessun tick e' stato scritto: "nessuna decisione ancora"
        e "decisione vuota" sono cose diverse e vanno viste diverse.
        """
        if self._last is None:
            return None
        governor = self._last.get("governor") or {}
        proposal = governor.get("proposal") or {}
        return {
            "tick": self._last["tick"],
            "application_step": self._last["application_step"],
            "picture_step": self._last["picture_step"],
            "missed": self._last["missed"],
            "rationale": governor.get("rationale", ""),
            "tokens_out": governor.get("tokens_out", 0),
            "latency_s": governor.get("latency_s", 0.0),
            "rules_proposed": len(proposal.get("rules") or []),
            "rules_in_force": len((self._last.get("policy") or {}).get("rules") or []),
        }

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


def load_records(path) -> list[dict]:
    """Le righe del registro, in ordine di tick.

    Divide su `\\n` e non con `splitlines()`: quest'ultimo spezza anche su
    U+2028, U+2029 e U+0085, che dentro una stringa JSON sono caratteri
    legittimi.
    """
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return sorted(
        (json.loads(line) for line in lines if line.strip()),
        key=lambda row: int(row["tick"]),
    )


class ReplayProposer:
    """Restituisce le policy registrate, in ordine, senza rete.

    Esaurito il registro propone il nulla invece di inventare: una run piu'
    lunga di quella registrata deve degradare in modo visibile, non simulare
    una policy che nessuno aveva prodotto.
    """

    def __init__(self, path) -> None:
        self._rows = load_records(path)
        for row in self._rows:
            if "directive" in row or "governors" in row:
                raise ValueError(
                    f"il registro {path} e' nel formato del consiglio "
                    "(direttive per cella), precedente al ridisegno a "
                    "governatore unico del 2026-08-24: non e' rieseguibile con "
                    "questo codice. Usare il commit che lo ha scritto."
                )
        self._next = 0

    async def propose(self, picture, bounds):
        from src.governors.arms import GovernorProposal

        del bounds
        if self._next >= len(self._rows):
            return GovernorProposal(policy=Policy(), provider="replay", model="replay")
        row = self._rows[self._next]
        self._next += 1
        policy = _policy_from_json(row.get("policy"))
        return GovernorProposal(
            policy=policy if policy is not None else Policy(),
            rationale=str((row.get("policy") or {}).get("rationale", "")),
            provider="replay",
            model="replay",
        )
