from __future__ import annotations

import threading
from dataclasses import dataclass, field

#: I governatori chiamano dal PROPRIO thread (il consiglio possiede un loop
#: asyncio su un thread dedicato), mentre gli agenti chiamano dal thread della
#: shell. `record` fa una decina di letture-modifiche-scritture su interi e
#: liste, e nessuna di quelle e' atomica: senza lock due chiamate concorrenti
#: perdono conteggi in modo silenzioso, cioe' proprio nel dato che serve a dire
#: quanto e' costato un esperimento.
#:
#: Il lock e' di modulo e non un campo: `CostTracker` e' una dataclass che
#: qualcuno potrebbe voler copiare o serializzare verso un processo figlio, e un
#: `threading.Lock` come attributo lo renderebbe non picklabile. La contesa e'
#: nulla — poche chiamate per tick, ciascuna preceduta da una richiesta di rete.
_RECORD_LOCK = threading.Lock()


@dataclass
class CostTracker:
    max_calls_per_run: int = 0
    max_calls_per_day: int = 0
    max_estimated_cost_usd: float = 0.0
    calls: int = 0
    calls_today: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_total: int = 0
    estimated_cost_usd: float = 0.0
    by_provider_model: dict[str, dict] = field(default_factory=dict)
    call_records: list[dict] = field(default_factory=list)

    def can_call(self) -> bool:
        return (
            (self.max_calls_per_run <= 0 or self.calls < self.max_calls_per_run)
            and (self.max_calls_per_day <= 0 or self.calls_today < self.max_calls_per_day)
            and (self.max_estimated_cost_usd <= 0 or self.estimated_cost_usd <= self.max_estimated_cost_usd)
        )

    def record(
        self,
        cost_usd: float = 0.0,
        success: bool = True,
        provider: str = "",
        model: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_total: int = 0,
        pricing_source: str = "",
    ) -> None:
        with _RECORD_LOCK:
            self._record_locked(
                cost_usd, success, provider, model, tokens_in, tokens_out, tokens_total, pricing_source
            )

    def _record_locked(
        self,
        cost_usd: float,
        success: bool,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        tokens_total: int,
        pricing_source: str,
    ) -> None:
        self.calls += 1
        self.calls_today += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        total = tokens_total or tokens_in + tokens_out
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.tokens_total += total
        self.estimated_cost_usd += cost_usd
        key = f"{provider}:{model}" if provider and model else provider or "unknown"
        bucket = self.by_provider_model.setdefault(
            key,
            {
                "provider": provider,
                "model": model,
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "tokens_total": 0,
                "estimated_cost_usd": 0.0,
                "pricing_source": pricing_source,
            },
        )
        bucket["calls"] += 1
        bucket["successful_calls" if success else "failed_calls"] += 1
        bucket["tokens_in"] += tokens_in
        bucket["tokens_out"] += tokens_out
        bucket["tokens_total"] += total
        bucket["estimated_cost_usd"] += cost_usd
        if pricing_source:
            bucket["pricing_source"] = pricing_source
        self.call_records.append(
            {
                "call_index": self.calls,
                "provider": provider,
                "model": model,
                "success": success,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tokens_total": total,
                "estimated_cost_usd": cost_usd,
                "pricing_source": pricing_source,
            }
        )

    def new_day(self) -> None:
        self.calls_today = 0

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "calls_today": self.calls_today,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_total,
            "estimated_cost_usd": self.estimated_cost_usd,
            "by_provider_model": self.by_provider_model,
            "max_calls_per_run": self.max_calls_per_run,
            "max_calls_per_day": self.max_calls_per_day,
            "max_estimated_cost_usd": self.max_estimated_cost_usd,
        }
