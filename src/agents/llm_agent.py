from __future__ import annotations

from src.agents.action_space import ActionRequest, ActionType, valid_actions_for
from src.agents.rule_based_agent import RuleBasedAgent
from src.llm.prompt_builder import build_agent_decision_prompt
from src.llm.provider import FallbackProvider
from src.llm.structured_output import parse_decision_json


class LLMAgent(RuleBasedAgent):
    def __init__(self, *args, provider=None, cost_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "llm"
        self.provider = provider or FallbackProvider()
        self.cost_tracker = cost_tracker
        self.decision_history: list[dict] = []
        self.llm_provider_id = getattr(self.provider, "provider_id", "fallback")
        self.llm_model = getattr(self.provider, "model", "fallback")
        self._last_api_call_step: int | None = None

    def _tree_decide(self, observation, world) -> ActionRequest:
        """La policy ad albero, cioe' il comportamento storico del fallback."""
        return super().decide(observation, world)

    def _preference_decide(self, observation, world) -> ActionRequest:
        """La policy a preferenze, quella dei pari non-LLM in modalita' preferenze.

        Costruisce le maschere di cella per un solo agente, cioe' la via lenta:
        e' voluto. Questo percorso si esegue solo per un agente LLM che ha
        saltato la chiamata -- raro per costruzione -- e qui la correttezza vale
        piu' della velocita'. Se un giorno gli agenti LLM diventassero numerosi,
        il fallback andrebbe servito dal batch di fase B invece che ricalcolato.
        """
        from src.agents.preference_agent import decide_preferences
        from src.core.cell_action_mask import compute_cell_masks

        # `is not None` e non `or`: la scelta del ramo deve dipendere
        # dall'ESISTENZA dello stato ad array, non dalla sua verita'. Oggi
        # `CellArrays` non definisce ne' `__len__` ne' `__bool__` e quindi e'
        # sempre vero, ma quella garanzia poggia sull'assenza di un metodo. E il
        # ramo scavalcato non e' un'alternativa funzionante: `WorldView.cells`
        # e' una `_LazyCellGrid`, che non espone ne' `H` ne' le colonne che
        # `compute_cell_masks` legge.
        cells = getattr(world, "_cells", None)
        if cells is None:
            # Motore a oggetti (`GridWorld`): nessuno stato ad array, e nessuna
            # policy a preferenze eseguita da nessuno. `GridWorld.cells` e' una
            # lista di liste di `Cell`, che `compute_cell_masks` non sa leggere,
            # ma ricostruire le maschere da li' sarebbe comunque la risposta
            # sbagliata: in quella shell i pari decidono ad albero, quindi la
            # coerenza con loro -- l'unica cosa che questo metodo difende -- si
            # ottiene tornando all'albero.
            return self._tree_decide(observation, world)
        masks = compute_cell_masks(cells)
        cell = world.get_cell(self.x, self.y)
        return decide_preferences(
            self,
            world,
            masks[cell.y, cell.x],
            0.5,
            "greedy",
            {},
            int(getattr(world, "step", 0)),
        )

    def _fallback_decide(self, observation, world) -> ActionRequest:
        """La decisione da usare quando la chiamata non parte.

        Deve essere la STESSA policy dei pari dell'agente. Prima di questo fix era
        sempre quella ad albero, anche in modalita' preferenze, quindi un agente
        LLM che saltava la chiamata decideva con una politica diversa da quella di
        tutti gli altri e rendeva sporco il confronto fra popolazioni.
        """
        if getattr(self, "decision_mode", "") == "preferences":
            return self._preference_decide(observation, world)
        return self._tree_decide(observation, world)

    def decide(self, observation, world) -> ActionRequest:
        available = valid_actions_for(self, world)
        if self._last_api_call_step == world.step:
            world.log_event("llm_step_call_skipped", "LLM agent already called API during this simulation step", agent_id=self.agent_id, step=world.step)
            return self._fallback_decide(observation, world)
        if self.cost_tracker and not self.cost_tracker.can_call():
            world.log_event("llm_budget_limit_enforced", "LLM budget exhausted; falling back to deterministic rule-based decision", agent_id=self.agent_id)
            return self._fallback_decide(observation, world)
        prompt = build_agent_decision_prompt(self, observation, available, global_state=world.metrics(), current_step=world.step)
        world.log_event("llm_call_started", f"{self.name} calling {self.llm_provider_id}:{self.llm_model}", agent_id=self.agent_id, provider=self.llm_provider_id, model=self.llm_model)
        self._last_api_call_step = world.step
        response = self.provider.complete_json(prompt)
        world.log_event(
            "llm_call_completed",
            f"{self.name} received {response.provider} decision",
            agent_id=self.agent_id,
            provider=response.provider,
            api_call_attempted=response.api_call_attempted,
            error=response.error,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            tokens_total=response.tokens_total or response.tokens_in + response.tokens_out,
            estimated_cost_usd=response.cost_usd,
            pricing_source=response.pricing_source,
        )
        if self.cost_tracker and response.api_call_attempted:
            provider_id, model = _split_provider_model(response.provider, self.llm_provider_id, self.llm_model)
            self.cost_tracker.record(
                response.cost_usd,
                success=response.provider != "fallback",
                provider=provider_id,
                model=model,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                tokens_total=response.tokens_total,
                pricing_source=response.pricing_source,
            )
        data, errors = parse_decision_json(response.text)
        if not data or data["chosen_action"] not in available:
            self.memory.add_event(f"LLM decision rejected: {'; '.join(errors) or 'action unavailable'}")
            return self._fallback_decide(observation, world)
        data["llm_provider"] = response.provider
        self.decision_history.append(data)
        self.memory.add_event(data.get("thought_summary", "LLM decision made"))
        target = data.get("target")
        target = target if isinstance(target, dict) else {}
        return ActionRequest(self.agent_id, ActionType(data["chosen_action"]), target=target, message=data.get("public_message", ""))

    async def async_decide(self, observation, world) -> ActionRequest:
        available = valid_actions_for(self, world)
        if self._last_api_call_step == world.step:
            world.log_event("llm_step_call_skipped", "LLM agent already called API during this simulation step", agent_id=self.agent_id, step=world.step)
            return self._fallback_decide(observation, world)
        if self.cost_tracker and not self.cost_tracker.can_call():
            world.log_event("llm_budget_limit_enforced", "LLM budget exhausted; falling back to deterministic rule-based decision", agent_id=self.agent_id)
            return self._fallback_decide(observation, world)
        prompt = build_agent_decision_prompt(self, observation, available, global_state=world.metrics(), current_step=world.step)
        world.log_event("llm_call_started", f"{self.name} calling {self.llm_provider_id}:{self.llm_model} (async)", agent_id=self.agent_id, provider=self.llm_provider_id, model=self.llm_model)
        self._last_api_call_step = world.step
        response = await self.provider.async_complete_json(prompt)
        world.log_event(
            "llm_call_completed",
            f"{self.name} received {response.provider} decision (async)",
            agent_id=self.agent_id,
            provider=response.provider,
            api_call_attempted=response.api_call_attempted,
            error=response.error,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            tokens_total=response.tokens_total or response.tokens_in + response.tokens_out,
            estimated_cost_usd=response.cost_usd,
            pricing_source=response.pricing_source,
        )
        if self.cost_tracker and response.api_call_attempted:
            provider_id, model = _split_provider_model(response.provider, self.llm_provider_id, self.llm_model)
            self.cost_tracker.record(
                response.cost_usd,
                success=response.provider != "fallback",
                provider=provider_id,
                model=model,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                tokens_total=response.tokens_total,
                pricing_source=response.pricing_source,
            )
        data, errors = parse_decision_json(response.text)
        if not data or data["chosen_action"] not in available:
            self.memory.add_event(f"LLM decision rejected: {'; '.join(errors) or 'action unavailable'}")
            return self._fallback_decide(observation, world)
        data["llm_provider"] = response.provider
        self.decision_history.append(data)
        self.memory.add_event(data.get("thought_summary", "LLM decision made"))
        target = data.get("target")
        target = target if isinstance(target, dict) else {}
        return ActionRequest(self.agent_id, ActionType(data["chosen_action"]), target=target, message=data.get("public_message", ""))


def _split_provider_model(provider: str, fallback_provider: str, fallback_model: str) -> tuple[str, str]:
    if ":" in provider:
        left, right = provider.split(":", 1)
        return left or fallback_provider, right or fallback_model
    return provider or fallback_provider, fallback_model
