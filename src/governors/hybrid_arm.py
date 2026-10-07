"""Arm ibrido: SemIf bounded con unica escalation LLM esplicita."""

from __future__ import annotations

from src.governors.semif_arm import SemifGovernorProposer


class HybridGovernorProposer(SemifGovernorProposer):
    def __init__(
        self, semantic_provider, llm_proposer, *, run_id: str, candidate_factory=None
    ) -> None:
        if llm_proposer is None:
            raise ValueError("hybrid governor requires an explicit LLM proposer")
        super().__init__(
            semantic_provider,
            run_id=run_id,
            candidate_factory=candidate_factory,
            deep_reasoning_proposer=llm_proposer,
        )
