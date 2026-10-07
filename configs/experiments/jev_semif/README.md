# Configurazioni JEV/SemIf sperimentali

Questa cartella è separata dalle configurazioni storiche. I due YAML inclusi
usano esclusivamente provider offline:

- `offline_semif_fake.yaml`: decisione bounded fake e riproducibile;
- `offline_hybrid_fake.yaml`: seleziona esplicitamente `deep_reasoning`, ma il
  proposer profondo è `fallback`, quindi non contatta alcun modello.

Esempio sicuro:

```powershell
python scripts/run_governor_experiment.py --arms semif --seeds 101 --steps 4 --agents 8 --cadence 1 --wait 1 --semantic-provider fake --semantic-governor-decision select_candidate:mean_food_per_occupant --out runs/jev_semif_experiments/offline_semif_fake
```

I YAML sono caricabili anche tramite `src.experiments.runner.ExperimentRunner`,
lo stesso loader usato dalle configurazioni di test e dagli esperimenti salvati.

Il runner comparativo accetta inoltre `--arms semif hybrid` e scrive sotto
`runs/jev_semif_experiments/` quando `--out` non è specificato. `rest` richiede
sempre `--jev-endpoint`; le chiavi si indicano soltanto tramite il nome di una
variabile con `--jev-api-key-env`. Nessun endpoint viene scelto implicitamente.

Per il replay, usare il `semantic_decisions.jsonl` di una run completa e lo
stesso `run_id` registrato nel relativo `semantic_manifest.json`.

## Strato SemIf dei singoli agenti (dal 2026-09-22)

Tre configurazioni additive, tutte con `agents.decision_mode: preferences`
(lo strato lo richiede; il default del kernel e' `tree`):

- `offline_agents_fake.yaml`: provider fake, nessuna rete; ogni agente riceve
  `explore` al livello 1.
- `farm_agents_l1.yaml`: GPU farm privata tramite il gateway locale
  `semif-runtime` su `127.0.0.1:8008` (`VLLM_DECISION_API=chat`); solo livello 1,
  cadenza 4 passi. Costo zero.
- `typesafe_agents_smoke.yaml`: API ufficiale TypeSafe/Jev, **a pagamento**,
  chiave in `TYPESAFE_AI`, tetto cumulativo del registro `budget_usd`.

Dalla riga di comando lo strato si accende con `--semantic-agents all|hard`,
indipendentemente dal braccio di governo:

```powershell
python scripts/run_governor_experiment.py --arms none --seeds 101 --steps 4 --agents 8 --semantic-agents all --semantic-agents-cadence 1 --semantic-provider fake --semantic-agent-decision explore --semantic-run-id demo
```

Con `--semantic-agent-decision unknown` (default del fake) i fattori sono tutti
neutri e la run e' identica, sui file di stato, a quella senza strato. Il
registro `semantic_agent_decisions.jsonl` di una run si riproduce con
`--semantic-provider replay --semantic-replay-from <file> --semantic-run-id <stesso id>`.

TypeSafe: `--semantic-provider typesafe --typesafe-budget-usd <tetto>`; senza
tetto positivo la configurazione viene rifiutata e nessuna chiamata parte.

## Profili candidate di governatore e amministratori

`--semantic-candidate-profile v1|v2|v2h|v3|v4` (YAML `governors.semantic.candidate_profile`):

- `v1` (default): soglie alla media della colonia; inerte quando tutte le celle
  sono uguali.
- `v2`: stesse regole con soglie che coprono tutta la colonia quando le celle
  sono uguali, piu' la costituzione di riferimento; una sola domanda da 14
  opzioni.
- `v2h`: le candidate di v2 chieste in due livelli (area, poi candidata) e
  decise con la regola `is_decisive` (doppio del caso e 10 punti di margine
  sulla seconda). Il provider non taglia per confidenza. Vedi
  `docs/JEV_SEMIF_PHASE_6_10_DECISION_LOG.md`.
- `v3` (2026-09-23): due livelli come v2h, ma **vince l'opzione piu'
  probabile** (nessuna soglia, nessun distacco) perche' aspettare
  (`unknown`), non intervenire (`no_intervention`) e «nessuna di queste»
  (`none_of_the_above`) sono opzioni esplicite; ogni candidata dice condizione
  e portata attuale (nessuna / una parte / tutte le celle, cioe' una costante);
  la domanda dice quando NON intervenire; stato compatto con morti e legge in
  vigore; niente `power_coverage`. Stesse regole di v2h meno quella sulla
  corrente.
- `v4` (2026-09-24): v3 piu' i morti dall'ultima tornata per causa (colonia e
  distretto), la portata della legge del governo nel distretto e un criterio
  che lega il «problema» alle cause di morte.
