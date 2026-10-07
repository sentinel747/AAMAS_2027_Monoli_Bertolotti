from __future__ import annotations

"""Cio' che una shell deve sapere per innestare il governatore: la sezione
`governors` della configurazione, e l'adattamento delle metriche al quadro.

`arm: "none"` e' il default e disattiva del tutto il percorso. Non esiste una
combinazione in cui il governatore viene interrogato e la sua risposta
ignorata.

**Le chiavi del consiglio (`count`, `mandates`, `assignments` con piu' voci)
sono tollerate e ignorate con un avviso visibile**, non rifiutate: la funzione
"ripeti una run precedente" importa configurazioni intere, e una run governata
di prima del ridisegno deve poter essere ripetuta — sara' governata dal
governatore unico, e l'avviso dice esattamente questo invece di lasciarlo
scoprire dai risultati.

Le due shell (`AgentCoupledRunner`, `SimulationController`) hanno loop separati
e duplicati per scelta, ma quello che devono sapere del governatore sta qui una
volta sola: `build_governor` e `numeric_metrics`.
"""

from dataclasses import dataclass, field

from src.governors.arms import RandomProposer, ScriptedProposer
from src.governors.governor import Governor
from src.governors.context import normalizza as normalizza_contesto
from src.governors.districts import CELLE_PER_DISTRETTO
from src.governors.varianti_prompt import variante_da_sigla
from src.governors.policy import Bounds

VALID_ARMS = ("none", "random", "scripted", "llm", "semif", "hybrid")


def numeric_metrics(metrics) -> dict[str, float]:
    """Le sole metriche che il quadro sa rappresentare, cioe' i numeri.

    `ColonyPicture.metrics` e' dichiarato `dict[str, float]` e `build_picture`
    lo impone con `float(v)`, ma `_current_metrics` non promette affatto di
    restituire solo numeri: con lo strato planetario acceso ci finiscono dentro
    `unlocked_techs` (una lista) e tre stringhe di provenienza del clima.
    Passare il dizionario grezzo faceva uscire un `TypeError` da dentro il
    passo.

    Il filtro e' `float()` stesso e non un `isinstance`: dev'essere l'ESATTO
    inverso di cio' che rompe. Le stringhe vengono escluse prima della prova:
    `float("3.5")` riuscirebbe, e un campo testuale non deve diventare una
    metrica per il solo fatto di contenere cifre.
    """
    numeric: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, (str, bytes, bytearray)):
            continue
        try:
            numeric[str(key)] = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return numeric


@dataclass(frozen=True)
class GovernorSettings:
    cadence_steps: int
    arm: str
    bounds: Bounds
    #: L'assegnazione del braccio `llm`: provider, model, effort, temperature.
    assignment: dict = field(default_factory=dict)
    replay_from: str | None = None
    #: `0` = il governatore non fa mai attendere il passo. `> 0` = al confine
    #: di tick la simulazione attende la deliberazione, fino a questo limite.
    wait_seconds: float = 0.0
    #: Quanto contesto riceve il braccio `llm`: `completo`, `senza_aiuti`,
    #: `nomi_veri`, `cieco`. Vale solo per quel braccio, perche' gli altri non
    #: leggono un prompt. Vedi `src/governors/context.py`.
    context_level: str = "completo"
    #: Quale casella del disegno incrociato sui prompt: la sigla di
    #: `varianti_prompt.SIGLE`. `0` e' il prompt storico, parola per parola, ed
    #: e' il valore predefinito: una configurazione che non la nomina riceve il
    #: testo con cui l'archivio e' stato misurato.
    prompt_variant: str = "0"
    #: Vero quando la sezione chiede un governatore attivo.
    enabled: bool = False
    #: Configurazione esclusiva del provider SemIf. Nessun endpoint implicito.
    semantic: dict = field(default_factory=dict)


def read_settings(config: dict) -> GovernorSettings:
    """Le impostazioni, validate. Un disallineamento e' un errore, non una pezza."""
    raw = config.get("governors", {}) if isinstance(config.get("governors"), dict) else {}
    arm = str(raw.get("arm", "none")).strip().lower()
    if arm not in VALID_ARMS:
        raise ValueError(
            f"governors.arm={arm!r} non e' valido; scegliere fra {', '.join(VALID_ARMS)}"
        )

    # Le chiavi del consiglio, se presenti, vengono lette solo per due cose:
    # rispettare `count: 0` come interruttore spento (era la semantica di
    # allora) e avvisare che i mandati non esistono piu'.
    legacy_count = raw.get("count", None)
    enabled = arm != "none"
    if enabled and legacy_count is not None and int(legacy_count or 0) <= 0:
        enabled = False
    if enabled and (raw.get("mandates") or (legacy_count or 0) not in (None, 0, 1)):
        print(
            "[governatore] la configurazione porta chiavi del consiglio "
            "(count/mandates): il consiglio non esiste piu' dal 2026-08-24. "
            "La run usera' UN governatore senza mandato; della lista "
            "`assignments` conta solo la prima voce.",
            flush=True,
        )

    assignments = raw.get("assignments") or []
    assignment = dict(assignments[0]) if assignments else {}
    low, high = (raw.get("priority_multiplier_range") or [0.25, 4.0])[:2]
    return GovernorSettings(
        cadence_steps=max(1, int(raw.get("cadence_steps", 20) or 20)),
        arm=arm,
        bounds=Bounds(float(low), float(high)),
        assignment=assignment,
        replay_from=(str(raw["replay_from"]) if raw.get("replay_from") else None),
        wait_seconds=max(0.0, float(raw.get("wait_seconds", 0.0) or 0.0)),
        context_level=normalizza_contesto(raw.get("context_level")),
        prompt_variant=str(raw.get("prompt_variant", "0") or "0").strip().upper(),
        enabled=enabled,
        semantic=dict(raw.get("semantic") or {}),
    )


def _check_assignment(provider_id: str, model: str, registry: dict, consent: bool) -> None:
    """Ferma subito un'assegnazione che produrrebbe un governatore muto.

    I tre errori che questa funzione intercetta hanno in comune di NON dare
    errore a run: `LLMProposer` non solleva mai per progetto, e
    `ConfiguredLLMProvider` risponde con il fallback quando la chiave manca.
    Un provider inesistente, un modello non offerto o una chiave assente
    producono quindi 1500 passi di policy mai aggiornate, un registro pieno di
    "risposta non interpretabile" e risultati indistinguibili dalla baseline.
    Costano un errore adesso invece di una run intera dopo.
    """
    if provider_id not in registry:
        raise ValueError(
            f"governors.assignments indica il provider {provider_id!r}, che non "
            f"e' in configs/llm_providers.json. Provider disponibili: "
            f"{', '.join(sorted(registry)) or 'nessuno'}."
        )
    if not model:
        raise ValueError(
            f"governors.assignments non indica un modello per {provider_id!r}. "
            "Il modello e' la variabile indipendente dell'esperimento: "
            "lasciarlo scegliere al registro renderebbe il risultato non "
            "attribuibile."
        )
    missing_env = str(registry[provider_id].get("missing_env") or "")
    if consent and missing_env:
        raise ValueError(
            f"il provider {provider_id!r} richiede la variabile d'ambiente "
            f"{missing_env}, che non e' definita in questa shell. Con il "
            "consenso attivo ma la chiave assente il provider risponde con il "
            "fallback senza errori: la run sembrerebbe governata e non lo "
            "sarebbe."
        )


def directives_applied(days: int, cadence_steps: int, blocking: bool = False) -> int:
    """Quante policy entrano davvero in vigore in una run di `days` passi.

    I confini di tick cadono ai passi 1, 1+c, 1+2c...

    Nel regime NON bloccante la policy del tick `t` entra in vigore al confine
    SUCCESSIVO, `1 + (t+1)*c`: serve `1 + c <= days` perche' anche una sola sia
    applicata, e il conto e' `(days - 1) // c`.

    Nel regime bloccante entra in vigore al confine stesso, quindi ogni confine
    dentro la run produce una policy: `(days - 1) // c + 1`.
    """
    cadence = max(1, int(cadence_steps))
    boundaries = max(0, (int(days) - 1) // cadence)
    return boundaries + 1 if blocking and int(days) >= 1 else boundaries


def _build_llm_governor_proposer(settings, cost_tracker=None):
    """Costruisce il proposer storico; condiviso soltanto dall'arm ibrido."""
    from src.governors.llm_arm import LLMProposer
    from src.llm.provider_registry import (
        LLM_CONSENT_ENV,
        create_llm_provider,
        load_provider_registry,
        real_calls_allowed,
        resolve_offered_model,
    )

    registry = load_provider_registry()
    consent = real_calls_allowed()
    spec = settings.assignment
    provider_id = str(spec.get("provider") or "fallback")
    model = str(spec.get("model") or "")
    if provider_id != "fallback":
        _check_assignment(provider_id, model, registry, consent)
        model = resolve_offered_model(provider_id, registry[provider_id], model)
    provider = create_llm_provider(provider_id, model or None, registry)
    effort = str(spec.get("effort", "") or "").strip().lower()
    if effort and hasattr(provider, "options"):
        extra = dict(provider.options.get("extra_body") or {})
        if effort == "none":
            extra.pop("reasoning_effort", None)
        else:
            extra["reasoning_effort"] = effort
        provider.options["extra_body"] = extra
    pensiero = str(spec.get("thinking", "") or "").strip().lower()
    if pensiero and hasattr(provider, "options"):
        from src.llm.thinking import campi as campi_thinking, normalizza as norm_thinking

        provider.options["thinking"] = norm_thinking(pensiero)
        provider.thinking = provider.options["thinking"]
        provider._campi_thinking, avviso = campi_thinking(
            getattr(provider, "thinking_style", "none"), model, provider.thinking
        )
        if avviso:
            print(f"[governatore] {avviso}", flush=True)
    temperatura = spec.get("temperature", None)
    if temperatura is not None and hasattr(provider, "temperature"):
        provider.temperature = float(temperatura)
        if hasattr(provider, "options"):
            provider.options["temperature"] = float(temperatura)
    proposer = LLMProposer(
        provider, model, cost_tracker, livello_contesto=settings.context_level,
        variante=variante_da_sigla(settings.prompt_variant),
    )
    if not consent and provider_id != "fallback":
        print(
            f"[governatore] {LLM_CONSENT_ENV} assente: il governatore usa "
            "il fallback deterministico. Nessuna chiamata sara' fatta e la "
            "policy non verra' mai aggiornata.",
            flush=True,
        )
    return proposer


def _semantic_for_profile(semantic: dict) -> dict:
    """La sezione del provider per il profilo scelto.

    Con v2h la decisione la prende la regola `is_decisive` nel braccio, sulla
    distribuzione intera: il provider non deve tagliare per confidenza prima
    (lo stesso motivo per cui non lo fa lo strato degli agenti).
    """
    sezione = dict(semantic or {})
    # v3 (2026-09-23) decide a maggioranza relativa nel braccio: stesso motivo.
    if str(sezione.get("candidate_profile", "v1") or "v1").strip().lower() in ("v2h", "v3", "v4"):
        sezione["min_confidence"] = 0.0
    return sezione


def _candidate_profiles(semantic: dict) -> tuple[str, str]:
    """Versioni (governatore, amministratore) del profilo candidate scelto.

    Default `v1`: le run SemIf gia' prodotte restano riproducibili. `v2`
    (2026-09-22) evita le candidate inerti quando tutte le celle sono uguali.
    """
    from src.governors.policy_candidates import PROFILE_VERSIONS

    profilo = str((semantic or {}).get("candidate_profile", "v1") or "v1").strip().lower()
    if profilo not in PROFILE_VERSIONS:
        raise ValueError(
            f"governors.semantic.candidate_profile must be one of {sorted(PROFILE_VERSIONS)}, "
            f"got {profilo!r}"
        )
    return PROFILE_VERSIONS[profilo]


def build_governor(
    config: dict,
    seed: int,
    recorder=None,
    cost_tracker=None,
    semantic_provider=None,
    hybrid_llm_proposer=None,
) -> Governor | None:
    """Il governatore, o `None` quando il percorso e' disattivato.

    `cost_tracker` e' quello della shell: le chiamate del governatore devono
    finire nella stessa contabilita' di quelle degli agenti, altrimenti
    `api_usage.json` riporta zero per una run che ha chiamato.
    """
    settings = read_settings(config)
    if not settings.enabled:
        return None

    # **La cadenza va confrontata con la lunghezza della run, non solo con la
    # latenza.** Una cadenza piu' lunga della run lascia un solo confine di
    # tick — il passo 1 — e la policy di quel tick entrerebbe in vigore dopo la
    # fine: il governatore avvia le chiamate, non raccoglie mai, e la run
    # coincide con la baseline. Succede senza un solo errore, e
    # `governor_misses` resta a zero perche' i mancati aggiornamenti si contano
    # ai confini di tick, che non arrivano.
    days = config.get("days")
    if days:
        applied = directives_applied(
            int(days), settings.cadence_steps, blocking=settings.wait_seconds > 0.0
        )
        if settings.wait_seconds > 0.0:
            # Il prezzo del regime bloccante e' tempo di orologio, ed e'
            # prevedibile: va detto PRIMA della run, non scoperto durante.
            print(
                f"[governatore] regime bloccante: {applied} confini di tick da "
                f"attendere, fino a {settings.wait_seconds:.0f} s ciascuno "
                f"(al peggio {applied * settings.wait_seconds / 60.0:.0f} min "
                "oltre il tempo di simulazione)",
                flush=True,
            )
        if applied == 0:
            raise ValueError(
                f"governors.cadence_steps={settings.cadence_steps} su una run di "
                f"{int(days)} passi non applica NESSUNA policy: il primo confine "
                f"di tick e' il passo 1 e la sua policy entrerebbe in vigore al "
                f"passo {1 + settings.cadence_steps}. Serve cadence_steps <= "
                f"{max(1, int(days) - 1)}. Se la cadenza deriva dalla latenza del "
                "provider, il provider e' troppo lento per questa lunghezza di run: "
                "vedi docs/MARS_ABM_GOVERNATORI_LLM_ESECUZIONE.md."
            )

    if settings.replay_from:
        from src.governors.record import ReplayProposer

        proposer = ReplayProposer(settings.replay_from)
    elif settings.arm == "random":
        proposer = RandomProposer(seed=int(seed))
    elif settings.arm == "scripted":
        proposer = ScriptedProposer()
    elif settings.arm in {"semif", "hybrid"}:
        from src.governors.semif_arm import SemifGovernorProposer
        from src.semantic_governance.factory import build_semantic_provider

        from src.governors.policy_candidates import CandidatePolicyFactory

        profilo_governatore, _ = _candidate_profiles(settings.semantic)
        fabbrica_candidate = CandidatePolicyFactory(profilo_governatore)
        provider = semantic_provider or build_semantic_provider(
            _semantic_for_profile(settings.semantic)
        )
        run_id = str(settings.semantic.get("run_id") or f"semif-seed-{int(seed)}")
        if settings.arm == "hybrid":
            from src.governors.hybrid_arm import HybridGovernorProposer

            llm_proposer = hybrid_llm_proposer or _build_llm_governor_proposer(
                settings, cost_tracker
            )
            proposer = HybridGovernorProposer(
                provider, llm_proposer, run_id=run_id, candidate_factory=fabbrica_candidate
            )
        else:
            proposer = SemifGovernorProposer(
                provider, run_id=run_id, candidate_factory=fabbrica_candidate
            )
    else:
        from src.governors.llm_arm import LLMProposer
        from src.llm.provider_registry import (
            LLM_CONSENT_ENV,
            create_llm_provider,
            load_provider_registry,
            real_calls_allowed,
            resolve_offered_model,
        )

        registry = load_provider_registry()
        consent = real_calls_allowed()
        spec = settings.assignment
        provider_id = str(spec.get("provider") or "fallback")
        model = str(spec.get("model") or "")
        if provider_id != "fallback":
            _check_assignment(provider_id, model, registry, consent)
            model = resolve_offered_model(provider_id, registry[provider_id], model)
        # `create_llm_provider` applica l'interlock di consenso: senza
        # `MARSABM_ALLOW_LLM_CALLS` restituisce il fallback deterministico.
        provider = create_llm_provider(provider_id, model or None, registry)
        # **Lo sforzo di ragionamento va sovrascritto qui o non serve a
        # niente.** Il preflight lo accetta come opzione (`--effort`) e misura
        # una latenza; se la run non puo' riceverlo, la run gira con lo sforzo
        # del registro e la latenza misurata non e' quella della run.
        effort = str(spec.get("effort", "") or "").strip().lower()
        if effort and hasattr(provider, "options"):
            extra = dict(provider.options.get("extra_body") or {})
            if effort == "none":
                extra.pop("reasoning_effort", None)
            else:
                extra["reasoning_effort"] = effort
            provider.options["extra_body"] = extra
        # **La temperatura decide se ripetere lo stesso seme misuri qualcosa.**
        # Il confronto appaiato presuppone che fissare il seme fissi tutto
        # tranne il trattamento: vero per i bracci di controllo, che sono
        # deterministici, falso per questo.
        #
        # `is not None` e non `if temperatura`: **zero e' il valore che
        # interessa** e sarebbe falsy. E si scrive l'ATTRIBUTO, non solo il
        # dizionario: `ConfiguredLLMProvider` legge `options["temperature"]`
        # una volta nel costruttore e poi manda `self.temperature`.
        # **Il ragionamento va sovrascritto qui o resta quello del registro.**
        # Stessa ragione dello sforzo: il preflight lo misura come opzione, e
        # una run che non potesse riceverlo girerebbe con un'altra latenza e
        # un'altra varianza rispetto a quella misurata. Si riscrive l'attributo
        # gia' risolto e non solo il dizionario, perche' il provider traduce i
        # campi una volta nel costruttore.
        pensiero = str(spec.get("thinking", "") or "").strip().lower()
        if pensiero and hasattr(provider, "options"):
            from src.llm.thinking import campi as campi_thinking, normalizza as norm_thinking

            provider.options["thinking"] = norm_thinking(pensiero)
            provider.thinking = provider.options["thinking"]
            provider._campi_thinking, avviso = campi_thinking(
                getattr(provider, "thinking_style", "none"), model, provider.thinking
            )
            if avviso:
                print(f"[governatore] {avviso}", flush=True)
        temperatura = spec.get("temperature", None)
        if temperatura is not None and hasattr(provider, "temperature"):
            provider.temperature = float(temperatura)
            if hasattr(provider, "options"):
                provider.options["temperature"] = float(temperatura)
        proposer = LLMProposer(
            provider, model, cost_tracker, livello_contesto=settings.context_level,
            variante=variante_da_sigla(settings.prompt_variant),
        )

        if not consent and provider_id != "fallback":
            # Non e' un errore — e' il modo in cui questo repository esegue il
            # braccio `llm` senza consumare credito — ma dev'essere VISIBILE.
            # Silenzioso, produce una run che si chiama `llm`, non chiama
            # nessuno, e finisce identica alla baseline.
            print(
                f"[governatore] {LLM_CONSENT_ENV} assente: il governatore usa "
                "il fallback deterministico. Nessuna chiamata sara' fatta e la "
                "policy non verra' mai aggiornata.",
                flush=True,
            )

    return Governor(
        proposer,
        settings.bounds,
        settings.cadence_steps,
        recorder=recorder,
        wait_seconds=settings.wait_seconds,
    )


#: Gli arm amministrativi selezionabili. SemIf e hybrid sono stati aggiunti
#: soltanto dopo il gate della Fase 2, senza cambiare i quattro percorsi storici.
VALID_ADMIN_ARMS = ("none", "random", "scripted", "llm", "semif", "hybrid")


@dataclass(frozen=True)
class AdministratorSettings:
    """Lo strato fra il governo e le celle."""

    #: L'interruttore chiesto dal disegno: una run con o senza amministratori.
    enabled: bool = False
    #: Quante celle tiene un amministratore. La cella madre non conta: resta
    #: sotto il governo diretto.
    cells_per_district: int = CELLE_PER_DISTRETTO
    #: Vero: gli amministratori usano lo stesso braccio e lo stesso livello di
    #: contesto del governo. Falso: usano il braccio e il modello dichiarati
    #: qui, e vedono il quadro "normale" del proprio distretto.
    follow_governor_arm: bool = True
    #: Braccio proprio, quando non seguono quello del governo.
    arm: str = "llm"
    #: Provider/model/thinking proprio, quando non seguono quello del governo.
    assignment: dict = field(default_factory=dict)
    #: **Una lista di modelli, uno per distretto (2026-09-06).** Ogni distretto
    #: nuovo pesca dalla lista in modo riproducibile a parita' di seme, cosi' due
    #: run restano appaiate. E' la leva per studiare stili di governo diversi
    #: fra modelli; con la lista vuota vale la sola `assignment`.
    assignments: list = field(default_factory=list)
    #: Quanto del dominio vede il livello LOCALE. Fino al 2026-09-14 gli
    #: amministratori leggevano sempre il prompt completo, anche quando il
    #: governatore era cieco: il braccio misurava quanto del dominio vede il
    #: GOVERNO, non il sistema. Con questo campo il gradino si puo' applicare a
    #: tutti e due i livelli, che e' la casella che rende l'affermazione «la
    #: prudenza non scompare quando scompare il dominio» difendibile.
    context_level: str = "completo"
    #: Provider SemIf proprio, o ereditato dal governatore quando richiesto.
    semantic: dict = field(default_factory=dict)


def read_administrator_settings(config: dict, governor: GovernorSettings) -> AdministratorSettings:
    raw = config.get("governors", {}) if isinstance(config.get("governors"), dict) else {}
    sezione = raw.get("administrators")
    if not isinstance(sezione, dict):
        return AdministratorSettings()
    enabled = bool(sezione.get("enabled", False))
    # Una sezione spenta puo' restare in una configurazione storica senza
    # ereditare un arm nuovo che non esercita. Quando viene accesa, invece,
    # l'ereditarieta' e' reale e deve essere validata.
    inherited_default = governor.arm if enabled else "llm"
    arm = str(sezione.get("arm", inherited_default) or inherited_default).strip().lower()
    if arm not in VALID_ADMIN_ARMS:
        raise ValueError(
            f"governors.administrators.arm={arm!r} non e' valido; "
            f"scegliere fra {', '.join(VALID_ADMIN_ARMS)}"
        )
    lista = [dict(a) for a in (sezione.get("assignments") or []) if isinstance(a, dict) and a]
    # Una lista propria di modelli e' un braccio linguistico proprio: non si
    # eredita il modello del governo e non serve dichiararlo due volte.
    segue = bool(sezione.get("follow_governor_arm", True)) and not lista
    if lista:
        arm = "llm"
    singola = dict(sezione.get("assignment") or (governor.assignment if segue else {}))
    if not lista and singola:
        lista = [singola]
    return AdministratorSettings(
        # Senza dichiararlo, gli amministratori vedono il prompt completo: e' il
        # comportamento di tutte le campagne gia' in archivio e resta il default.
        context_level=normalizza_contesto(sezione.get("context_level", "completo")),
        enabled=enabled,
        cells_per_district=max(1, int(sezione.get("cells_per_district", CELLE_PER_DISTRETTO) or CELLE_PER_DISTRETTO)),
        follow_governor_arm=segue,
        arm=governor.arm if segue else arm,
        assignment=lista[0] if lista else singola,
        assignments=lista,
        semantic=dict(
            governor.semantic if segue else (sezione.get("semantic") or {})
        ),
    )


def build_administration(
    config: dict,
    seed: int,
    cost_tracker=None,
    semantic_provider_factory=None,
    hybrid_llm_factory=None,
):
    """Lo strato amministrativo, o `None` quando l'interruttore e' spento.

    **Il braccio dell'amministratore non e' mai piu' potente di quello del
    governo per caso.** Quando `follow_governor_arm` e' vero eredita braccio,
    provider, modello e livello di ragionamento: un decentramento che parlasse
    a un modello migliore misurerebbe il modello, non il decentramento.
    """
    from src.governors.administration import Amministrazione
    from src.governors.admin_arms import (
        AmministratoreAssente,
        AmministratoreCasuale,
        AmministratoreLLM,
        AmministratoreScritto,
    )
    from src.governors.districts import Distretti

    governatore = read_settings(config)
    impostazioni = read_administrator_settings(config, governatore)
    if not impostazioni.enabled:
        return None
    if (
        impostazioni.arm in {"semif", "hybrid"}
        and str((impostazioni.semantic or {}).get("candidate_profile", "v1") or "v1")
        .strip().lower() in ("v2h", "v3", "v4")
        and governatore.arm == "none"
    ):
        # Con v2h gli amministratori deliberano solo su una legge del governo:
        # senza governo non delibererebbero mai, cioe' un braccio inerte in
        # silenzio. Il decentramento puro resta disponibile con v1 e v2.
        raise ValueError(
            "administrators with candidate_profile v2h need a governor: they only "
            "adapt a governor law; use v1 or v2 for administrators without governor"
        )

    distretti = Distretti(impostazioni.cells_per_district)

    if impostazioni.arm == "none":
        def fabbrica(_distretto):
            return AmministratoreAssente()
    elif impostazioni.arm == "random":
        def fabbrica(distretto, _b=governatore.bounds, _s=seed):
            return AmministratoreCasuale(_s, distretto, _b)
    elif impostazioni.arm == "scripted":
        def fabbrica(_distretto, _b=governatore.bounds):
            return AmministratoreScritto(_b)
    elif impostazioni.arm == "semif":
        from src.governors.admin_semif_arm import AmministratoreSemIf
        from src.semantic_governance.factory import build_semantic_provider

        run_id = str(impostazioni.semantic.get("run_id") or f"semif-seed-{int(seed)}")
        _, profilo_amministratore = _candidate_profiles(impostazioni.semantic)

        def fabbrica(distretto, _b=governatore.bounds, _run=run_id,
                     _profilo=profilo_amministratore):
            provider = (
                semantic_provider_factory(distretto)
                if semantic_provider_factory is not None
                else build_semantic_provider(_semantic_for_profile(impostazioni.semantic))
            )
            return AmministratoreSemIf(
                provider, _b, run_id=_run, district=distretto,
                candidate_profile=_profilo,
            )
    elif impostazioni.arm == "hybrid":
        from src.governors.admin_semif_arm import AmministratoreIbrido
        from src.governors.admin_arms import AmministratoreLLM
        from src.llm.provider_registry import (
            LLM_CONSENT_ENV,
            create_llm_provider,
            load_provider_registry,
            real_calls_allowed,
            resolve_offered_model,
        )
        from src.semantic_governance.factory import build_semantic_provider

        registry = load_provider_registry()
        consenso = real_calls_allowed()
        scelte: list[tuple[str, str]] = []
        for spec in impostazioni.assignments or [impostazioni.assignment]:
            provider_id = str(spec.get("provider") or "fallback")
            model = str(spec.get("model") or "")
            if provider_id != "fallback":
                _check_assignment(provider_id, model, registry, consenso)
                model = resolve_offered_model(provider_id, registry[provider_id], model)
            scelte.append((provider_id, model))
        if not consenso and any(p != "fallback" for p, _ in scelte):
            print(
                f"[amministratori] {LLM_CONSENT_ENV} assente: il ramo profondo "
                "ibrido usa il fallback deterministico e si asterra' sempre.",
                flush=True,
            )
        run_id = str(impostazioni.semantic.get("run_id") or f"semif-seed-{int(seed)}")
        _, profilo_amministratore = _candidate_profiles(impostazioni.semantic)

        def fabbrica(
            distretto, _b=governatore.bounds, _run=run_id, _s=scelte,
            _r=registry, _c=cost_tracker, _seed=seed, _profilo=profilo_amministratore,
        ):
            semantic = (
                semantic_provider_factory(distretto)
                if semantic_provider_factory is not None
                else build_semantic_provider(_semantic_for_profile(impostazioni.semantic))
            )
            if hybrid_llm_factory is not None:
                llm = hybrid_llm_factory(distretto)
            else:
                import random

                indice = (
                    0 if len(_s) == 1
                    else random.Random(f"{_seed}:{distretto}").randrange(len(_s))
                )
                provider_id, model = _s[indice]
                llm = AmministratoreLLM(
                    create_llm_provider(provider_id, model or None, _r), model, _c
                )
            return AmministratoreIbrido(
                semantic, llm, _b, run_id=_run, district=distretto,
                candidate_profile=_profilo,
            )

        modelli = [f"{p}:{m}" for p, m in scelte]
    else:
        from src.llm.provider_registry import (
            LLM_CONSENT_ENV,
            create_llm_provider,
            load_provider_registry,
            real_calls_allowed,
            resolve_offered_model,
        )

        import random

        registry = load_provider_registry()
        consenso = real_calls_allowed()
        # Una voce per modello: ogni distretto ne pesca una. Le si verificano
        # tutte PRIMA della run, per la stessa ragione di `_check_assignment`:
        # un modello inesistente nella lista darebbe un distretto muto ogni N.
        scelte: list[tuple[str, str]] = []
        for spec in impostazioni.assignments or [impostazioni.assignment]:
            provider_id = str(spec.get("provider") or "fallback")
            model = str(spec.get("model") or "")
            if provider_id != "fallback":
                _check_assignment(provider_id, model, registry, consenso)
                model = resolve_offered_model(provider_id, registry[provider_id], model)
            scelte.append((provider_id, model))
        if not consenso and any(p != "fallback" for p, _ in scelte):
            print(
                f"[amministratori] {LLM_CONSENT_ENV} assente: lo strato usa il "
                "fallback deterministico e si asterra' sempre.",
                flush=True,
            )

        def fabbrica(distretto, _s=scelte, _r=registry, _c=cost_tracker, _seed=seed):
            # Riproducibile a parita' di seme e distretto, indipendente
            # dall'ordine in cui i distretti compaiono nella run.
            indice = 0 if len(_s) == 1 else random.Random(f"{_seed}:{distretto}").randrange(len(_s))
            provider_id, model = _s[indice]
            return AmministratoreLLM(create_llm_provider(provider_id, model or None, _r), model, _c)

        modelli = [f"{p}:{m}" for p, m in scelte]

    if impostazioni.arm not in {"llm", "hybrid"}:
        modelli = []
    print(
        f"[amministratori] attivi: braccio {impostazioni.arm}, "
        f"{impostazioni.cells_per_district} celle per distretto, "
        f"{'stesso braccio del governo' if impostazioni.follow_governor_arm else 'braccio proprio'}"
        + (f", modelli per distretto: {', '.join(modelli)}" if len(modelli) > 1 else ""),
        flush=True,
    )
    # La variante del prompt e' quella del governo: il vocabolario dei due
    # livelli deve restare uno solo (vedi `costruisci_prompt`).
    return Amministrazione(
        fabbrica, distretti, governatore.bounds, attivo=True, modelli=modelli,
        variante=variante_da_sigla(governatore.prompt_variant),
        livello=impostazioni.context_level,
    )
