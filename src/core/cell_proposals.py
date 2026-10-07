from __future__ import annotations

"""Cell-level task proposals for the preference policy.

The cell converts its already-vectorized state and need matrix into a small
menu of feasible work.  It does not assign a specific colonist: each agent
still scores the menu with its own preferences, urgency and (later) skills.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.agents import pillars
from src.agents.action_space import ActionType, BUILD_ACTIONS
from src.agents.build_policy import (
    COLONISTS_PER_STRUCTURE,
    POPOLAZIONE_PER_DADO_DI_NATALITA,
    costruzione_satura,
)
from src.core import constants as C
from src.agents.preference_agent import RAZIONE_MINIMA
from src.simulation.extreme_events import (
    DUST_STORM_OPACITY_SURGE,
    SOLAR_FLARE_RADIATION_SURGE,
)
from src.simulation.step_effects import (
    CAPACITA_MANUTENZIONE,
    RIPARAZIONE_MANUTENZIONE,
)

#: Quante strutture di sviluppo devono mancare perche' il bisogno conti per
#: intero. Sotto questa soglia la priorita' scala in proporzione, cosi' un
#: avamposto piccolo non compete con la colonia madre per gli stessi slot di
#: menu: a dodici coloni ne manca una sola, a duecento ne mancano otto.
DEVELOPMENT_FULL_DEFICIT = 4.0
from src.world.occupancy import housing_slots_from_structures
from src.world.structures import StructureType


PERSONAL_ACTIONS = frozenset(
    {
        ActionType.DRINK_WATER,
        ActionType.REFILL_WATER,
        ActionType.EAT_FOOD,
        ActionType.REST,
        ActionType.USE_MED_KIT,
        ActionType.DO_NOTHING,
    }
)
ALWAYS_CELL_ACTIONS = frozenset({ActionType.MOVE, ActionType.OBSERVE})
CELL_ONLY_ACTIONS = frozenset({ActionType.PHYSIOLOGICAL_RECOVERY})
TASK_ACTIONS = tuple(
    action
    for action in pillars.ACTION_ORDER
    if action in pillars.ACTION_TO_PILLAR
    and action not in PERSONAL_ACTIONS
    and action not in ALWAYS_CELL_ACTIONS
    and action not in CELL_ONLY_ACTIONS
)


@dataclass(frozen=True)
class CellProposals:
    mask: np.ndarray
    priority: np.ndarray
    quota: np.ndarray


def explore_frontier(cells) -> np.ndarray:
    """``[H, W]``: True where a cell touches unexplored ground.

    This is the ONLY formula in this module that reads cells other than the one
    it scores - the four von Neumann neighbours, wrapping in x and clamping in
    y.  It is therefore the only one that cannot run on a subset of cells, and
    it is extracted so ``compute_cell_proposals`` can accept it precomputed on
    the full grid (see ``src/core/cell_subset.py``).
    """
    unknown = ~cells.explored
    frontier = np.roll(unknown, 1, axis=1) | np.roll(unknown, -1, axis=1)
    frontier[:-1, :] |= unknown[1:, :]
    frontier[1:, :] |= unknown[:-1, :]
    return frontier


def compute_cell_proposals(
    cells,
    base_masks: np.ndarray,
    needs: np.ndarray,
    *,
    top_k: int = 5,
    individual_survival_priority_enabled: bool = True,
    frontier: np.ndarray | None = None,
    priority_tilt: Callable[[np.ndarray], None] | None = None,
    development_priority: float = 0.30,
    pavimento_cantieri: float = 1.1,
) -> CellProposals:
    """Build ``[H,W,A]`` proposal mask, priority and per-step quota arrays.

    ``top_k`` limits cell work tasks only. With individual survival enabled,
    personal actions remain outside that cap. In cell-only mode they are
    replaced by one unlimited ``PHYSIOLOGICAL_RECOVERY`` proposal in every
    cell; the decision kernel keeps stable preferences and disables only
    personal vital-state urgency and physiological overrides.

    ``priority_tilt`` e' il punto d'innesto della direttiva dei governatori: una
    funzione che muta ``priority`` in posto subito prima del taglio top-k. Con
    ``None`` — il caso senza governatori — non viene invocata affatto, quindi la
    baseline resta bit-exact per costruzione e non per verifica.
    """

    masks = np.asarray(base_masks, dtype=np.bool_).copy()
    priority = np.zeros(masks.shape, dtype=np.float64)
    quota = np.zeros(masks.shape, dtype=np.int32)
    raw_occupancy = cells.occupancy.astype(np.int32)
    occupancy = np.maximum(raw_occupancy, 1)

    def idx(action: ActionType) -> int:
        return pillars.ACTION_INDEX[action]

    def set_priority(action: ActionType, values) -> None:
        priority[:, :, idx(action)] = np.where(
            masks[:, :, idx(action)], np.asarray(values, dtype=np.float64), 0.0
        )

    recovery_idx = idx(ActionType.PHYSIOLOGICAL_RECOVERY)
    if individual_survival_priority_enabled:
        # Personal actions are never centrally rationed. Their real
        # feasibility is still narrowed per agent by agent_action_mask(_batch).
        for action in PERSONAL_ACTIONS:
            set_priority(action, 1.0)
            quota[:, :, idx(action)] = -1
        masks[:, :, recovery_idx] = False
    else:
        for action in PERSONAL_ACTIONS:
            action_idx = idx(action)
            masks[:, :, action_idx] = False
            priority[:, :, action_idx] = 0.0
            quota[:, :, action_idx] = 0
        masks[:, :, recovery_idx] = True
        priority[:, :, recovery_idx] = 1.0
        quota[:, :, recovery_idx] = -1

    # Safe mobility is always proposed, independently from the work top-k.
    set_priority(ActionType.MOVE, 0.25)
    set_priority(ActionType.OBSERVE, 0.20)
    quota[:, :, idx(ActionType.MOVE)] = -1
    quota[:, :, idx(ActionType.OBSERVE)] = -1

    occ_f = occupancy.astype(np.float64)

    def normalized_need(name: str, scale: float) -> np.ndarray:
        return np.clip(needs[:, :, C.ND[name]] / np.maximum(1.0, occ_f * scale), 0.0, 1.0)

    water_p = 0.10 + 0.90 * normalized_need("water", 8.0)
    # A greenhouse is not a passive food printer: routine cultivation remains
    # useful even when the warehouse is currently stocked.  Its baseline must
    # therefore survive the mature-cell top-k menu, while scarcity can still
    # raise it to the same maximum priority as every other urgent task.
    food_p = 0.22 + 0.78 * normalized_need("food", 4.0)
    materials_p = 0.10 + 0.90 * normalized_need("materials", 4.0)
    set_priority(ActionType.COLLECT_ICE, water_p)
    set_priority(ActionType.FORAGE, food_p)
    set_priority(ActionType.COLLECT_MATERIALS, materials_p)
    set_priority(ActionType.COLLECT_MINERALS, 0.08 + 0.82 * materials_p)

    #: Quante strutture MANCANO, per tipo. Il modello lo calcola gia' per
    #: decidere la priorita'; da qui in poi lo usa anche per decidere quanti
    #: cantieri la cella puo' aprire. Vedi il blocco delle quote piu' sotto.
    deficit_cantieri: dict[ActionType, np.ndarray] = {}

    deficit_cantieri[ActionType.BUILD_SOLAR_ARRAY] = needs[:, :, C.ND["build_solar"]]
    set_priority(
        ActionType.BUILD_SOLAR_ARRAY,
        0.08 + 0.92 * np.clip(needs[:, :, C.ND["build_solar"]], 0.0, 1.0),
    )
    deficit_cantieri[ActionType.BUILD_OXYGEN_PLANT] = needs[
        :, :, C.ND["build_oxygen_plant"]
    ]
    set_priority(
        ActionType.BUILD_OXYGEN_PLANT,
        0.08 + 0.92 * np.clip(needs[:, :, C.ND["build_oxygen_plant"]], 0.0, 1.0),
    )
    deficit_cantieri[ActionType.BUILD_WATER_EXTRACTOR] = needs[:, :, C.ND["build_water"]]
    set_priority(
        ActionType.BUILD_WATER_EXTRACTOR,
        0.08 + 0.92 * np.clip(needs[:, :, C.ND["build_water"]], 0.0, 1.0),
    )
    greenhouse_priority = 0.08 + 0.92 * np.clip(
        needs[:, :, C.ND["build_greenhouse"]], 0.0, 1.0
    )
    # ``compute_needs`` intentionally publishes work only for settled cells.
    # That contract made a provisioned pioneer standing on virgin ground look
    # need-free, so BUILD_GREENHOUSE lost the top-k menu and an arriving
    # settlement expedition could never execute its first homestead.  The cell
    # can identify the exceptional transition without assigning an agent:
    # occupied + no structure + no open site means one shared greenhouse slot.
    # Personal affordability remains in agent_action_mask(_batch), preserving
    # preference ranking and preventing an unprovisioned colonist from binding
    # the proposal.
    # **La condizione e' «non ha ancora una serra», non «non ha nulla»
    # (2026-08-30).** Legarla all'assenza di QUALSIASI struttura la spegneva
    # appena il fondatore posava il primo pannello: misurato, un fondatore ben
    # equipaggiato arrivato su un avamposto gia' avviato costruiva ricovero e
    # impianto d'ossigeno e **mai la serra**, e senza serra
    # `local_life_support_capacity` resta ZERO — l'avamposto non puo' accogliere
    # nessuno e muore. Sei avamposti fondati, sei avamposti vuoti, tutti con
    # pannelli e nessuna serra.
    #
    # La ragione della regola non e' cambiata: un insediamento nuovo deve prima
    # assicurarsi cibo e acqua. E' la condizione che era piu' stretta del suo
    # scopo.
    # La stessa deroga vale per il pozzo, e per la stessa ragione: senza acqua
    # `local_life_support_capacity` resta ZERO e l'avamposto non puo' accogliere
    # nessuno. Erano due requisiti con lo stesso destino e una sola eccezione.
    senza_pozzo = cells.struct_count[:, :, C.S[StructureType.WATER_EXTRACTOR]] == 0
    cantiere_pozzo = cells.site_progress[:, :, C.S[StructureType.WATER_EXTRACTOR]] >= 0.0
    pozzo_da_aprire = (raw_occupancy > 0) & senza_pozzo & ~cantiere_pozzo
    priority[:, :, idx(ActionType.BUILD_WATER_EXTRACTOR)] = np.where(
        pozzo_da_aprire & masks[:, :, idx(ActionType.BUILD_WATER_EXTRACTOR)],
        1.0,
        priority[:, :, idx(ActionType.BUILD_WATER_EXTRACTOR)],
    )
    deficit_cantieri[ActionType.BUILD_WATER_EXTRACTOR] = np.where(
        pozzo_da_aprire, 1.0, needs[:, :, C.ND["build_water"]]
    )
    senza_serra = cells.struct_count[:, :, C.S[StructureType.GREENHOUSE]] == 0
    cantiere_serra = cells.site_progress[:, :, C.S[StructureType.GREENHOUSE]] >= 0.0
    virgin_occupied = (raw_occupancy > 0) & senza_serra & ~cantiere_serra
    greenhouse_priority = np.where(
        virgin_occupied, 1.0, greenhouse_priority
    )
    set_priority(ActionType.BUILD_GREENHOUSE, greenhouse_priority)
    deficit_cantieri[ActionType.BUILD_GREENHOUSE] = np.where(
        virgin_occupied, 1.0, needs[:, :, C.ND["build_greenhouse"]]
    )
    # **Gli alloggi si misurano in UNITA' mancanti, come tutte le altre
    # costruzioni (2026-08-30).** Era `pressione / occupanti`: con diciassette
    # persone scoperte su centootto la priorita' valeva 0,225 e finiva sesta,
    # cioe' fuori dal taglio `top_k = 5`, mentre `explore` entrava con 0,400.
    # E' la terza volta che questo modello divide un arretrato ESTENSIVO per il
    # numero di persone e lo fa sparire: era gia' successo alla manutenzione
    # (0,09 contro `observe` a 0,20) e alla prosperita'. Serre, pannelli e
    # impianti d'ossigeno usano gia' `clip(unita_mancanti, 0, 1)`: gli alloggi
    # erano l'eccezione, non la regola.
    # `habitat_pressure` conta le PERSONE scoperte ORA. Qui si guarda invece il
    # fabbisogno DA COPRIRE — presenti piu' margine di crescita — perche' una
    # cella che ha esattamente i posti che le servono non puo' piu' crescere: la
    # nascita richiede spazio libero, e senza margine il modello si chiude in un
    # altopiano permanente. Il bisogno pubblicato resta quello esatto: lo legge
    # anche la logistica fra celle, e gonfiarlo spegneva le donazioni.
    # `raw_occupancy` e non `occ_f`: quest'ultima ha il pavimento a uno, e una
    # cella disabitata risulterebbe bisognosa di un alloggio che non serve.
    presenti = raw_occupancy.astype(np.float64)
    coperti_alloggi = np.where(
        presenti > 0.0,
        presenti
        + np.maximum(1.0, np.ceil(presenti / float(POPOLAZIONE_PER_DADO_DI_NATALITA))),
        0.0,
    )
    posti_costruiti = housing_slots_from_structures(
        cells.struct_count[:, :, C.S[StructureType.SHELTER]],
        cells.struct_count[:, :, C.S[StructureType.HABITAT]],
        cells.struct_count[:, :, C.S[StructureType.INFIRMARY]],
    )
    pressione = np.maximum(0.0, coperti_alloggi - posti_costruiti)
    habitat_p = 0.08 + 0.92 * np.clip(pressione, 0.0, 1.0)
    set_priority(ActionType.BUILD_SHELTER, habitat_p)
    set_priority(ActionType.BUILD_HABITAT, habitat_p)
    # Un rifugio copre una persona, un habitat due.
    deficit_cantieri[ActionType.BUILD_SHELTER] = pressione
    deficit_cantieri[ActionType.BUILD_HABITAT] = pressione / 2.0
    # **La priorita' si normalizza sulle STRUTTURE, non sulle persone
    # (2026-08-27).** Era `bisogno / occupanti`: in una cella di duecento
    # coloni il rapporto restava minuscolo e la manutenzione valeva 0,09, cioe'
    # MENO di `observe`, che non fa nulla e vale 0,20. La sola run sopravvissuta
    # delle sette del 26-27 agosto ha ricominciato a manutenere soltanto DOPO
    # essere crollata a un abitante, perche' con un occupante lo stesso rapporto
    # satura: piu' la colonia era grande, meno si manuteneva, che e' l'esatto
    # contrario di cio' che serve — le strutture da tenere in piedi crescono con
    # la colonia, non calano.
    #
    # Il bisogno vale al massimo `RIPARAZIONE_MANUTENZIONE` per struttura,
    # quindi dividerlo per quel massimo lo riporta in [0, 1] e la priorita'
    # diventa **la frazione di produzione perduta**: 0,08 a impianti nuovi,
    # 0,26 quando manca un decimo di integrita' (e supera `observe`), satura a
    # 0,6. E' il segnale giusto perche' la resa di una struttura E' la sua
    # integrita'.
    capienza_riparabile = RIPARAZIONE_MANUTENZIONE * cells.struct_count.sum(
        axis=-1
    ).astype(np.float64)
    set_priority(
        ActionType.MAINTAIN_STRUCTURE,
        0.08
        + 0.92
        * np.clip(
            np.divide(
                needs[:, :, C.ND["maintenance"]],
                capienza_riparabile,
                out=np.zeros_like(capienza_riparabile),
                where=capienza_riparabile > 0.0,
            ),
            0.0,
            1.0,
        ),
    )

    # Development actions stay possible at low priority when a cell has spare
    # task slots; they no longer win just because a rigid binder listed them
    # first.
    directly_prioritized = {
        ActionType.BUILD_SOLAR_ARRAY,
        ActionType.BUILD_OXYGEN_PLANT,
        ActionType.BUILD_GREENHOUSE,
        ActionType.BUILD_SHELTER,
        ActionType.BUILD_HABITAT,
        ActionType.BUILD_WATER_EXTRACTOR,
    }
    # **0,07 rende questo livello irraggiungibile, e non per poco.** Il menu
    # tiene le prime `top_k` azioni di lavoro (default 5), e in una colonia sana
    # le prime cinque valgono 0,45 / 0,39 / 0,16 / 0,15 / 0,10: il livello di
    # sviluppo e' fuori sempre. Misurato in una cella da 200 coloni, laboratori
    # di ricerca (12 richiesti, 0 presenti), infermerie (15 richieste, 0) e
    # stazioni meteo hanno domanda non soddisfatta e non entrano mai nel menu,
    # mentre `observe` — che non fa nulla: `ActionResult(True, "observed area")`
    # senza alcun effetto — entra con 0,20 e batte tre azioni di lavoro vero.
    #
    # La conseguenza non e' solo estetica. I laboratori sono l'unica sorgente di
    # `tools` (l'effetto `knowledge` in `kernel_biology`), e i `tools` alzano
    # `tool_factor` da 1,0 fino a 1,75 moltiplicando la produzione di cibo,
    # materiale e acqua. La colonia non entra mai in quel giro: `tool_stock`
    # vale 303,7 costante in ogni run mai eseguita in questo progetto.
    #
    # **Il valore NON e' piu' 0,07: il default e' 0,30** (`kernel.py`,
    # `development_build_priority`), e il paragrafo qui sopra e' la diagnosi
    # storica, non lo stato attuale. Restava scritto «il valore resta 0,07» dopo
    # che era gia' stato alzato, e con esso l'affermazione «`tool_stock` vale
    # 303,7 costante in ogni run»: misurato il 2026-08-30 su duecento coloni
    # varia fra 158 e 203 nella stessa run. Un commento che contraddice il
    # codice e' un difetto come un altro, e in un lavoro di tesi e' peggio,
    # perche' e' il commento che viene letto.
    # **Guidata dal fabbisogno, non piatta.** La versione precedente assegnava a
    # tutto questo livello un valore costante (0,07) che con `top_k = 5` lo
    # teneva fuori dal menu SEMPRE: in una cella da duecento coloni il modello
    # chiedeva otto laboratori e dieci infermerie, ne aveva zero, e ne
    # costruiva zero per l'intera run. Era un'incoerenza fra due parti dello
    # stesso modello — `COLONISTS_PER_STRUCTURE` dichiarava un fabbisogno che il
    # menu rendeva irraggiungibile — e non una scelta di bilanciamento.
    #
    # Ora la priorita' e' proporzionale alla quota di fabbisogno scoperto: piena
    # quando non ce n'e' nemmeno uno, nulla quando il rapporto e' soddisfatto.
    # La maschera di saturazione poco sopra li toglie comunque al completamento,
    # quindi questo termine non li fa costruire oltre il necessario: decide solo
    # se possono entrare in gara mentre servono davvero.
    for action, structure_type in BUILD_ACTIONS.items():
        if action in directly_prioritized:
            continue
        rapporto = COLONISTS_PER_STRUCTURE.get(structure_type)
        if rapporto is None:
            set_priority(action, development_priority)
            continue
        richiesti = np.maximum(1.0, np.ceil(occ_f / float(rapporto)))
        presenti = cells.struct_count[:, :, C.S[structure_type]].astype(np.float64)
        mancanti = np.maximum(0.0, richiesti - presenti)
        # **Il peso e' il numero di strutture mancanti, non la loro frazione.**
        # Con la sola frazione un avamposto da dodici coloni, cui ne serve UNA di
        # ciascun tipo e non ne ha nessuna, vedeva ogni tipo di sviluppo
        # "interamente scoperto" e li metteva tutti al massimo: cinque azioni a
        # pari punteggio riempivano il menu e ne cacciavano la raccolta dalle
        # serre mature — misurato, ed e' cio' che rompeva
        # `test_mature_greenhouses_publish_one_routine_cultivation_job_each`.
        # Il bisogno di infrastruttura di sviluppo va con quante persone serve,
        # e diventa pieno quando ne mancano almeno `DEVELOPMENT_FULL_DEFICIT`.
        urgenza = np.clip(mancanti / DEVELOPMENT_FULL_DEFICIT, 0.0, 1.0)
        set_priority(action, development_priority * urgenza)
        deficit_cantieri[action] = mancanti
    total_pressure = np.clip(
        needs.sum(axis=2) / np.maximum(1.0, occ_f * 12.0), 0.0, 1.0
    )
    if frontier is None:
        frontier = explore_frontier(cells)
    set_priority(
        ActionType.EXPLORE,
        0.10 + 0.25 * (1.0 - total_pressure) + 0.30 * frontier,
    )

    # **La cella sa che c'e' una tempesta, e lo dice (2026-08-31).** Fino a
    # oggi il livello decisionale non aveva alcun riferimento agli eventi
    # estremi: una colonia sotto una tempesta di polvere globale componeva lo
    # stesso identico menu di una giornata serena. Gli eventi agivano solo
    # sulla fisica — polvere e radiazione della cella — e da li' su
    # abitabilita', salute e usura, ma nessuno ne traeva una conseguenza sulle
    # scelte.
    #
    # **La severita' e' derivata, non dichiarata.** Un evento si riconosce
    # dall'eccesso di polvere o radiazione sopra la LINEA DI BASE della cella,
    # normalizzato per la stessa ampiezza che il motore degli eventi applica
    # (`DUST_STORM_OPACITY_SURGE`, `SOLAR_FLARE_RADIATION_SURGE`). Non serve un
    # nuovo campo di stato ne' una nuova costante: la firma dell'evento e' gia'
    # nella cella.
    #
    # **Si scoraggia il VIAGGIO, non il lavoro.** In questo modello un colono
    # non e' "dentro" o "fuori" rispetto alla propria cella: l'evento colpisce
    # chi ci sta, riparato o no. Cio' che una tempesta rende davvero
    # sconsigliabile e' attraversare il territorio, ed e' esattamente `move` ed
    # `explore`. La manutenzione non va incoraggiata a mano: la tempesta alza
    # gia' l'usura, quindi il bisogno di manutenzione e la sua priorita'
    # salgono da soli.
    #
    # **E si smette di scoraggiarlo quando restare uccide.** Il fattore si
    # annulla dove la cella non e' in grado di erogare ne' cibo ne' acqua: li'
    # restare fermi non e' prudenza ma condanna, e un colono deve poter
    # rischiare il viaggio. E' la stessa deroga, e la stessa condizione, della
    # guardia di sopravvivenza in `preference_agent`.
    # Dove la linea di base non e' stata registrata si assume che il valore
    # corrente SIA la linea di base, cioe' nessun evento. L'alternativa,
    # trattarla come zero, farebbe apparire come tempesta permanente la polvere
    # ordinaria di qualunque cella priva di storia.
    base_polvere = np.where(np.isnan(cells.baseline_dust), cells.dust, cells.baseline_dust)
    base_radiazione = np.where(
        np.isnan(cells.baseline_radiation), cells.radiation, cells.baseline_radiation
    )
    eccesso_polvere = np.maximum(0.0, cells.dust - base_polvere) / DUST_STORM_OPACITY_SURGE
    eccesso_radiazione = (
        np.maximum(0.0, cells.radiation - base_radiazione) / SOLAR_FLARE_RADIATION_SURGE
    )
    severita_evento = np.clip(
        np.maximum(eccesso_polvere, eccesso_radiazione), 0.0, 1.0
    )
    la_cella_eroga = (
        cells.cell_res[:, :, C.R["food"]] >= RAZIONE_MINIMA
    ) & (cells.cell_res[:, :, C.R["water"]] >= RAZIONE_MINIMA)
    fattore_viaggio = 1.0 - severita_evento * la_cella_eroga.astype(np.float64)
    for azione in (ActionType.MOVE, ActionType.EXPLORE):
        priority[:, :, idx(azione)] *= fattore_viaggio

    # The proposal layer and the executor must share the same infrastructure
    # coverage contract.  A low-priority saturated build is still a wasted
    # seven-day turn, so remove it here instead of relying on validation to
    # reject it later.  Existing sites remain eligible until completion.
    alloggi_costruiti = housing_slots_from_structures(
        cells.struct_count[:, :, C.S[StructureType.SHELTER]],
        cells.struct_count[:, :, C.S[StructureType.HABITAT]],
        cells.struct_count[:, :, C.S[StructureType.INFIRMARY]],
    )
    for action, structure_type in BUILD_ACTIONS.items():
        action_idx = idx(action)
        structure_idx = C.S[structure_type]
        active_site = cells.site_progress[:, :, structure_idx] >= 0.0
        # **Era una SECONDA regola di saturazione, e vinceva su tutte
        # (2026-08-30).** Applicare il margine di crescita alla maschera di
        # cella e alla saturazione a oggetti non cambiava la simulazione di un
        # solo passo, perche' qui il cancello si richiudeva col criterio vecchio
        # — copertura esatta della popolazione presente. Ora la regola e' una.
        saturated = costruzione_satura(
            structure_type,
            cells.struct_count[:, :, structure_idx],
            occupancy,
            alloggi_costruiti,
        )
        remove = saturated & ~active_site
        masks[:, :, action_idx] &= ~remove
        priority[:, :, action_idx] *= masks[:, :, action_idx]
        # A half-built site is already a colony commitment. Keep it inside
        # the bounded menu ahead of fresh jobs, otherwise top-k can hide the
        # continuation and strand a founder among several parallel starts.
        #
        # **Il valore e' configurabile dal 2026-09-14, e il difetto e' proprio
        # il valore.** Ogni altra priorita' di lavoro vale `0,08 + 0,92 x ...`,
        # cioe' al massimo 1,0: un pavimento a 1,1 non tiene il cantiere
        # «dentro» il menu, lo mette SOPRA tutto il resto. Con undici tipi di
        # struttura bastano cinque cantieri aperti perche' il taglio a cinque
        # espella manutenzione, raccolta del ghiaccio e foraggiamento --- e un
        # cantiere fermo resta aperto, quindi occupa il posto per sempre.
        # Il valore di sempre resta il predefinito; abbassarlo e' un braccio
        # sperimentale, da confrontare appaiato.
        if pavimento_cantieri > 0.0:
            priority[:, :, action_idx] = np.where(
                active_site & masks[:, :, action_idx],
                np.maximum(priority[:, :, action_idx], pavimento_cantieri),
                priority[:, :, action_idx],
            )

    # Il governatore inclina le priorita' QUI e non dopo: il top-k sceglie il
    # menu leggendo `priority`, quindi una direttiva applicata piu' tardi
    # riordinerebbe soltanto azioni gia' ammesse e non potrebbe promuoverne una
    # esclusa — su una priorita' appena azzerata dal taglio un moltiplicatore e'
    # un non-intervento. La maschera resta sovrana: un'azione non ammissibile ha
    # priorita' 0 e nessun moltiplicatore la fa rientrare.
    # **Provato e RITIRATO: frenare i cantieri con l'arretrato di
    # manutenzione (2026-08-31).** L'ipotesi era che la colonia
    # sovracostruisse rispetto alla manodopera che ha per mantenere, e nasceva
    # da un confronto fra due run che differivano ANCHE per il sito: 919
    # strutture a integrita' 0,386 contro 578 a 0,673. Un controllo pulito, a
    # parita' di sito e di seme su 3000 passi, l'ha rifiutata.
    #
    #   senza freno   919 strutture   integrita' 0,386   12 celle   119 morti
    #   con freno     196 strutture   integrita' 0,921    5 celle   122 morti
    #
    # Il freno funzionava, e proprio per questo si vede che l'ipotesi era
    # sbagliata: l'integrita' sale da 0,386 a 0,921 e i morti non calano. La
    # ragione sta in una grandezza che nessuna delle due letture guardava, la
    # CAPIENZA EFFICACE, cioe' il numero di strutture per la loro efficienza:
    #
    #   919 x 0,193 = 177,4        196 x 0,921 = 180,5        (1,7% di scarto)
    #
    # I due regimi producono la stessa capienza efficace, e infatti gli stessi
    # morti (119 contro 122, 2,5% di scarto). La colonia e' indifferente fra
    # "molte strutture malandate" e "poche ben tenute", perche' il vincolo vero
    # e' la manodopera: un colono puo' costruire oppure mantenere, e in un modo
    # o nell'altro la capienza efficace per colono ha lo stesso tetto.
    #
    # Il freno costava percio' l'espansione, che e' la grandezza di esito del
    # protocollo (da dodici celle a cinque), senza comprare sopravvivenza.
    if priority_tilt is not None:
        priority_tilt(priority)

    top_k = max(0, min(int(top_k), len(TASK_ACTIONS)))
    task_indices = np.asarray([idx(action) for action in TASK_ACTIONS], dtype=np.int64)
    selected = np.zeros((cells.H, cells.W, len(TASK_ACTIONS)), dtype=np.bool_)
    if top_k:
        task_priority = priority[:, :, task_indices]
        order = np.argsort(-task_priority, axis=2, kind="stable")[:, :, :top_k]
        np.put_along_axis(selected, order, True, axis=2)
    masks[:, :, task_indices] &= selected
    priority[:, :, task_indices] *= masks[:, :, task_indices]

    # Finite quotas prevent an entire macro-cell from redundantly performing
    # the same seven-day job. Zero remains "not proposed"; -1 is unlimited.
    def need_quota(name: str, units_per_agent: float = 2.0) -> np.ndarray:
        return np.minimum(
            occupancy,
            np.maximum(
                1,
                np.ceil(needs[:, :, C.ND[name]] / units_per_agent).astype(np.int32),
            ),
        )

    def stock_quota(stock, units_per_job: float = 1.0) -> np.ndarray:
        """Maximum useful jobs supported by the stock visible this step."""
        return np.minimum(
            occupancy,
            np.ceil(np.maximum(np.asarray(stock, dtype=np.float64), 0.0) / units_per_job)
            .astype(np.int32),
        )

    ice_stock = cells.water_ice + cells.cell_res[:, :, C.R["ice"]]
    greenhouse_slots = cells.struct_count[:, :, C.S[StructureType.GREENHOUSE]].astype(
        np.int32
    )
    # Open-field forage removes 0.1 biomass per job; a greenhouse represents
    # one weekly production slot. The action mask still enforces soil maturity.
    forage_slots = np.where(
        greenhouse_slots > 0,
        greenhouse_slots,
        np.ceil(np.maximum(cells.vegetation, 0.0) / 0.1).astype(np.int32),
    )

    quota[:, :, idx(ActionType.COLLECT_ICE)] = np.minimum(
        need_quota("water"), stock_quota(ice_stock, 2.0)
    )
    food_need_jobs = need_quota("food")
    greenhouse_jobs = np.minimum(occupancy, greenhouse_slots)
    quota[:, :, idx(ActionType.FORAGE)] = np.where(
        greenhouse_slots > 0,
        greenhouse_jobs,
        np.minimum(food_need_jobs, np.minimum(occupancy, forage_slots)),
    )
    quota[:, :, idx(ActionType.COLLECT_MATERIALS)] = np.minimum(
        need_quota("materials"),
        stock_quota(cells.cell_res[:, :, C.R["construction_material"]]),
    )
    quota[:, :, idx(ActionType.COLLECT_MINERALS)] = np.minimum(
        need_quota("materials"),
        stock_quota(cells.cell_res[:, :, C.R["minerals"]]),
    )
    # **La quota dei cantieri segue il deficit (2026-08-25).** Era una costante
    # scritta a mano — `= 1` — mentre acqua, cibo, materiali, minerali,
    # manutenzione ed esplorazione hanno tutte una quota derivata dal bisogno.
    # Una colonia cui mancano undici serre poteva aprire un solo cantiere per
    # passo, esattamente come una cui ne manca una.
    #
    # **E' il difetto che rendeva muto il governatore.** Misurato dopo una crisi
    # al 50%: la costruzione arriva in cima al menu (priorita' 1,000 contro
    # 0,727 della raccolta minerali) e il peso del governatore la porta a 4,000,
    # ma `_apply_claim_limits` la maschera appena UN colono l'ha rivendicata.
    # Novantadue coloni nella cella, uno solo poteva costruire, ottantotto
    # scavare. La priorita' decide CHI; la quota decide QUANTI, e nessun peso
    # puo' muoverla. Con la costruzione gia' prima in classifica, moltiplicarla
    # per quattro non cambiava nemmeno chi occupava l'unico posto: il tilt era
    # un non-intervento perche' ridondante, non perche' rotto.
    #
    # Il pavimento a 1 tiene il comportamento storico dove il deficit e' nullo o
    # frazionario (compreso l'homestead su terra vergine, che vale un solo
    # slot), quindi la quota puo' solo crescere rispetto a prima, mai calare.
    for action in BUILD_ACTIONS:
        mancanti = deficit_cantieri.get(action)
        if mancanti is None:
            quota[:, :, idx(action)] = 1
            continue
        quota[:, :, idx(action)] = np.minimum(
            occupancy,
            np.maximum(1, np.ceil(np.maximum(0.0, mancanti)).astype(np.int32)),
        )
    # **La quota conta i manutentori, non le passate (2026-08-27).** Una
    # manutenzione vale `CAPACITA_MANUTENZIONE` unita' di integrita', cioe' il
    # lavoro di un colono in un passo, quindi un arretrato grande si smaltisce
    # con piu' persone e non con una sola che ripara tutto. E' la stessa forma
    # delle altre quote derivate dal bisogno: `need_quota` divide l'arretrato
    # per cio' che una persona riesce a fare.
    quota[:, :, idx(ActionType.MAINTAIN_STRUCTURE)] = need_quota(
        "maintenance", CAPACITA_MANUTENZIONE
    )
    quota[:, :, idx(ActionType.EXPLORE)] = np.maximum(
        1, np.ceil(occupancy * 0.15).astype(np.int32)
    )
    quota[:, :, task_indices] *= masks[:, :, task_indices]

    for action in pillars.AUTOMATIC_CELL_SERVICES:
        masks[:, :, idx(action)] = False
        priority[:, :, idx(action)] = 0.0
        quota[:, :, idx(action)] = 0

    return CellProposals(mask=masks, priority=priority, quota=quota)
