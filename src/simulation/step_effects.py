from __future__ import annotations

"""Shared per-step colony effects used by BOTH simulation engines.

SimulationController (live GUI / scripts/headless_runner.py) and
AgentCoupledRunner (YAML experiments / benchmarks) must stay behaviorally
identical: any change to the colony step logic belongs here, not in the
engines themselves.
"""

from src.core.constants import (
    AGENT_ENERGY_RESERVE_CAP,
    AGENT_OXYGEN_RESERVE_CAP,
    AGENT_REFILL_PER_STEP,
)
from src.world.grid import GridWorld
from src.world.structures import StructureType


#: **Rese di produzione delle strutture, per unita' di effetto e per passo.**
#:
#: Derivate, non scelte: ciascuna risolve
#:
#:     effetto_pro_capite x tool_tipico x resa = razione x margine
#:
#: con la razione di 0,1 unita' (la stessa di un pasto o di una bevuta), il
#: margine 1,3 — il bersaglio "la colonia sopravvive se ben gestita" — e gli
#: effetti pro capite MISURATI nella run di riferimento da cui vengono anche i
#: coefficienti di supporto (300 coloni, 150 passi): 0,1682 di cibo, 0,0234 di
#: acqua, 0,1268 di ossigeno.
#:
#: **Perche' i valori precedenti erano fuori scala.** La produzione era scalata
#: `* years` (dt/365, cioe' 0,0192 a passo settimanale) come se l'effetto
#: dichiarato fosse una resa ANNUA, mentre il consumo e' per passo. Finche' il
#: supporto era gratuito la discrepanza non si vedeva: la colonia non
#: consumava nulla e la produzione poteva essere qualunque cosa. Chiuso
#: l'anello, una serra che produce lo 0,4% di cio' che i suoi coloni mangiano
#: e' una colonia che muore di fame in una stagione.
#:
#: Il valore dell'acqua e' il piu' grande perche' l'effetto idrico dichiarato
#: dalle strutture e' il piu' piccolo (0,08 per serra): la resa non e' una
#: misura fisica in se', e' il cambio fra un effetto adimensionale e una massa.
#: Cio' che ha significato e' il rapporto fra produzione e consumo, ed e' 1,3
#: per tutte e tre.
RESA_CIBO = 0.7360
RESA_ACQUA = 5.2910
RESA_OSSIGENO = 1.0252


#: **Capienza del magazzino di cella, per risorsa vitale.**
#:
#: `SCORTA_PRO_CAPITE` e' il bersaglio che la colonia si da' da sola
#: (`knapsack_targets`, 2,0 per occupante contati due volte in
#: `needs.compute_needs`): quaranta razioni a testa, cioe' circa quaranta passi
#: di autonomia. Un valore, non una scelta nuova.
#:
#: `CAPIENZA_PER_DEPOSITO` da' finalmente uno scopo all'effetto `storage`: un
#: deposito non produce nulla — prima ne fabbricava materiale dal nulla — ma
#: aumenta di duecento razioni cio' che la cella puo' tenere da parte.
#:
#: **Perche' serve un tetto.** Con la produzione al 30% sopra il consumo la
#: giacenza cresce senza fine, e una colonia con mille passi di scorta e'
#: di nuovo insensibile a qualunque crisi: il difetto che questo lavoro ha
#: chiuso rientrerebbe dalla finestra, solo piu' lentamente. Il surplus oltre
#: la capienza non si conserva ed e' giusto cosi': e' raccolto che marcisce e
#: ossigeno che si disperde. L'audit lo registra come pozzo dichiarato.
#: **Il pavimento per cella con strutture.** Legare la capienza ai soli
#: occupanti azzera la giacenza di un avamposto rimasto vuoto: le sue serre
#: perderebbero il raccolto nell'istante in cui l'ultimo colono se ne va. Non
#: si vedeva finche' si guardavano le celle della colonia, che occupanti ne
#: hanno sempre — l'ha trovato un test sul carico elettrico, su una cella con
#: due impianti e nessuno dentro. Il massimo fra i due termini lascia il tetto
#: della colonia dov'era (120 coloni fanno 480, 176 strutture ne farebbero
#: 176) e da' comunque un magazzino a chi non ha nessuno.
CAPIENZA_PER_STRUTTURA = 1.0
SCORTA_PRO_CAPITE = 4.0
CAPIENZA_PER_DEPOSITO = 20.0
#: **Materiale e minerali NON sono a tetto, e la ragione e' misurata
#: (2026-08-31).** Provato ad aggiungerli quando l'ISRU ha cominciato a
#: produrli: il tetto CANCELLA il giacimento naturale della cella appena
#: qualcuno vi costruisce qualcosa — una cella con 500 unita' di minerale
#: scendeva a 21 al primo passo, cioe' un pozzo di massa non pagato proprio
#: sulle celle che la colonia deve raggiungere. Un magazzino tiene cio' che
#: tiene, ma un giacimento sta nel terreno e non e' merce in un capannone;
#: cibo, acqua e ossigeno si guastano e si disperdono, il metallo no.
#: E il tetto non serviva nemmeno allo scopo per cui l'avevo messo: la scorta
#: di materiale della colonia sta nelle SACCHE dei coloni (misurato: 1.184
#: unita' su 89 coloni, e 2,0 nel magazzino della cella madre), e il tetto
#: sulle celle non le tocca. Resta annotato come questione aperta.
RISORSE_A_TETTO = ("food", "water", "oxygen")


#: **Carico elettrico per struttura, e per passo.**
#:
#: La forma non e' nuova: e' esattamente quella che `colony_dynamics` gia'
#: DICHIARA nel denominatore di `power_margin`
#: (`oxygen_capacity + food_capacity x 0,5`, cioe' impianto d'ossigeno 1,0 +
#: habitat 0,35 + infermeria 0,6, piu' meta' di serra 1,0 + habitat 0,35).
#: Quel carico era pero' soltanto dichiarato: **nessuno lo pagava**. L'energia
#: veniva prodotta dai pannelli, versata nelle riserve dei coloni fino al
#: tetto, e poi si accumulava senza fine — misurato, da 240 a 503 unita' in
#: duecento passi. I pannelli solari non servivano a nulla dopo essere stati
#: costruiti.
#:
#: `RESA_ENERGIA_PER_CARICO` converte quella forma adimensionale in massa per
#: passo. E' derivata, non scelta: risolve `produzione = carico x margine` sul
#: parco strutture MISURATO a 120 coloni (18 pannelli, 12 impianti d'ossigeno,
#: 18 serre, 24 habitat, 4 infermerie), con la produzione a 0,075 per pannello
#: e per passo (effetto 1,5 x RESA_ENERGIA) e il margine 1,3.
#:
#: **Cosa succede quando non basta.** La copertura — energia disponibile
#: diviso carico — scala la produzione di ossigeno, cibo e acqua di quella
#: cella. Un impianto senza corrente non produce, ed e' l'anello che mancava:
#: i pannelli ora contano, e distruggerli affama la colonia con il ritardo del
#: magazzino invece che mai.
#: **Il riscaldatore c'e' (2026-08-25).** La metrica non lo dichiarava, e per
#: un attimo l'ho lasciato fuori per non inventare un numero che il modello non
#: dava. Ma un riscaldatore che scalda +2,0 gradi senza consumare corrente e'
#: l'unica struttura del parco che viola il buon senso in modo evidente: su
#: Marte il riscaldamento e' il carico dominante. Gli si da' il peso di un
#: impianto di processo (1,0, come il generatore d'ossigeno), e la scala si
#: ritara per lasciare il margine dov'era: il parco misurato a 120 coloni
#: passa da 36,0 a 42,0 unita' di carico, quindi 0,0290 x 36/42 = 0,0249.
CARICO_ELETTRICO = {
    StructureType.OXYGEN_PLANT: 1.0,
    # Il pozzo perfora, scalda e condensa: e' un impianto di processo come
    # il generatore d'ossigeno, e consuma come lui.
    StructureType.WATER_EXTRACTOR: 1.0,
    StructureType.GREENHOUSE: 0.5,
    StructureType.HABITAT: 0.525,
    StructureType.INFIRMARY: 0.6,
    StructureType.HEATER: 1.0,
}
RESA_ENERGIA_PER_CARICO = 0.0249

#: Energia prodotta da un'unita' di effetto `energy` per passo, prima della
#: resa solare. Estratta qui perche' la usano sia la produzione sia la metrica
#: `power_margin`, che senza una costante condivisa tornerebbero a divergere.
RESA_ENERGIA = 0.05


#: **Scala dell'usura delle strutture, per giorno simulato.**
#:
#: Derivata, non scelta: risolve `usura_tipica x scala x giorni = perdita di
#: riferimento`, dove la perdita di riferimento e' 0,4 — cioe' la distanza fra
#: una struttura nuova e la soglia di intervento della manutenzione (0,6) — e i
#: giorni sono quelli di un anno marziano (687 giorni terrestri). Una struttura
#: abbandonata a se' stessa scende sotto la soglia in un anno, ed e' un tempo
#: che si puo' raccontare.
#:
#: **Perche' il valore precedente non voleva dire nulla.** L'usura era scalata
#: `min(1, (dt/365,25)/10)`, cioe' su un DECENNIO: 2,0e-5 per passo, che porta
#: l'integrita' da 1,0 a 0,6 in 19.878 passi. Una sola azione di manutenzione
#: ne ripara 0,4, cioe' novantanove run da duecento passi. Eppure la
#: manutenzione veniva scelta 117 volte in duecento passi e costava 117 unita'
#: di materiale: **il 26% di tutta la spesa di materiale della colonia**, per
#: riparare un degrado che non avveniva.
USURA_TIPICA_PER_UNITA = 0.004 + 0.010 * 0.35 + 0.003
PERDITA_DI_RIFERIMENTO = 0.4
GIORNI_ANNO_MARZIANO = 687.0
SCALA_USURA_PER_GIORNO = (
    PERDITA_DI_RIFERIMENTO / GIORNI_ANNO_MARZIANO
) / USURA_TIPICA_PER_UNITA

#: Quanta integrita' ripristina una singola azione di manutenzione, per ogni
#: struttura della cella.
RIPARAZIONE_MANUTENZIONE = 0.4

#: Le tre rese lente, espresse **per passo di riferimento** come tutto il resto
#: del modello dal 2026-08-25.
#:
#: **Erano rese annue travestite (2026-08-28).** Restavano moltiplicate per
#: `years = dt/365,25` mentre cibo, acqua, ossigeno e conversione ISRU erano
#: gia' state portate per passo: dentro la stessa funzione lo stesso dizionario
#: `local_effect` veniva letto su DUE scale temporali, e il valore vero era
#: nascosto dentro una divisione per 365,25 invece di essere dichiarato. Una
#: trappola per chi legge — ha ingannato anche l'analisi che l'ha trovata, che
#: aveva concluso «il livello tecnologico e' irraggiungibile» estrapolando da
#: 150 passi.
#:
#: **I valori NON cambiano.** `K x (dt/7)` e `0,20 x (dt/365,25)` sono
#: identici per ogni `dt` — la riscrittura e' esattamente a comportamento
#: invariato, ed e' verificata dai test di parita' fra i motori. Misurato su
#: 1500 passi (28,7 anni, l'orizzonte di campagna): conoscenza 111,6 e
#: `tech_tier` 1 raggiunto. E' la ragione per cui la scala non e' stata
#: toccata: portarla per passo senza ritararla avrebbe sbloccato tutti i
#: livelli tecnologici in due anni.
GIORNI_PER_PASSO_RIFERIMENTO = 7.0
GIORNI_ANNO_TERRESTRE = 365.25

#: Attrezzi prodotti da un'unita' di effetto `knowledge`, per passo.
RESA_ATTREZZI = 0.20 * GIORNI_PER_PASSO_RIFERIMENTO / GIORNI_ANNO_TERRESTRE

#: Conoscenza accumulata da un'unita' di effetto `knowledge`, per passo.
#: Le soglie di `tech_tier` (50 / 150 / 400) sono tarate su questa.
RESA_CONOSCENZA = GIORNI_PER_PASSO_RIFERIMENTO / GIORNI_ANNO_TERRESTRE

#: Consumo ambientale di scorte personali, per colono e per passo. Non e' il
#: consumo vitale (quello sta in `vitals.py` ed e' molto piu' grande): e' il
#: logorio del materiale e del cibo trasportati.
PRELIEVO_CIBO_PERSONALE = 0.04 * GIORNI_PER_PASSO_RIFERIMENTO / GIORNI_ANNO_TERRESTRE
PRELIEVO_MATERIALE_PERSONALE = 0.01 * GIORNI_PER_PASSO_RIFERIMENTO / GIORNI_ANNO_TERRESTRE


#: Quanta integrita' un colono ripara in un passo, sull'intera cella.
#:
#: **Il costo non era sbagliato: era l'azione a essere indivisibile
#: (2026-08-27).** Dal 2026-08-25 il costo segue cio' che si ripara, ma cio'
#: che si riparava era il parco INTERO della cella mentre a pagare era un solo
#: colono. In una cella di 235 strutture al 77% di integrita' faceva 13,7 unita'
#: di materiale contro le 1,9 che un colono porta addosso: l'azione veniva
#: proposta e respinta a ogni passo, e la colonia si spegneva con il magazzino
#: pieno di cibo. In aggregato la spesa e' invece minima — tenere 235 strutture
#: costa 0,22 unita' di materiale per passo — quindi il difetto non stava nel
#: prezzo ma nel fatto che una manutenzione fosse un atto unico e indivisibile.
#:
#: Il valore e' tarato sul portafoglio di un colono: 4,0 unita' di integrita'
#: costano 4,0 x 0,25 = 1,0 di materiale, cioe' meta' delle 2,0 che la
#: redistribuzione tiene nella sacca di ciascuno. Una singola persona deve
#: sempre potersi permettere una singola manutenzione, altrimenti l'azione
#: esiste solo sulla carta. Dove l'arretrato sta sotto questa capienza il
#: comportamento e' identico a prima: una passata ripara tutto.
CAPACITA_MANUTENZIONE = 4.0

#: **Il costo segue cio' che si ripara.** Prima era un'unita' fissa di materiale
#: qualunque fosse il parco riparato: in una cella con sessanta strutture
#: un'unita' comprava ventiquattro unita' di integrita'. Ora si paga a unita'
#: ripristinata, a un quarto di quanto costa costruire (un rifugio costa circa
#: 2,9 di materiale per un'integrita' di 1,0, quindi 0,73 per unita': un
#: quarto e' 0,18, arrotondato a 0,25 per tenere la manutenzione conveniente
#: rispetto alla ricostruzione).
COSTO_MANUTENZIONE_PER_INTEGRITA = 0.25


#: **Kit medici prodotti da un'unita' di effetto `healing_bonus`, per passo.**
#:
#: Prima del 2026-08-25 nessuno li produceva: erano solo consumati
#: (`USE_MED_KIT`) e scambiati, e l'unica sorgente era la dotazione dei nuovi
#: nati. L'infermeria — che costa materiale, minerali, ossigeno ed energia —
#: dava `healing_bonus` ma non un solo kit, mentre il piano della colonia
#: dichiara un fabbisogno di 1,0 kit per abitante che nessuna produzione
#: poteva soddisfare. Su orizzonti lunghi la scorta poteva solo calare.
#:
#: Derivata come le altre: il consumo misurato e' di 4 kit in duecento passi a
#: 120 coloni (0,02 per passo), e il parco infermerie della stessa run e' di 4
#: unita' con `healing_bonus` 0,55 ciascuna, cioe' 2,2 di effetto. Con margine
#: 1,3 la resa e' 0,02 x 1,3 / 2,2.
#:
#: Il kit e' l'unica risorsa che nasce senza consumarne un'altra, ed e'
#: deliberato: rappresenta lavoro di laboratorio e farmaci sintetizzati in
#: loco, che nel modello non hanno un ingrediente. L'audit lo registra come
#: sorgente dichiarata.
RESA_KIT_MEDICI = 0.0118


def structure_cells(world: GridWorld):
    """Yield (x, y, cell) for every cell that currently hosts structures.

    Uses the spatial index kept by GridWorld instead of a full-grid scan.
    """
    positions = getattr(world, "_structure_positions", None)
    if positions is None:
        for y, row in enumerate(world.cells):
            for x, cell in enumerate(row):
                if cell.structures:
                    yield x, y, cell
        return
    for x, y in sorted(positions):
        if world.in_bounds(x, y):
            cell = world.get_cell(x, y)
            if cell.structures:
                yield x, y, cell


def solar_yield(dust_level: float) -> float:
    """Quanta della resa nominale un pannello ottiene sotto quel cielo.

    Vale uno alla polvere ordinaria e scende all'11% a polvere satura: e' il
    rapporto che il MCD misura fra climatologia (527 W/m2) e tempesta globale
    (57,8 W/m2). Il fattore e' RELATIVO alla polvere tipica perche' la colonia
    e' dimensionata su quel cielo; una resa assoluta ricalibrerebbe l'intero
    bilancio energetico per rimetterlo dov'era.

    Vive qui e non nel kernel perche' i due motori devono condividerla: senza,
    la parita' si romperebbe appena la polvere sale sopra il tipico — ed e'
    successo, misurato (0,075 contro 0,0439 a dust 0,60) prima che questa
    funzione esistesse.
    """
    from src.core.kernel_biology import SOLAR_DUST_LOSS, TYPICAL_DUST

    eccesso = (float(dust_level) - TYPICAL_DUST) / (1.0 - TYPICAL_DUST)
    return 1.0 - SOLAR_DUST_LOSS * min(1.0, max(0.0, eccesso))


def _power_coverage_map(world: GridWorld) -> dict:
    """Copertura elettrica per cella, calcolata dove l'energia si produce.

    Vive sul `world` come gli altri cache di fase (`_habitability_cache` e
    compagnia) perche' i due momenti che la usano sono funzioni diverse:
    `apply_structure_effects` la calcola e la applica all'ossigeno,
    `apply_colony_resource_feedback` la applica a cibo e acqua. Il kernel non
    ne ha bisogno — li' e' una variabile locale nella stessa funzione — ma il
    valore per cella deve essere lo STESSO nei due momenti, ed e' la ragione
    per cui si calcola una volta sola invece che due.
    """
    mappa = getattr(world, "_power_coverage", None)
    if mappa is None:
        mappa = {}
        world._power_coverage = mappa
    return mappa


def apply_structure_effects(world: GridWorld) -> None:
    """Per-step local production of structures plus greenhouse fertilization."""
    coperture = _power_coverage_map(world)
    for x, y, cell in structure_cells(world):
        biomass_effect = 0.0
        resa = solar_yield(cell.dust_level)
        effetti = [struttura.local_effect for struttura in cell.structures]
        for effects in effetti:
            cell.resources.energy += effects.get("energy", 0.0) * RESA_ENERGIA * resa
            cell.resources.biomass += effects.get("biomass", 0.0) * 0.02
            biomass_effect += effects.get("biomass", 0.0)

        # **Il carico si paga (2026-08-25).** L'energia prodotta in questo
        # stesso passo e' gia' nella giacenza: gli impianti attingono da li'.
        # La copertura che ne risulta scala cio' che producono — qui
        # l'ossigeno, piu' avanti cibo e acqua.
        carico = sum(
            CARICO_ELETTRICO.get(struttura.type, 0.0) for struttura in cell.structures
        ) * RESA_ENERGIA_PER_CARICO
        if carico > 0.0:
            disponibile = float(cell.resources.energy)
            usata = min(disponibile, carico)
            cell.resources.energy = max(0.0, disponibile - usata)
            copertura = usata / carico
        else:
            copertura = 1.0
        coperture[(x, y)] = copertura

        for effects in effetti:
            cell.resources.oxygen += effects.get("oxygen", 0.0) * RESA_OSSIGENO * copertura
        if biomass_effect > 0.0:
            # Greenhouse cultivation seeds local vegetation and fertilizes the
            # regolith (N/P/C), the pathway that later unlocks Liebig-limited
            # natural growth once pressure/temperature allow it.
            cell.vegetation_biomass = min(10.0, cell.vegetation_biomass + biomass_effect * 0.01)
            cell.nutrients["N"] = min(20.0, cell.nutrients["N"] + biomass_effect * 0.10)
            cell.nutrients["P"] = min(4.0, cell.nutrients["P"] + biomass_effect * 0.02)
            cell.nutrients["C"] = min(10.0, cell.nutrients["C"] + biomass_effect * 0.05)
        cell.recompute_habitability(world.planetary_state)
        world._habitability_cache[y, x] = cell.habitability_score
        world._vegetation_cache[y, x] = cell.vegetation_biomass
        world._radiation_cache[y, x] = cell.radiation_level
        world._temperature_cache[y, x] = cell.local_temperature_modifier


def apply_colony_resource_feedback(world: GridWorld, agents: dict, dt: float) -> None:
    """ABM colony loop: structures and tools create local capacity consumed by population.

    Also closes the energy/oxygen loop: solar arrays and oxygen plants stock the
    cell, and agents standing at the site implicitly recharge suit reserves from
    that stock (energy is otherwise only consumed by construction costs).
    """
    passo = dt / GIORNI_PER_PASSO_RIFERIMENTO
    coperture = _power_coverage_map(world)
    for _x, _y, cell in structure_cells(world):
        tool_factor = 1.0 + min(0.75, cell.resources.tools * 0.05)
        # Senza corrente non si coltiva e non si ricicla: la stessa copertura
        # gia' applicata all'ossigeno vale per cibo e acqua. Gli attrezzi no:
        # sono lavoro di laboratorio, non impianto di processo.
        copertura = coperture.get((_x, _y), 1.0)
        for structure in cell.structures:
            effects = structure.local_effect
            efficiency = structure.efficiency * tool_factor
            cell.resources.food += effects.get("food", 0.0) * efficiency * RESA_CIBO * copertura
            # **Gli attrezzi si fabbricano, non si evocano (2026-08-25).** Il
            # laboratorio li produceva dal nulla; ora consumano il materiale da
            # costruzione con cui sono fatti, uno a uno, e si fermano quando la
            # cella non ne ha piu'. E' poca massa (+3,2 in duecento passi) ma e'
            # la stessa famiglia di difetti, e chiuderla costa due righe.
            attrezzi = effects.get("knowledge", 0.0) * RESA_ATTREZZI * efficiency * passo
            attrezzi = min(attrezzi, max(0.0, float(cell.resources.construction_material)))
            cell.resources.construction_material -= attrezzi
            cell.resources.tools += attrezzi
            cell.resources.water += effects.get("water", 0.0) * efficiency * RESA_ACQUA * copertura
            cell.resources.med_kits += effects.get("healing_bonus", 0.0) * efficiency * RESA_KIT_MEDICI * copertura

    per_agent_food_draw = PRELIEVO_CIBO_PERSONALE * passo
    per_agent_material_draw = PRELIEVO_MATERIALE_PERSONALE * passo
    for agent in agents.values():
        agent.inventory.food = max(0.0, agent.inventory.food - per_agent_food_draw)
        agent.inventory.construction_material = max(0.0, agent.inventory.construction_material - per_agent_material_draw)
        cell = world.get_cell(agent.x, agent.y)
        if not cell.structures:
            continue
        if cell.resources.energy > 0.0 and agent.inventory.energy < AGENT_ENERGY_RESERVE_CAP:
            draw = min(AGENT_REFILL_PER_STEP, cell.resources.energy, AGENT_ENERGY_RESERVE_CAP - agent.inventory.energy)
            cell.resources.energy -= draw
            agent.inventory.energy += draw
        if cell.resources.oxygen > 0.0 and agent.inventory.oxygen < AGENT_OXYGEN_RESERVE_CAP:
            draw = min(AGENT_REFILL_PER_STEP, cell.resources.oxygen, AGENT_OXYGEN_RESERVE_CAP - agent.inventory.oxygen)
            cell.resources.oxygen -= draw
            agent.inventory.oxygen += draw

    # Il tetto si applica DOPO tutte le aggiunte del passo, una volta sola, in
    # entrambi i motori nello stesso punto: e' cio' che rende la parita'
    # verificabile senza duplicare la regola ramo per ramo.
    for _x, _y, cell in structure_cells(world):
        capienza_deposito = sum(
            float(s.local_effect.get("storage", 0.0)) for s in cell.structures
        ) * CAPIENZA_PER_DEPOSITO
        tetto = max(
            len(cell.agents_present) * SCORTA_PRO_CAPITE,
            len(cell.structures) * CAPIENZA_PER_STRUTTURA,
        ) + capienza_deposito
        for risorsa in RISORSE_A_TETTO:
            giacenza = float(getattr(cell.resources, risorsa, 0.0))
            if giacenza > tetto:
                setattr(cell.resources, risorsa, tetto)

def research_knowledge_gain(world: GridWorld, agents: dict, dt: float) -> float:
    """Scientific knowledge produced per step by research assets and tools."""
    research_assets = sum(
        1
        for _x, _y, cell in structure_cells(world)
        for structure in cell.structures
        if structure.type.value in {"research_lab", "weather_station"}
    )
    tool_stock = sum(float(getattr(agent.inventory, "tools", 0.0)) for agent in agents.values())
    if research_assets <= 0 and tool_stock <= 0:
        return 0.0
    return (research_assets * 0.5 + tool_stock * 0.03) * (dt / 365.0)


def apply_structure_wear(world: GridWorld, dt: float) -> None:
    """Degrade structure integrity based on local climate (dust, wind, radiation)."""
    years = float(dt) * SCALA_USURA_PER_GIORNO

    # TECH BONUS: Wear reduction from scientific progress
    wear_reduction = float(world.planetary_state.get("wear_reduction", 0.0))
    wear_multiplier = 1.0 - wear_reduction

    for x, y, cell in structure_cells(world):
        # Base wear (natural aging)
        base_wear = 0.004 * years

        # Climate-driven wear
        # Dust abrasion: higher dust = faster degradation
        dust_wear = cell.dust_level * 0.010 * years

        # Mechanical stress from wind (MCD wind speed)
        wind_speed = float(world.planetary_state.get("wind_speed_m_s", 5.0))
        # Wind wear scales quadratically with speed above 10m/s
        wind_wear = max(0.0, (wind_speed - 10.0) ** 2 / 400.0) * 0.006 * years

        # Radiation fatigue
        rad_wear = cell.radiation_level * 0.003 * years

        total_wear = (base_wear + dust_wear + wind_wear + rad_wear) * wear_multiplier

        for s in cell.structures:
            s.integrity = max(0.0, s.integrity - total_wear)
            if s.integrity < 0.2:
                world.log_event(
                    "structure_warning",
                    f"{s.type.value} at ({x},{y}) integrity is critical ({s.integrity * 100:.1f}%)",
                    x=x,
                    y=y,
                )
