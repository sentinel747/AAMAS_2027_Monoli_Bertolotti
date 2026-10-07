from __future__ import annotations

import math

from src.world.cell import cell_degradation_enabled
from src.world.grid import GridWorld

RATION_EPSILON = 1.0e-6

#: **Finestre letali, ancorate alla fisiologia e alla durata del passo.**
#:
#: Con passo settimanale i valori precedenti (7 e 14) valevano 49 e 98 giorni:
#: un essere umano muore di sete in circa tre giorni e di fame in quaranta-
#: sessanta. La fame e' ora fisiologicamente giusta — sei passi sono
#: quarantadue giorni. La sete resta piu' generosa del vero (due passi,
#: quattordici giorni) perche' il contatore misura l'assenza TOTALE di acqua:
#: la borraccia personale (`consume_water_ration`) lo azzera finche' c'e'
#: qualcosa dentro, e a risoluzione settimanale una finestra di un solo passo
#: renderebbe letale qualunque disguido.
LETHAL_STEPS_WITHOUT_WATER = 2
LETHAL_STEPS_WITHOUT_FOOD = 6

#: **Soglie di sintomo, ricavate dalle finestre letali.**
#:
#: Erano scritte a mano (4 passi senz'acqua, 10 senza cibo) su finestre di 7 e
#: 14: con le finestre nuove sarebbero oltre la morte, cioe' codice
#: irraggiungibile. Ricavarle dalla finestra conserva la proporzione originale
#: e le tiene allineate a qualunque futura ritaratura.
PASSI_SINTOMO_SENZA_ACQUA = max(1, round(LETHAL_STEPS_WITHOUT_WATER * 4 / 7))
PASSI_SINTOMO_SENZA_CIBO = max(1, round(LETHAL_STEPS_WITHOUT_FOOD * 10 / 14))

#: Durata del passo su cui sono tarati i consumi per passo. Un passo piu' lungo
#: consuma in proporzione: e' l'unico modo perche' una run a passo diverso
#: descriva lo stesso mondo.
GIORNI_PER_PASSO_RIFERIMENTO = 7.0

#: **Consumo dei vitali di privazione, per passo.**
#:
#: Ciascuno e' l'inverso della propria finestra letale: senza rifornimento il
#: vitale arriva a zero esattamente quando il contatore uccide. Prima il
#: consumo era `0,015 x dt` con `dt = dt_days/3650`, cioe' 2,9e-5 — l'aritmetica
#: dei vitali era decorativa e i contatori facevano tutto da soli.
#:
#: L'ossigeno usa la finestra dell'acqua: l'ipossia e' un orologio veloce.
CONSUMO_SAZIETA_PER_PASSO = 1.0 / LETHAL_STEPS_WITHOUT_FOOD
CONSUMO_IDRATAZIONE_PER_PASSO = 1.0 / LETHAL_STEPS_WITHOUT_WATER
CONSUMO_OSSIGENO_PER_PASSO = 1.0 / LETHAL_STEPS_WITHOUT_WATER

#: **Una razione vale esattamente un passo.**
#:
#: Prima erano numeri sciolti: bere dava +0,35 di idratazione, mangiare +0,25
#: di sazieta'. Con il consumo per passo (0,5 e 0,167) un colono che beveva la
#: sua razione a ogni passo perdeva comunque 0,15 di idratazione per passo e
#: moriva di sete con la borraccia in mano — misurato, 60 morti su 120 coloni
#: in duecento passi, tutti per disidratazione.
#:
#: Legare la razione al consumo toglie il numero sciolto: 0,1 unita' di massa
#: sono, per definizione, l'acqua di un passo. Vale sia per l'azione
#: (`DRINK_WATER`, `EAT_FOOD`) sia per la scorta personale
#: (`consume_water_ration` e sorelle) sia per l'erogazione dell'impianto.
RISTORO_ACQUA = CONSUMO_IDRATAZIONE_PER_PASSO
RISTORO_CIBO = CONSUMO_SAZIETA_PER_PASSO
RISTORO_OSSIGENO = CONSUMO_OSSIGENO_PER_PASSO


#: Passi settimanali in un anno marziano (687 giorni terrestri).
PASSI_ANNO_MARZIANO = 687.0 / GIORNI_PER_PASSO_RIFERIMENTO

#: **Radiazione: la salute persa per passo a radiazione 1,0 e schermatura nulla.**
#:
#: Ancorata alla fisica, non al modello: la superficie marziana consegna circa
#: 230 mSv all'anno (misura RAD di Curiosity) contro un limite di carriera
#: dell'ordine del sievert, cioe' **quattro anni marziani** di esposizione piena
#: prima che la salute sia consumata. E' la scala che questo coefficiente
#: riproduce.
#:
#: Prima era `0,014 x dt` con `dt = dt_days/3650`: 2,7e-5 per passo, cioe' 0,003
#: di salute in cento passi. La radiazione — il pericolo che definisce Marte —
#: era di fatto assente dal modello.
#:
#: Misurato: i coloni vedono radiazione 0,84 con schermatura 0,19-0,29, quindi
#: una radiazione efficace di ~0,65 e una perdita di ~0,33 di salute in duecento
#: passi. E' molto, ed e' voluto: rende la schermatura (habitat e rifugi) una
#: scelta con un prezzo, e le infermerie qualcosa che serve davvero.
USURA_RADIAZIONE_PER_PASSO = 1.0 / (4.0 * PASSI_ANNO_MARZIANO)

#: **Inquinamento locale: quota del consumo d'ossigeno che aggiunge.**
#:
#: A inquinamento pieno il colono consuma il 30% di ossigeno in piu'. Al valore
#: misurato nelle celle della colonia (0,33) e' un decimo in piu': percettibile
#: senza essere una seconda causa di morte. Il termine sulla fatica segue la
#: stessa scala.
QUOTA_OSSIGENO_DA_INQUINAMENTO = 0.30
FATICA_DA_INQUINAMENTO_PER_PASSO = 0.010

#: **Freddo polare: passi di esposizione piena prima che la salute finisca.**
#:
#: L'esposizione misurata alla colonia e' ZERO — le celle insediate non sono
#: polari — quindi questo termine governa soltanto le spedizioni. Dieci passi
#: (settanta giorni) a esposizione piena e senza supporto vitale: abbastanza
#: severo da rendere le regioni polari una scelta con un costo, abbastanza lento
#: da lasciare il tempo di rientrare.
#:
#: I termini su ossigeno e fatica conservano il rapporto originale con quello
#: sulla salute (0,018 e 0,080 contro 0,030), cosi' la taratura sposta la scala
#: e non la forma.
PASSI_LETALI_ESPOSIZIONE_POLARE = 10.0
SALUTE_POLARE_PER_PASSO = 1.0 / PASSI_LETALI_ESPOSIZIONE_POLARE
OSSIGENO_POLARE_PER_PASSO = SALUTE_POLARE_PER_PASSO * (0.018 / 0.030)
FATICA_POLARE_PER_PASSO = SALUTE_POLARE_PER_PASSO * (0.080 / 0.030)

#: **Fatica ambientale**, che in una cella abitabile e' un RECUPERO: il fattore
#: `1,05 - abitabilita' x 1,8` e' negativo sopra 0,58 di abitabilita'. Il
#: coefficiente resta 0,035 e cambia solo la scala: a passo, +0,037 in una cella
#: ostile e -0,026 in una piena, contro i +0,08/+0,35 delle azioni. E' una
#: modulazione, non il motore della fatica — quello sono le azioni, che erano
#: gia' per passo.
FATICA_AMBIENTALE_PER_PASSO = 0.035


#: **Coefficienti del supporto di struttura, ritarati sulla scala del passo.**
#:
#: Derivati, non scelti: ciascuno risolve `supporto_pro_capite x coefficiente =
#: consumo_per_passo x margine`, con margine 1,3 — il bersaglio "la colonia
#: sopravvive se ben gestita" — e i supporti pro capite MISURATI nella run di
#: riferimento (300 coloni, 150 passi): 0,1682 di cibo, 0,0234 di acqua,
#: 0,1268 di ossigeno, 0,1139 di abitabilita', 0,0255 di cura.
#:
#: **L'acqua ha DUE contributi**, e derivarne uno solo sbaglia il bersaglio:
#: `local_support` (abitabilita') e `local_water_support` entrano insieme
#: nell'idratazione. I due sono ricavati insieme conservandone il rapporto
#: originale (1 : 2,75).
#:
#: L'ossigeno si tara sul consumo GIA' ridotto dall'abitabilita': dove ci sono
#: strutture il fattore di abitabilita' porta il consumo al 10% del nominale,
#: ed e' li' che il supporto deve arrivare.
#:
#: **Dal 2026-08-25 il consumo e' per passo** (`CONSUMO_*_PER_PASSO`), non piu'
#: `x dt` con `dt = dt_days/3650`: i valori precedenti (0,1159 / 0,3610 /
#: 0,1313 / 0,2563) compensavano un consumo cinquemila volte piu' piccolo.
#:
#: Restano il RITMO con cui il supporto ricarica un vitale; a decidere se il
#: supporto eroga qualcosa e' la razione (`RAZIONE_CIBO` e sorelle), che e'
#: massa prelevata dalla giacenza della cella. Un ritmo generoso su un
#: magazzino vuoto non nutre nessuno.
#:
#: Stanno qui e non nel kernel perche' `kernel_vitals` importa da questo
#: modulo: una sola definizione, e i due motori non possono divergere su un
#: numero.
_MARGINE = 1.3
_SUPPORTO_CIBO_PRO_CAPITE = 0.1682
_SUPPORTO_ACQUA_PRO_CAPITE = 0.0234
_SUPPORTO_OSSIGENO_PRO_CAPITE = 0.1268
_SUPPORTO_ABITABILITA_PRO_CAPITE = 0.1139
_RAPPORTO_ABITABILITA_ACQUA = 2.75
#: Quota del consumo nominale d'ossigeno che resta dove l'abitabilita' e' piena
#: (il fattore `1,10 - min(1, hab x 6 + ...)` della formula dei vitali).
_QUOTA_OSSIGENO_RESIDUA = 0.10

SUPPORT_FOOD_K = (
    CONSUMO_SAZIETA_PER_PASSO * _MARGINE / _SUPPORTO_CIBO_PRO_CAPITE
)
SUPPORT_HAB_WATER_K = (
    CONSUMO_IDRATAZIONE_PER_PASSO
    * _MARGINE
    / (
        _SUPPORTO_ABITABILITA_PRO_CAPITE
        + _SUPPORTO_ACQUA_PRO_CAPITE * _RAPPORTO_ABITABILITA_ACQUA
    )
)
SUPPORT_WATER_K = SUPPORT_HAB_WATER_K * _RAPPORTO_ABITABILITA_ACQUA
SUPPORT_OXYGEN_K = (
    CONSUMO_OSSIGENO_PER_PASSO
    * _QUOTA_OSSIGENO_RESIDUA
    * _MARGINE
    / _SUPPORTO_OSSIGENO_PRO_CAPITE
)
SUPPORT_HEAL_K = 0.3000


#: **La razione che il sistema di supporto eroga a ogni colono, ogni passo.**
#:
#: E' la stessa massa di un pasto (`EAT_FOOD`, 0,1 unita') e di una bevuta
#: (`DRINK_WATER`, 0,1): il modello aveva gia' definito quanto vale una
#: razione, e il supporto passivo la erogava senza pagarla.
#:
#: **Perche' e' questo il punto in cui la conservazione si rompeva.** Il
#: meccanismo che tiene vivi i coloni non e' l'aritmetica dei vitali — che
#: scorre su `dt = 7/3650`, cioe' 2,9e-5 di sazieta' per passo, e un solo
#: pasto ne coprirebbe ottomila — ma i CONTATORI di privazione, che uccidono
#: in 7 passi senz'acqua e 14 senza cibo. E quei contatori si azzeravano al
#: solo esistere di una struttura con un effetto alimentare positivo,
#: qualunque fosse la giacenza. Una serra sopravvissuta bastava a nutrire
#: chiunque, per sempre: e' la ragione misurata per cui distruggere l'80%
#: delle strutture non produceva ne' morti ne' sofferenza.
#:
#: Ora la razione e' massa vera, prelevata dalla giacenza della cella. Se la
#: cella e' vuota il contatore riparte, il colono deve mangiare dal proprio
#: inventario o andare a procurarsi il necessario, e una produzione
#: interrotta si sente entro una stagione invece che mai.
RAZIONE_CIBO = 0.1
RAZIONE_ACQUA = 0.1
RAZIONE_OSSIGENO = 0.1


#: **Le linee di base dello strato psicosociale.**
#:
#: Sono i valori con cui il modello dichiara che nasce un colono nominale
#: (`BaseAgent`: `stress_index` 0,15, `morale` 0,82, `cooperation` 0,70,
#: `protocol_compliance` 0,85, e le stesse medie nelle gaussiane di
#: `_assign_psychological_profile`). Non sono numeri nuovi: sono l'unico
#: valore che il modello aveva gia' dichiarato come "condizione normale", ed
#: e' verso quello che le quattro variabili ora tornano quando le spinte si
#: annullano. Un test verifica che restino uguali ai default di `BaseAgent`:
#: se qualcuno li cambia li', vanno cambiati anche qui.
LINEA_BASE_STRESS = 0.15
LINEA_BASE_MORALE = 0.82
LINEA_BASE_COOPERAZIONE = 0.70
LINEA_BASE_ADERENZA = 0.85

#: **Il sostegno massimo, misurato.** E' l'unica delle spinte che non ha un
#: tetto strutturale: `habitability_support` e `healing_bonus` sono somme di
#: effetti di struttura divise per gli occupanti, e nulla le limita per
#: costruzione. Misurato su tre semi a 400 passi con il layer acceso, 2033
#: campioni: mediana 0,325, p99 0,396, massimo 0,560.
SOSTEGNO_MASSIMO_MISURATO = 0.56

#: **Privazione massima strutturale**: i tre vitali a zero valgono 0,55
#: ciascuno per costruzione della formula, la fatica piena vale 1,00 - 0,55.
PRIVAZIONE_MASSIMA = 0.55 * 3 + 0.45

#: **Spinta ambientale massima strutturale**: radiazione 1,0 oltre la soglia
#: di 0,8 e polvere 1,0.
SPINTA_AMBIENTALE_MASSIMA = (1.0 - 0.8) * 0.05 + 1.0 * 0.035

#: **Ritardo di comunicazione massimo**: e' il tetto gia' scritto nella
#: formula (`min(0,055, ...)`), con autonomia nulla.
RITARDO_MASSIMO = 0.055

#: **I tassi di rilassamento, e la sola regola che li ricava.**
#:
#: Fino al 2026-08-31 le quattro variabili erano integratori puri:
#: `x += (spinte) * passo`, senza alcun termine proporzionale a `x`. Una
#: variabile che integra una costante non ha punto fisso, e i `clip` a [0,1]
#: non sono un equilibrio ma un muro. Misurato su tre semi a 400 passi con il
#: layer acceso, la quota di coloni ESATTAMENTE su 0 o su 1:
#:
#:   stress 98-100%   morale 97-100%   cooperazione 87-94%   aderenza 91-98%
#:
#: Uno strato in cui nove coloni su dieci stanno sul muro non misura piu'
#: nulla: coesione, rischio di conflitto e aderenza diventano costanti
#: travestite da variabili.
#:
#: La forma con equilibrio aggiunge un richiamo verso la linea di base:
#:
#:   x += (spinte - tasso * (x - linea_di_base)) * passo
#:
#: e il punto fisso diventa `linea_di_base + spinte / tasso`, cioe' la
#: variabile smette di accumulare la storia e misura la CONDIZIONE CORRENTE.
#:
#: Il tasso non e' scelto: risolve, per ciascuna variabile, l'equazione
#:
#:   spinta_massima_in_salita / tasso = spazio fra la linea di base e il tetto
#:
#: cioe' la condizione piu' forte che il modello sa produrre porta la
#: variabile esattamente al tetto della scala, e nulla di meno forte ci
#: arriva. La direzione scelta e' quella in salita perche' e' li' che stava il
#: muro che ha rotto la misura (morale, cooperazione e aderenza incollate a 1)
#: e perche' per lo stress e' l'unica che deve restare raggiungibile: la
#: soglia di danno a 0,82 sarebbe codice morto se il tetto non lo fosse.
#: Verso il basso lo zero resta raggiungibile ed e' una lettura legittima
#: ("nessuna pressione"), non un muro: la spinta in discesa piu' forte vale
#: 0,056 e servirebbe un sostegno oltre il p99 per toccarlo.
#:
#: I tempi caratteristici che ne risultano (1/tasso, in passi settimanali):
#: stress 3,5 (25 giorni), morale 2,4 (17 giorni), cooperazione 55,6,
#: aderenza 29,4. I due lenti lo sono perche' i loro pesi sono i piu' piccoli
#: dello strato: sono disposizioni, non stati d'animo.
_SPINTA_STRESS_MASSIMA = (
    0.045                            # isolamento, colono solo
    + PRIVAZIONE_MASSIMA * 0.045     # tre vitali a zero e fatica piena
    + SPINTA_AMBIENTALE_MASSIMA      # radiazione e polvere al massimo
    + RITARDO_MASSIMO                # ritardo con la Terra, autonomia nulla
)
_SPINTA_MORALE_MASSIMA = (
    SOSTEGNO_MASSIMO_MISURATO * 0.075
    + 0.18 * (0.008 / 0.045)         # compagnia al suo tetto
)
_SPINTA_COOPERAZIONE_MASSIMA = (1.0 - 0.55) * 0.012      # morale pieno, stress nullo
_SPINTA_ADERENZA_MASSIMA = (1.0 - 0.0 - 0.15) * 0.006    # morale pieno, stress nullo

TASSO_RILASSAMENTO_STRESS = _SPINTA_STRESS_MASSIMA / (1.0 - LINEA_BASE_STRESS)
TASSO_RILASSAMENTO_MORALE = _SPINTA_MORALE_MASSIMA / (1.0 - LINEA_BASE_MORALE)
TASSO_RILASSAMENTO_COOPERAZIONE = _SPINTA_COOPERAZIONE_MASSIMA / (1.0 - LINEA_BASE_COOPERAZIONE)
TASSO_RILASSAMENTO_ADERENZA = _SPINTA_ADERENZA_MASSIMA / (1.0 - LINEA_BASE_ADERENZA)


def verso_l_equilibrio(valore, linea_di_base: float, spinta,
                       tasso: float, passo: float):
    """Avvicina una variabile psicosociale al suo equilibrio, sul passo dato.

    L'equazione dello strato e' un ritardo del primo ordine:

        dx/dt = spinta - tasso * (x - linea_di_base)

    e il suo equilibrio e' `linea_di_base + spinta / tasso`. Qui la si integra
    in forma ESATTA su un passo di durata qualunque invece di avanzarla con lo
    schema esplicito `x += (spinta - tasso * (x - base)) * passo`.

    **Perche' non lo schema esplicito.** Il modello espone la durata del passo
    nella configurazione (`dt_days`), e con lo schema esplicito l'equilibrio
    dipende da quella durata: finche' `tasso x passo` resta sotto 1 il punto
    fisso e' quello giusto, sopra 1 la variabile viene prima riportata di colpo
    sulla linea di base e poi spinta di un intero passo di pressione, cioe'
    finisce contro il muro. Misurato: una prova a 365 giorni per passo (52
    passi di riferimento) portava lo stress a 1,000 e uccideva l'equipaggio col
    ramo di danno, mentre l'equilibrio corretto per le stesse spinte vale 0,75.
    Con la forma esatta la stessa colonia simulata a passo settimanale o annuale
    converge allo STESSO equilibrio, che e' la proprieta' che rende confrontabili
    due run con passo diverso.

    Funziona identica su un float e su un ndarray -- usa solo aritmetica e un
    `exp` fra scalari -- quindi i due motori eseguono questa stessa riga invece
    di due trascrizioni che possono divergere.
    """
    equilibrio = linea_di_base + spinta / tasso
    return (equilibrio - valore) * (1.0 - math.exp(-tasso * passo))


def preleva_razione(cell, risorsa: str, razione: float) -> bool:
    """Toglie una razione dalla giacenza della cella; falso se non c'e'.

    Il prelievo e' sequenziale per costruzione nel motore a oggetti (un
    agente per volta), e il kernel lo riproduce con `_serialized_ration_draw`:
    due coloni nella stessa cella non possono attingere alla stessa unita'.
    """
    giacenza = float(getattr(cell.resources, risorsa, 0.0))
    if giacenza + RATION_EPSILON < razione:
        return False
    setattr(cell.resources, risorsa, max(0.0, giacenza - razione))
    return True


def cell_occupants(cell) -> float:
    """Quanti coloni si spartiscono il supporto di questa cella, almeno uno.

    Il minimo a uno non e' una comodita' numerica: una cella vuota non ha
    nessuno da servire, e dividere per zero darebbe un supporto infinito
    proprio dove non serve a nessuno.
    """
    return max(1.0, float(len(getattr(cell, "agents_present", ()) or ())))


def oxygen_support(cell) -> float:
    """Ossigeno prodotto dalle strutture, PER OCCUPANTE.

    **La divisione e' la correzione dimensionale del 2026-08-25.** Queste
    somme sono quantita' estensive — crescono con le strutture della cella —
    ed erano lette come dose ricevuta da ciascun colono. Misurato a piano di
    costruzione rispettato: un colono consuma 0,105 di sazieta' per passo e
    ne riceveva 15,68 a trecento abitanti, 31,36 a seicento. Il supporto
    cresceva con la colonia mentre il bisogno pro capite restava fermo, ed e'
    la ragione per cui nessuna scarsita', nessun evento e nessuna politica
    riuscivano a mettere in difficolta' la colonia.

    Diviso, il supporto pro capite diventa indipendente dalla taglia quando il
    piano e' rispettato: 0,213 di cibo per colono da trecento abitanti in su.
    """
    return sum(float(s.local_effect.get("oxygen", 0.0)) for s in cell.structures) / cell_occupants(cell)


def healing_bonus(cell) -> float:
    return sum(float(s.local_effect.get("healing_bonus", 0.0)) for s in cell.structures) / cell_occupants(cell)


def habitability_support(cell) -> float:
    return sum(float(s.local_effect.get("habitability", 0.0)) for s in cell.structures) / cell_occupants(cell)


def food_support(cell) -> float:
    return sum(float(s.local_effect.get("food", 0.0)) for s in cell.structures) / cell_occupants(cell)


def water_support(cell) -> float:
    return sum(float(s.local_effect.get("water", 0.0)) for s in cell.structures) / cell_occupants(cell)


def consume_water_ration(agent) -> bool:
    if getattr(agent.inventory, "water", 0.0) + RATION_EPSILON >= 0.1:
        agent.inventory.water = max(0.0, agent.inventory.water - 0.1)
        return True
    if getattr(agent.inventory, "ice", 0.0) + RATION_EPSILON >= 0.1:
        agent.inventory.ice = max(0.0, agent.inventory.ice - 0.1)
        return True
    return False


def consume_food_ration(agent) -> bool:
    if getattr(agent.inventory, "food", 0.0) + RATION_EPSILON >= 0.1:
        agent.inventory.food = max(0.0, agent.inventory.food - 0.1)
        return True
    return False


def consume_oxygen_ration(agent) -> bool:
    if getattr(agent.inventory, "oxygen", 0.0) + RATION_EPSILON >= 0.1:
        agent.inventory.oxygen = max(0.0, agent.inventory.oxygen - 0.1)
        return True
    return False


def death_cause(agent) -> str:
    """Classify a death for autopsy-friendly logs (dead_agents.jsonl, events).

    The 1000-step audits diagnose failures from the corpses: without an
    explicit cause every record needed manual cross-reading of four vitals.
    Order matters — dehydration is the faster clock, so it wins ties.
    """
    if agent.hydration <= 0.01 or int(getattr(agent, "steps_without_water", 0)) >= LETHAL_STEPS_WITHOUT_WATER:
        return "dehydration"
    if agent.satiety <= 0.01 or int(getattr(agent, "steps_without_food", 0)) >= LETHAL_STEPS_WITHOUT_FOOD:
        return "starvation"
    if agent.oxygen_level <= 0.05:
        return "hypoxia"
    return "health_collapse"


def refresh_cell_alerts(agent, cell) -> None:
    """Keep the transient SYSTEM ALERTs aligned with the CURRENT cell.

    Alerts live in memory.active_alerts (not recent_events): they appear while
    the condition holds and vanish the step it clears or the agent moves on —
    a warning about an overcrowded cell left ten steps ago is noise, not
    memory. With the cell-degradation toggle OFF no alert is ever raised.
    """
    memory = getattr(agent, "memory", None)
    if memory is None:
        return
    if not cell_degradation_enabled():
        memory.clear_alert("overcrowding")
        memory.clear_alert("pollution")
        return
    capacity = float(cell.occupancy_capacity())
    excess = float(cell.overcrowding_excess())
    if excess > 0.0:
        memory.set_alert(
            "overcrowding",
            f"SYSTEM ALERT: Overcrowding in cell ({cell.x},{cell.y}): "
            f"{len(cell.agents_present)} occupants for {capacity:.1f} supported places. "
            "Local oxygen is low!",
        )
    else:
        memory.clear_alert("overcrowding")
    if cell.pollution_risk > 0.04:
        memory.set_alert("pollution", f"SYSTEM ALERT: Dense local activity in cell ({cell.x},{cell.y}) is degrading oxygen quality.")
    else:
        memory.clear_alert("pollution")


def tick_agent_vitals(agent, cell, dt_days: float) -> None:
    """
    Per-step survival needs: oxygen, hydration, fatigue, passive recovery near medical facilities.
    dt_days is usually large (e.g. 3650); scale effects to stay stable.
    """
    # A simulation step can represent years; survival vitals are abstracted per decision step.
    dt = min(1.0, max(0.0, float(dt_days) / 3650.0))
    # **Due scale, e sono deliberate (2026-08-25).** `passo` e' la durata del
    # passo rapportata a quella di riferimento: la usano i vitali di
    # PRIVAZIONE (sazieta', idratazione, ossigeno) e il supporto che li
    # compensa, perche' il consumo di un colono e' per passo e la sua finestra
    # letale si conta in passi. `dt` resta la scala lunga (dt_days/3650) per i
    # processi ambientali — usura da radiazione, esposizione polare, fatica,
    # inquinamento — che si misurano in anni e la cui taratura attuale produce
    # erosioni sensate su orizzonti lunghi.
    #
    # Su quella scala lunga quei termini restano di fatto immobili (la
    # radiazione toglie 0,0027 di salute in cento passi), ed e' un limite noto
    # e dichiarato: portarli sulla scala del passo con i coefficienti attuali
    # ucciderebbe l'intera colonia in un centinaio di passi, e richiede una
    # taratura propria che non e' stata fatta.
    passo = max(0.0, float(dt_days)) / GIORNI_PER_PASSO_RIFERIMENTO

    # **Il supporto e' una capacita', la razione e' la massa che la paga.**
    # Un effetto di struttura positivo dice che l'impianto c'e' e funziona;
    # non dice che ci sia qualcosa da erogare. Il prelievo decide.
    local_support = habitability_support(cell)
    local_water_support = water_support(cell)
    local_food_support = food_support(cell)
    razione_ossigeno = (
        oxygen_support(cell) > 0.0
        and preleva_razione(cell, "oxygen", RAZIONE_OSSIGENO)
    )
    razione_acqua = (
        (local_water_support > 0.0 or local_support > 0.0)
        and preleva_razione(cell, "water", RAZIONE_ACQUA)
    )
    razione_cibo = (
        local_food_support > 0.0
        and preleva_razione(cell, "food", RAZIONE_CIBO)
    )
    if not razione_acqua:
        local_water_support = 0.0
    if not razione_cibo:
        local_food_support = 0.0
    o2_from_struct = (
        oxygen_support(cell) * SUPPORT_OXYGEN_K * passo if razione_ossigeno else 0.0
    )
    hab = max(0.0, min(1.0, cell.habitability_score + local_support))
    
    # Base oxygen drain increases if habitability is low
    o2_drain = CONSUMO_OSSIGENO_PER_PASSO * passo * (1.10 - min(1.0, hab * 6.0 + o2_from_struct * 4.0))
    
    if cell.pollution_risk > 0.0:
        o2_drain += (
            cell.pollution_risk
            * QUOTA_OSSIGENO_DA_INQUINAMENTO
            * CONSUMO_OSSIGENO_PER_PASSO
            * passo
        )
        agent.fatigue = min(
            1.0,
            agent.fatigue
            + cell.pollution_risk * FATICA_DA_INQUINAMENTO_PER_PASSO * passo,
        )

    agent.oxygen_level = max(0.0, min(1.0, agent.oxygen_level - o2_drain + o2_from_struct))
    if o2_from_struct <= 0.0 and agent.oxygen_level < 0.72 and consume_oxygen_ration(agent):
        agent.oxygen_level = min(1.0, agent.oxygen_level + RISTORO_OSSIGENO)

    # L'abitabilita' idrata attraverso il riciclo dell'habitat: e' acqua, e
    # va pagata come tale. Se la razione non e' stata erogata resta la sola
    # umidita' ambientale (acqua liquida e ghiaccio della cella), che non e'
    # un impianto e non si preleva.
    supporto_idrico = (
        local_support * SUPPORT_HAB_WATER_K + local_water_support * SUPPORT_WATER_K
        if razione_acqua
        else 0.0
    )
    agent.hydration = max(
        0.0,
        min(1.0, agent.hydration - CONSUMO_IDRATAZIONE_PER_PASSO * passo + (cell.liquid_water * 0.002 + cell.water_ice * 0.0008 + supporto_idrico) * passo),
    )

    agent.satiety = max(0.0, min(1.0, agent.satiety - CONSUMO_SAZIETA_PER_PASSO * passo + local_food_support * SUPPORT_FOOD_K * passo))

    agent.fatigue = max(
        0.0,
        min(1.0, agent.fatigue + FATICA_AMBIENTALE_PER_PASSO * passo * (1.05 - hab * 1.8)),
    )

    structure_heat = cell.structure_heat() if hasattr(cell, "structure_heat") else 0.0
    cold_modifier = max(0.0, -(float(getattr(cell, "local_temperature_modifier", 0.0)) + structure_heat) - 10.0) / 20.0
    polar_exposure = max(float(getattr(cell, "polar_severity", 0.0)), min(1.0, cold_modifier))
    if polar_exposure > 0.0:
        life_support = min(0.85, habitability_support(cell) + oxygen_support(cell) * 0.25)
        exposure = polar_exposure * max(0.0, 1.0 - life_support)
        agent.fatigue = min(1.0, agent.fatigue + FATICA_POLARE_PER_PASSO * exposure * passo)
        agent.oxygen_level = max(0.0, agent.oxygen_level - OSSIGENO_POLARE_PER_PASSO * exposure * passo)
        agent.health = max(0.0, agent.health - SALUTE_POLARE_PER_PASSO * exposure * passo)
        if exposure >= 0.55:
            agent.memory.add_event("Severe polar cold exposure: retreat or build life support before sustained work.")

    shielding = min(0.75, local_support * 2.5)
    rad_wear = USURA_RADIAZIONE_PER_PASSO * passo * cell.radiation_level * (1.0 - shielding)
    # Le penalita' da privazione seguono i vitali che le generano: su `dt`
    # sarebbero nulle proprio quando il colono sta peggio.
    hypoxia = max(0.0, 0.35 - agent.oxygen_level) * 0.04 * passo
    dehydrate = max(0.0, 0.3 - agent.hydration) * 0.025 * passo
    starvation = max(0.0, 0.25 - agent.satiety) * 0.03 * passo
    agent.health = max(0.0, min(1.0, agent.health - rad_wear - hypoxia - dehydrate - starvation))

    drank_this_step = bool(agent.recent_actions and agent.recent_actions[-1] == "drink_water")
    ate_this_step = bool(agent.recent_actions and agent.recent_actions[-1] == "eat_food")
    # **La razione personale disseta e sfama davvero (2026-08-25).** Prima
    # azzerava soltanto il CONTATORE di privazione: la borraccia si svuotava,
    # il colono risultava "rifornito", e l'idratazione continuava a scendere.
    # Era invisibile finche' l'idratazione non si muoveva (2,9e-5 per passo);
    # sulla scala del passo diventa un colono che beve la sua scorta e muore
    # di sete lo stesso. La razione d'ossigeno gia' faceva la cosa giusta
    # (+0,25), ed e' quella che fa da modello: acqua e cibo usano gli stessi
    # cambi delle azioni corrispondenti, +0,35 di idratazione per 0,1 d'acqua
    # e +0,25 di sazieta' per 0,1 di cibo.
    used_water_ration = not razione_acqua and not drank_this_step and consume_water_ration(agent)
    if used_water_ration:
        agent.hydration = min(1.0, agent.hydration + RISTORO_ACQUA)
    used_food_ration = not razione_cibo and not ate_this_step and consume_food_ration(agent)
    if used_food_ration:
        agent.satiety = min(1.0, agent.satiety + RISTORO_CIBO)

    if drank_this_step or used_water_ration or razione_acqua:
        agent.steps_without_water = 0
    else:
        agent.steps_without_water = int(getattr(agent, "steps_without_water", 0)) + 1
        agent.hydration = min(
            agent.hydration,
            max(0.0, 1.0 - agent.steps_without_water / LETHAL_STEPS_WITHOUT_WATER),
        )

    if ate_this_step or used_food_ration or razione_cibo:
        agent.steps_without_food = 0
    else:
        agent.steps_without_food = int(getattr(agent, "steps_without_food", 0)) + 1
        agent.satiety = min(
            agent.satiety,
            max(0.0, 1.0 - agent.steps_without_food / LETHAL_STEPS_WITHOUT_FOOD),
        )

    if agent.hydration <= 0.0:
        agent.health = 0.0
        agent.memory.add_event(
            f"Died from dehydration after {agent.steps_without_water} consecutive steps without drinking."
        )
    elif agent.steps_without_water >= PASSI_SINTOMO_SENZA_ACQUA:
        agent.health = max(0.0, agent.health - 0.12 * dt)
        agent.fatigue = min(1.0, agent.fatigue + 0.10 * dt)
        agent.memory.add_event(
            f"Dehydration symptoms after {PASSI_SINTOMO_SENZA_ACQUA} "
            "consecutive steps without drinking."
        )

    if agent.satiety <= 0.0:
        agent.health = 0.0
        agent.memory.add_event(
            f"Died from starvation after {agent.steps_without_food} consecutive steps without food."
        )
    elif agent.steps_without_food >= PASSI_SINTOMO_SENZA_CIBO:
        agent.health = max(0.0, agent.health - 0.08 * dt)
        agent.fatigue = min(1.0, agent.fatigue + 0.08 * dt)
        agent.memory.add_event(
            f"Starvation symptoms after {PASSI_SINTOMO_SENZA_CIBO} "
            "consecutive steps without food."
        )

    refresh_cell_alerts(agent, cell)

    if agent.health <= 0.0:
        return

    hb = healing_bonus(cell)
    if hb > 0:
        # I rapporti fra i quattro effetti sono quelli originali
        # (0,03 / 0,02 / 0,015 / 0,03): cambia solo la scala, ritarata sul
        # `healing_bonus` normalizzato. Prima della normalizzazione una
        # capitale rigenerava +1,73 di salute per passo contro un danno
        # massimo di 0,10 da un brillamento, ed era il motivo per cui non
        # moriva mai nessuno.
        # La cura e' un processo per passo come il vitto: sulla scala lunga
        # un'infermeria restituiva 3,2e-4 di salute a settimana, meno di
        # quanto le penalita' da privazione tolgano nello stesso passo.
        agent.health = min(1.0, agent.health + hb * SUPPORT_HEAL_K * passo)
        agent.oxygen_level = min(1.0, agent.oxygen_level + hb * SUPPORT_HEAL_K * 0.667 * passo)
        agent.hydration = min(1.0, agent.hydration + hb * SUPPORT_HEAL_K * 0.5 * passo)
        agent.fatigue = max(0.0, agent.fatigue - hb * SUPPORT_HEAL_K * passo)


def tick_agent_psychosocial(agent, cell, world: GridWorld, dt_days: float, config: dict | None = None) -> None:
    """
    Deterministic psychosocial pressure from isolated/confined spaceflight analogues:
    isolation, fatigue, unmet survival needs, dangerous terrain and communication delay raise stress;
    nearby crew and habitable infrastructure support morale.
    """
    # **Anche il layer psicosociale e' per passo (2026-08-25).** Su `dt` uno
    # stress cronico toglieva 2,3e-5 di salute a settimana: il ramo che
    # dichiara la morte da stress non poteva scattare in nessuna run di durata
    # realistica. E' lo stesso difetto dei vitali di privazione, in un altro
    # strato. Il layer resta comunque spento per default
    # (`psychosocial_enabled`). La variabile `dt` della vecchia scala e'
    # stata rimossa il 2026-08-31: era assegnata e mai usata, e la sua sola
    # presenza faceva dubitare che il layer girasse ancora sulla scala
    # annuale.
    passo = max(0.0, float(dt_days)) / GIORNI_PER_PASSO_RIFERIMENTO
    cfg = config or {}
    social_cfg = cfg.get("social", {}) if isinstance(cfg.get("social", {}), dict) else {}
    delay_minutes = float(social_cfg.get("earth_mars_delay_minutes", 12.0) or 0.0)
    nearby = max(0, len(world.get_agents_in_range(agent.x, agent.y, 1)) - 1)
    compagnia = min(0.18, nearby * 0.045)
    support = habitability_support(cell) + healing_bonus(cell) * 0.5 + compagnia
    deprivation = (
        max(0.0, 0.55 - agent.oxygen_level)
        + max(0.0, 0.55 - agent.hydration)
        + max(0.0, 0.55 - agent.satiety)
        + max(0.0, agent.fatigue - 0.55)
    )
    environmental_pressure = max(0.0, cell.radiation_level - 0.8) * 0.05 + cell.dust_level * 0.035
    # **La compagnia era contata due volte, e ora una sola (2026-08-31).**
    # Il ramo `else` valeva `-0,025 x min(3, vicini)`: con tre o piu' vicini
    # toglieva 0,0750 di stress a ogni passo, mentre lo stesso beneficio era
    # gia' dentro `support` attraverso `compagnia`, li' con un tetto a 0,18.
    # Gli stessi vicini valevano due volte e la seconda senza tetto, ed e'
    # quel termine a spingere il 98% dei coloni sul muro dello zero: la
    # mediana dei vicini misurata e' 78, quindi il ramo era sempre attivo e
    # sempre al suo massimo.
    #
    # Toglierlo era gia' stato provato il 2026-08-31 e RITIRATO, perche'
    # senza punto fisso spostava solo la saturazione all'altro muro (stress
    # medio 0,1791 / 0,7556 / 0,9272 sui tre semi, oltre la soglia di 0,82
    # che degrada la salute). Con il richiamo verso la linea di base quel
    # muro non c'e' piu', e la correzione puo' stare in piedi.
    #
    # Ora la variabile dice cio' che il suo nome dice: l'isolamento e' una
    # pressione, la compagnia e' l'assenza di quella pressione, e il suo
    # beneficio resta dove era gia' contato.
    isolation_pressure = 0.045 if nearby == 0 else 0.0
    delay_pressure = min(0.055, delay_minutes / 22.0 * 0.035) * (1.0 - agent.autonomy_preference * 0.45)
    spinta_stress = (
        isolation_pressure + deprivation * 0.045 + environmental_pressure
        + delay_pressure - support * 0.10
    )
    agent.stress_index = max(0.0, min(1.0, agent.stress_index + verso_l_equilibrio(
        agent.stress_index, LINEA_BASE_STRESS, spinta_stress,
        TASSO_RILASSAMENTO_STRESS, passo)))
    # **La compagnia ha un tetto anche nel morale (2026-08-31).** Lo stesso
    # gruppo di vicini contribuiva al massimo 0,18 a `support`, dove il
    # tetto c'era, e un valore NON limitato al morale: con novantanove
    # vicini valeva 0,792 per passo, cioe' il morale saturava a uno in un
    # solo passo. Misurato su tre semi a 400 passi con il layer acceso:
    # stress medio 0,0000-0,0128 e morale 0,9972-1,0000, con ogni singolo
    # colono a 0,000 e 1,000 esatti sul seme 33. Un layer che satura non
    # misura piu' nulla, e gli indici che ne discendono -- coesione,
    # rischio di conflitto, autonomia -- diventano costanti.
    #
    # Il tetto e' lo STESSO gia' usato per `support`, riscalato: poiche'
    # `vicini x 0,008` e' `(vicini x 0,045) x (0,008/0,045)`, limitare il
    # prodotto interno lascia il termine identico sotto il tetto e lo
    # rende limitato sopra. Nessuna costante nuova.
    spinta_morale = (
        support * 0.075 + compagnia * (0.008 / 0.045)
        - agent.stress_index * 0.045 - deprivation * 0.025
    )
    agent.morale = max(0.0, min(1.0, agent.morale + verso_l_equilibrio(
        agent.morale, LINEA_BASE_MORALE, spinta_morale,
        TASSO_RILASSAMENTO_MORALE, passo)))
    spinta_cooperazione = (agent.morale - 0.55) * 0.012 - agent.stress_index * 0.008
    agent.cooperation = max(0.0, min(1.0, agent.cooperation + verso_l_equilibrio(
        agent.cooperation, LINEA_BASE_COOPERAZIONE, spinta_cooperazione,
        TASSO_RILASSAMENTO_COOPERAZIONE, passo)))
    spinta_aderenza = (agent.morale - agent.stress_index - 0.15) * 0.006
    agent.protocol_compliance = max(0.0, min(1.0, agent.protocol_compliance
        + verso_l_equilibrio(agent.protocol_compliance, LINEA_BASE_ADERENZA,
                             spinta_aderenza, TASSO_RILASSAMENTO_ADERENZA, passo)))
    if agent.stress_index > 0.82:
        agent.health = max(0.0, agent.health - 0.012 * passo)
        agent.memory.add_event("High stress is degrading performance; prioritize rest, communication, or habitat support.")


def tick_all_agents(agents: list, world: GridWorld, dt_days: float, config: dict | None = None) -> None:
    # tick_agent_vitals normalizes dt_days internally (dt = min(1, dt_days/3650)), so the
    # per-step survival pressure saturates at deep-time steps instead of exploding; an
    # extra day-cap here would double-scale and freeze all continuous drains.
    cfg = config or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    psychosocial_enabled = bool(model_cfg.get("psychosocial_enabled", False))
    for agent in agents:
        cell = world.get_cell(agent.x, agent.y)
        tick_agent_vitals(agent, cell, dt_days)
        if psychosocial_enabled:
            tick_agent_psychosocial(agent, cell, world, dt_days, config=config)
