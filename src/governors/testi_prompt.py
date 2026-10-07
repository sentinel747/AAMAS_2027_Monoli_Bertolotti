# -*- coding: utf-8 -*-
"""Il testo dei prompt, nelle due lingue.

**Perche' il testo sta qui e non dentro le funzioni che lo compongono.** La
lingua e' una variabile sperimentale (`varianti_prompt.py`), e finche' le frasi
stavano dentro `build_governor_prompt` la seconda lingua avrebbe richiesto di
duplicare la funzione. Due copie della stessa logica divergono: basta che una
riga di dati venga aggiunta a una sola delle due e i bracci smettono di
differire per la sola lingua, che e' esattamente cio' che la campagna misura.

Qui dentro c'e' solo prosa. L'assemblaggio --- quali blocchi entrano a quale
gradino di contesto, quali numeri ci vanno dentro --- resta in `llm_arm.py` e in
`administration.py` ed e' scritto una volta sola per tutte le lingue.

**L'italiano e' congelato.** Le stringhe italiane riproducono parola per parola
il prompt con cui sono state misurate tutte le campagne dell'archivio; i file
d'oro in `tests/governors/oro/` lo verificano a ogni esecuzione dei test. Se una
di queste righe cambia, i numeri gia' raccolti smettono di essere confrontabili
con quelli nuovi e il test lo dice.

**L'inglese e' una traduzione, non una riscrittura.** Ogni blocco dice le stesse
cose nello stesso ordine e con lo stesso peso: un inglese piu' chiaro
dell'italiano misurerebbe la chiarezza e non la lingua. Dove l'italiano e'
goffo, l'inglese resta goffo allo stesso modo.

**Cio' che NON si traduce mai:** i nomi degli indicatori
(`food_per_occupant`...), quelli delle categorie pesabili (`sustenance`...), gli
operatori e le chiavi del JSON di risposta. Sono gia' inglesi da sempre e sono
il contratto con la macchina: se cambiassero, i due bracci non differirebbero
per la lingua ma per il protocollo.
"""

from __future__ import annotations

# --------------------------------------------------------------------- italiano

_IT = {
    # llm_arm: apertura al gradino che dichiara il dominio
    "ruolo": (
        "Sei il governatore della colonia marziana. Non sei sulla "
        "superficie: osservi gli aggregati e scrivi la POLITICA con cui "
        "ogni colono decidera' da solo.\n\n"
        "Come decide un colono: a ogni passo campiona {uno_fra_sei} "
        "con punteggi = preferenze individuali x urgenze personali x "
        "priorita' della cella x abilita'. La tua politica moltiplica {i_plurale}"
        ", cella per cella, dove le sue condizioni scattano. Dove "
        "nessuna regola scatta, il colono sceglie con le sole proprie "
        "preferenze. Un peso non puo' rendere possibile un'azione che la "
        "cella vieta.\n"
    ),
    # llm_arm: apertura ai gradini che il dominio non lo dichiarano
    "ruolo_neutro": (
        "Osservi un sistema attraverso nove indicatori misurati su ciascuna "
        "delle sue unita', e scrivi le regole con cui le unita' "
        "ripartiranno il proprio lavoro fra cinque attivita'. Le regole "
        "vengono valutate su ogni unita' a ogni passo, senza chiederti "
        "altro.\n"
    ),
    "aiuti": (
        "\nLa politica di riferimento del braccio non-LLM e':\n"
        "{regole}.\n"
        "Non sei obbligato a quella: puoi scrivere qualunque politica nel "
        "vocabolario qui sotto - soglie, pesi e {plurale_diversi} - e devi "
        "spiegare perche'.\n"
    ),
    "dati": (
        "\nPasso: {step}. Popolazione: {popolazione}. "
        "Unita' occupate: {celle}.\n"
        "Aggregati:\n{metriche}\n\n"
        "Distribuzione degli indicatori sulle unita' occupate (mean/std/min/max):\n"
        "{indicatori}\n\n"
        "Gli stessi indicatori PESATI per {colono} (ogni unita' pesa quanti "
        "{coloni} ospita; mean/std pesate, min/max per unita'):\n"
        "{indicatori_pesati}\n\n"
    ),
    "code_dati": (
        "Medie di popolazione:\n{medie}\n"
        "Totali di struttura:\n{strutture}\n\n"
    ),
    "dove_stanno": (
        "Dove stanno gli {coloni}: la unita' piu' popolata, {cella}, ospita "
        "{occupanti} {coloni} ({quota} della {colonia}). I suoi "
        "indicatori: {valori}. Una regola che non "
        "scatta li' non tocca il {quota} degli {coloni}.\n\n"
    ),
    "morti_nessuno": "Morti dall'ultimo tick: nessuno.\n\n",
    "morti_intestazione": "Morti dall'ultimo tick: {totale}, per unita' e causa:\n",
    "morti_riga": "  unita' {cella}: {dettaglio}",
    "avvertenze_intestazione": "Avvertenze sullo scenario corrente:\n",
    "avvertenza_zero": (
        "- {nome} e' a ZERO su tutte le celle occupate: una condizione "
        "`<` con soglia positiva scatta ovunque, una `>` mai. Una regola "
        "cosi', scritta per prima, OSCURA tutte le regole successive: "
        "nessuna potra' piu' scattare."
    ),
    "avvertenza_piatta": (
        "- {nome} vale {valore} su TUTTE le celle occupate: una "
        "soglia non distingue una cella dall'altra, accende o spegne "
        "la regola per tutte insieme."
    ),
    "avvertenza_una_cella": (
        "- c'e' UNA sola cella occupata: ogni condizione vale "
        "tutto-o-niente finche' la colonia non si espande."
    ),
    "consuntivo_intestazione": (
        "Che cosa ha fatto la TUA politica precedente, dall'ultimo aggiornamento "
        "(per ogni cella vale la prima regola che scatta, quindi una regola larga "
        "scritta per prima toglie le celle a tutte le successive):\n"
    ),
    "consuntivo_altrimenti": "altrimenti (nessuna regola)",
    "consuntivo_regola": "regola {indice}",
    "consuntivo_regola_con_testo": "regola {indice}: {testo}",
    "consuntivo_riga": "- {etichetta}: {quante} celle-passo{coda}",
    "consuntivo_ombra": "  <-- IN OMBRA: non ha mai scattato",
    "vocabolario": (
        "Vocabolario della politica.\n"
        "Indicatori (le condizioni si valutano su OGNI unita', a ogni passo):\n"
        "{indicatori}\n"
        "Operatori: < e >.\n"
        "{pesabili}: {categorie}.\n"
    ),
    "glossario": (
        "{glossario_apertura}\n"
        "- sustenance: bere, mangiare, raccogliere ghiaccio, coltivare e "
        "raccogliere cibo. {glossario_fame}\n"
        "- resources: raccogliere minerali dal giacimento e construction_material "
        "dal magazzino della cella. ATTENZIONE: raccogliere construction_material "
        "non ne PRODUCE — lo sposta dal magazzino alle tasche dei coloni, e le "
        "costruzioni attingono gia' direttamente dal magazzino. Il materiale "
        "nuovo nasce dai depositi (e dall'ISRU, se attivo); i minerali dal "
        "giacimento. Pesare `resources` per 'avere piu' materiale' e' quindi "
        "lavoro sprecato: serve solo dove i coloni devono portarselo dietro "
        "(spostamenti, fondazioni).\n"
        "- build: erigere strutture (serre, habitat, pannelli, laboratori...) e "
        "mantenerle. Una struttura il cui rapporto per abitante e' gia' "
        "soddisfatto non viene costruita comunque.\n"
        "- life: riposo, cure, recupero. Non produce nulla di per se'.\n"
        "- explore: muoversi ed esplorare, cioe' fondare nuovi insediamenti.\n"
    ),
    "esempio_da_proteggere": " (per esempio explore, senza cui nessuno fonda nuove celle)",
    "chiusura": (
        "Un peso e' un moltiplicatore fra {peso_min} e "
        "{peso_max} (1.0 = nessun cambiamento; sotto 1 scoraggia, "
        "sopra 1 incoraggia).\n"
        "I pesi sono RELATIVI {fra_le_plurale}: i punteggi vengono rinormalizzati, "
        "quindi alzare {tre_plurale} a x2-x3 lasciando le altre a 1 equivale a "
        "PENALIZZARE queste ultime, che perdono quota nel campionamento. {una_da_proteggere} "
        "che vuoi proteggere{esempio} {va_alzata}, "
        "non lasciata a 1.\n"
        "Al massimo {max_regole} regole. Per ogni unita' vale la PRIMA regola "
        "che scatta, nell'ordine in cui le scrivi; le altre vengono ignorate. "
        "Una regola con \"if\": null scatta sempre e chiude la catena.\n\n"
        "Rispondi SOLO con questo JSON:\n"
        '{{"policy": [{{"if": {{"indicator": "<nome>", "op": "<|>", '
        '"value": <float>}} | null, "weights": {{"<{segnaposto}>": <float>}}}}], '
        '"rationale": "<perche\'>"}}\n'
    ),
    # parole del dominio, o le loro versioni neutre al gradino cieco
    # --- administration: il prompt dell'amministratore di distretto
    "amm_ruolo": (
        "Sei l'amministratore del distretto {distretto} di una colonia "
        "marziana. Il governo centrale ha appena emanato la politica con cui i "
        "coloni scelgono che cosa fare. Tu amministri "
        "{celle} celle e non le altre.\n\n"
    ),
    "amm_catena": (
        "La politica COMPLETA del governo, nell'ordine in cui vale (per ogni "
        "cella conta la PRIMA regola che scatta):\n{catena}\n\n"
        "Ecco che cosa questa politica produce, cella per cella, nel "
        "TUO distretto:\n"
    ),
    "amm_scelte": (
        "Hai due scelte, e nessuna delle due e' quella di riserva:\n"
        "  ACCETTA se la politica del governo e' gia' adeguata alle tue celle: "
        "le applicheranno cosi' come sono.\n"
        "  RISCRIVI se non lo e': la tua catena prende il posto di quella del "
        "governo, per le tue celle e solo per quelle. Puoi ripartire dalle "
        "regole del governo e cambiarne soglie, pesi o ordine, oppure "
        "scriverne di nuove: cio' che consegni e' la legge INTERA delle tue "
        "celle, non una modifica.\n"
        "Accettare una politica che funziona e' corretto quanto correggerne una "
        "che non funziona. Accettare per inerzia una politica che non interviene "
        "sulle tue celle, no: se sopra leggi NESSUN INTERVENTO, il governo ti "
        "sta lasciando senza politica, e la decisione e' interamente tua.\n\n"
        "Passo: {step}. Coloni nel tuo distretto: {popolazione}.\n"
    ),
    "amm_dati": (
        "\nMetriche dell'intera colonia (il quadro nazionale):\n{metriche}\n\n"
        "Distribuzione degli indicatori sulle TUE celle (mean/std/min/max):\n"
        "{indicatori}\n\n"
        "Gli stessi indicatori sull'INTERA colonia, per confronto:\n"
        "{indicatori_colonia}\n\n"
        "Stato dei coloni del tuo distretto (media e coda pericolosa: min per "
        "salute/idratazione/sazieta'/morale, max per fatica/stress):\n"
        "{stato}\n\n"
    ),
    "amm_vocabolario": (
        "Vocabolario (identico a quello del governo).\n"
        "Indicatori di cella:\n{indicatori}\n"
        "Operatori: < e >.\n"
        "{pesabili}: {categorie}.\n"
        "Un peso e' un moltiplicatore fra {peso_min} e "
        "{peso_max} (1.0 = nessun cambiamento).\n"
        "I pesi sono RELATIVI: i punteggi vengono rinormalizzati, quindi "
        "alzarne tre lasciando gli altri a 1 equivale a penalizzare questi "
        "ultimi. {una_da_proteggere} che vuoi proteggere {va_alzata}.\n"
        "Al massimo {max_regole} regole, e per ogni cella vale la PRIMA che "
        "scatta.\n\n"
        "Rispondi SOLO con uno di questi due JSON:\n"
        '{{"accept": true, "rationale": "<perche\' la politica del governo va bene>"}}\n'
        '{{"accept": false, "policy": [{{"if": {{"indicator": "<nome>", "op": "<|>", '
        '"value": <float>}} | null, "weights": {{"<{segnaposto}>": <float>}}}}], '
        '"rationale": "<perche\' la cambio>"}}\n'
    ),
    "amm_morti_nessuno": "Morti nel tuo distretto dall'ultima tornata: nessuno.\n",
    "amm_morti": (
        "Morti nel tuo distretto dall'ultima tornata: {totale}{quota} \u2014 {dettaglio}.\n"
    ),
    "amm_morti_quota": " ({quota} dei tuoi coloni)",
    # --- administration, gradino cieco: stessa struttura, nessun dominio
    "amm_ruolo_neutro": (
        "Amministri {celle} unita' di un sistema, e non le altre. Un livello "
        "centrale ha appena emanato le regole con cui ogni unita' ripartisce il "
        "proprio lavoro fra cinque attivita'.\n\n"
    ),
    "amm_catena_neutro": (
        "Le regole COMPLETE del livello centrale, nell'ordine in cui valgono "
        "(per ogni unita' conta la PRIMA che scatta):\n{catena}\n\n"
        "Ecco che cosa queste regole producono, unita' per unita', fra le TUE:\n"
    ),
    "amm_scelte_neutro": (
        "Hai due scelte, e nessuna delle due e' quella di riserva:\n"
        "  ACCETTA se le regole del livello centrale vanno gia' bene per le tue "
        "unita': verranno applicate cosi' come sono.\n"
        "  RISCRIVI se non vanno bene: la tua catena prende il posto di quella "
        "centrale, per le tue unita' e solo per quelle. Puoi ripartire dalle "
        "regole centrali e cambiarne soglie, pesi o ordine, oppure scriverne di "
        "nuove: cio' che consegni e' la catena INTERA delle tue unita', non una "
        "modifica.\n"
        "Accettare regole che funzionano e' corretto quanto correggerne di "
        "sbagliate. Accettare per inerzia regole che non intervengono sulle tue "
        "unita', no: se sopra leggi NESSUN INTERVENTO, il livello centrale ti "
        "sta lasciando senza regole, e la decisione e' interamente tua.\n\n"
        "Passo: {step}. Occupanti nelle tue unita': {popolazione}.\n"
    ),
    "amm_dati_neutro": (
        "\nMisure dell'intero sistema:\n{metriche}\n\n"
        "Distribuzione degli indicatori sulle TUE unita' (mean/std/min/max):\n"
        "{indicatori}\n\n"
        "Gli stessi indicatori sull'INTERO sistema, per confronto:\n"
        "{indicatori_colonia}\n\n"
        "Stato degli occupanti delle tue unita' (media e coda pericolosa):\n"
        "{stato}\n\n"
    ),
    "amm_vocabolario_neutro": (
        "Vocabolario (identico a quello del livello centrale).\n"
        "Indicatori di unita':\n{indicatori}\n"
        "Operatori: < e >.\n"
        "{pesabili}: {categorie}.\n"
        "Un peso e' un moltiplicatore fra {peso_min} e "
        "{peso_max} (1.0 = nessun cambiamento).\n"
        "I pesi sono RELATIVI: i punteggi vengono rinormalizzati, quindi "
        "alzarne tre lasciando gli altri a 1 equivale a penalizzare questi "
        "ultimi. {una_da_proteggere} che vuoi proteggere {va_alzata}.\n"
        "Al massimo {max_regole} regole, e per ogni unita' vale la PRIMA che "
        "scatta.\n\n"
        "Rispondi SOLO con uno di questi due JSON:\n"
        '{{"accept": true, "rationale": "<perche\' le regole centrali vanno bene>"}}\n'
        '{{"accept": false, "policy": [{{"if": {{"indicator": "<nome>", "op": "<|>", '
        '"value": <float>}} | null, "weights": {{"<{segnaposto}>": <float>}}}}], '
        '"rationale": "<perche\' le cambio>"}}\n'
    ),
    "amm_morti_nessuno_neutro": "Perdite nelle tue unita' dall'ultima tornata: nessuna.\n",
    "amm_morti_neutro": (
        "Perdite nelle tue unita' dall'ultima tornata: {totale}{quota} \u2014 {dettaglio}.\n"
    ),
    "amm_morti_quota_neutro": " ({quota} dei tuoi occupanti)",
    "amm_unita": "unita'",
    "colono": "colono",
    "coloni": "coloni",
    "colonia": "colonia",
    "colono_cieco": "occupante",
    "coloni_cieco": "occupanti",
    "colonia_cieco": "popolazione totale",

    # --- Testi che l'amministratore legge come parte del quadro, non del ruolo.
    # Stavano dentro administration.py come f-string: in inglese uscivano in
    # italiano, e al gradino cieco nominavano i coloni.
    "amm_cella": "cella",
    "vista_pura": "{parola} {cella}: nessuna regola, preferenze pure",
    "vista_nulla": ("{parola} {cella}: NESSUN INTERVENTO — nessuna regola scatta, "
                    "i coloni seguono le proprie preferenze"),
    "vista_pesi_uno": ("{parola} {cella}: NESSUN INTERVENTO — scatta «{regola}» "
                       "ma tutti i pesi valgono 1, quindi non cambia niente"),
    "vista_regola": "{parola} {cella}: {regola}  ->  {pesi}",
    "catena_vuota": "  (nessuna politica: il governo non ha emanato regole)",
    "prec_accettato": ("La volta scorsa hai ACCETTATO la politica del governo. "
                       "Motivo che avevi dato: {motivo}"),
    "prec_riscritto": "La volta scorsa avevi RISCRITTO cosi': {regole}",
    "prec_a_vuoto": (" — ma su nessuna delle tue celle scattava con pesi diversi da 1: "
                     "le hai lasciate SENZA POLITICA, ne' quella del governo ne' la tua."),
    "amm_cella_neutro": "unita'",
    "vista_pura_neutro": "{parola} {cella}: nessuna regola, preferenze pure",
    "vista_nulla_neutro": ("{parola} {cella}: NESSUN INTERVENTO — nessuna regola "
                           "scatta, gli occupanti seguono le proprie preferenze"),
    "vista_pesi_uno_neutro": ("{parola} {cella}: NESSUN INTERVENTO — scatta «{regola}» "
                              "ma tutti i pesi valgono 1, quindi non cambia niente"),
    "vista_regola_neutro": "{parola} {cella}: {regola}  ->  {pesi}",
    "catena_vuota_neutro": "  (nessuna regola: il livello centrale non ne ha emanate)",
    "prec_accettato_neutro": ("La volta scorsa hai ACCETTATO le regole del livello "
                              "centrale. Motivo che avevi dato: {motivo}"),
    "prec_riscritto_neutro": "La volta scorsa avevi RISCRITTO cosi': {regole}",
    "prec_a_vuoto_neutro": (" — ma su nessuna delle tue unita' scattava con pesi "
                            "diversi da 1: le hai lasciate SENZA REGOLE, ne' quelle del "
                            "livello centrale ne' le tue."),
}

# ---------------------------------------------------------------------- inglese

_EN = {
    "ruolo": (
        "You are the governor of the Martian colony. You are not on the "
        "surface: you observe the aggregates and you write the POLICY by which "
        "every colonist will then decide on their own.\n\n"
        "How a colonist decides: at each step they sample {uno_fra_sei} "
        "with scores = individual preferences x personal urgencies x "
        "cell priority x skill. Your policy multiplies {i_plurale}"
        ", cell by cell, wherever its conditions fire. Where "
        "no rule fires, the colonist chooses on their own "
        "preferences alone. A weight cannot make possible an action that the "
        "cell forbids.\n"
    ),
    "ruolo_neutro": (
        "You observe a system through nine indicators measured on each "
        "of its units, and you write the rules by which the units "
        "will divide their own work among five activities. The rules "
        "are evaluated on every unit at every step, without asking you "
        "anything further.\n"
    ),
    "aiuti": (
        "\nThe reference policy of the non-LLM arm is:\n"
        "{regole}.\n"
        "You are not bound to it: you may write any policy in the "
        "vocabulary below - different thresholds, weights and {plurale_diversi} - and you must "
        "explain why.\n"
    ),
    "dati": (
        "\nStep: {step}. Population: {popolazione}. "
        "Occupied units: {celle}.\n"
        "Aggregates:\n{metriche}\n\n"
        "Distribution of the indicators over the occupied units (mean/std/min/max):\n"
        "{indicatori}\n\n"
        "The same indicators WEIGHTED per {colono} (each unit weighs as many "
        "{coloni} as it hosts; weighted mean/std, min/max per unit):\n"
        "{indicatori_pesati}\n\n"
    ),
    "code_dati": (
        "Population averages:\n{medie}\n"
        "Structure totals:\n{strutture}\n\n"
    ),
    "dove_stanno": (
        "Where the {coloni} are: the most populated unit, {cella}, hosts "
        "{occupanti} {coloni} ({quota} of the {colonia}). Its "
        "indicators: {valori}. A rule that does not "
        "fire there does not touch {quota} of the {coloni}.\n\n"
    ),
    "morti_nessuno": "Deaths since the last tick: none.\n\n",
    "morti_intestazione": "Deaths since the last tick: {totale}, by unit and cause:\n",
    "morti_riga": "  unit {cella}: {dettaglio}",
    "avvertenze_intestazione": "Warnings about the current scenario:\n",
    "avvertenza_zero": (
        "- {nome} is ZERO on every occupied cell: a condition "
        "`<` with a positive threshold fires everywhere, a `>` never. Such a rule, "
        "written first, SHADOWS all the rules that follow: "
        "none of them will ever fire again."
    ),
    "avvertenza_piatta": (
        "- {nome} equals {valore} on ALL occupied cells: a "
        "threshold does not tell one cell from another, it switches "
        "the rule on or off for all of them together."
    ),
    "avvertenza_una_cella": (
        "- there is ONE occupied cell only: every condition is "
        "all-or-nothing until the colony expands."
    ),
    "consuntivo_intestazione": (
        "What YOUR previous policy did, since the last update "
        "(for each cell the first rule that fires is the one that counts, so a broad rule "
        "written first takes the cells away from all the later ones):\n"
    ),
    "consuntivo_altrimenti": "otherwise (no rule)",
    "consuntivo_regola": "rule {indice}",
    "consuntivo_regola_con_testo": "rule {indice}: {testo}",
    "consuntivo_riga": "- {etichetta}: {quante} cell-steps{coda}",
    "consuntivo_ombra": "  <-- SHADOWED: it never fired",
    "vocabolario": (
        "Policy vocabulary.\n"
        "Indicators (conditions are evaluated on EVERY unit, at every step):\n"
        "{indicatori}\n"
        "Operators: < and >.\n"
        "{pesabili}: {categorie}.\n"
    ),
    "glossario": (
        "{glossario_apertura}\n"
        "- sustenance: drinking, eating, collecting ice, farming and "
        "harvesting food. {glossario_fame}\n"
        "- resources: collecting minerals from the deposit and construction_material "
        "from the cell's warehouse. WARNING: collecting construction_material "
        "does not PRODUCE any — it moves it from the warehouse into the colonists' "
        "pockets, and construction already draws directly from the warehouse. New "
        "material comes from the deposits (and from ISRU, if active); minerals from "
        "the seam. Weighting `resources` in order to 'have more material' is "
        "therefore wasted work: it helps only where the colonists must carry it "
        "with them (moves, foundations).\n"
        "- build: erecting structures (greenhouses, habitats, panels, labs...) and "
        "maintaining them. A structure whose per-inhabitant ratio is already "
        "satisfied is not built anyway.\n"
        "- life: rest, care, recovery. It produces nothing by itself.\n"
        "- explore: moving and exploring, that is, founding new settlements.\n"
    ),
    "esempio_da_proteggere": " (for example explore, without which nobody founds new cells)",
    "chiusura": (
        "A weight is a multiplier between {peso_min} and "
        "{peso_max} (1.0 = no change; below 1 discourages, "
        "above 1 encourages).\n"
        "Weights are RELATIVE {fra_le_plurale}: the scores are renormalised, "
        "so raising {tre_plurale} to x2-x3 while leaving the others at 1 amounts to "
        "PENALISING the latter, which lose share in the sampling. {una_da_proteggere} "
        "you want to protect{esempio} {va_alzata}, "
        "not left at 1.\n"
        "At most {max_regole} rules. For each unit the FIRST rule that fires is "
        "the one that counts, in the order in which you write them; the others are ignored. "
        "A rule with \"if\": null always fires and closes the chain.\n\n"
        "Reply ONLY with this JSON:\n"
        '{{"policy": [{{"if": {{"indicator": "<name>", "op": "<|>", '
        '"value": <float>}} | null, "weights": {{"<{segnaposto}>": <float>}}}}], '
        '"rationale": "<why>"}}\n'
    ),
    "amm_ruolo": (
        "You are the administrator of district {distretto} of a Martian "
        "colony. The central government has just issued the policy by which the "
        "colonists choose what to do. You administer "
        "{celle} cells and not the others.\n\n"
    ),
    "amm_catena": (
        "The COMPLETE policy of the government, in the order in which it holds (for each "
        "cell the FIRST rule that fires is the one that counts):\n{catena}\n\n"
        "Here is what this policy produces, cell by cell, in "
        "YOUR district:\n"
    ),
    "amm_scelte": (
        "You have two choices, and neither of them is the fallback:\n"
        "  ACCEPT if the government's policy is already adequate for your cells: "
        "they will apply it exactly as it is.\n"
        "  REWRITE if it is not: your chain takes the place of the government's, "
        "for your cells and those only. You may start from the government's "
        "rules and change their thresholds, weights or order, or "
        "write new ones: what you hand over is the ENTIRE law of your "
        "cells, not an amendment.\n"
        "Accepting a policy that works is as correct as correcting one "
        "that does not. Accepting out of inertia a policy that does not act "
        "on your cells is not: if above you read NO INTERVENTION, the government is "
        "leaving you without a policy, and the decision is entirely yours.\n\n"
        "Step: {step}. Colonists in your district: {popolazione}.\n"
    ),
    "amm_dati": (
        "\nMetrics of the whole colony (the national picture):\n{metriche}\n\n"
        "Distribution of the indicators over YOUR cells (mean/std/min/max):\n"
        "{indicatori}\n\n"
        "The same indicators over the WHOLE colony, for comparison:\n"
        "{indicatori_colonia}\n\n"
        "State of the colonists of your district (mean and dangerous tail: min for "
        "health/hydration/satiety/morale, max for fatigue/stress):\n"
        "{stato}\n\n"
    ),
    "amm_vocabolario": (
        "Vocabulary (identical to the government's).\n"
        "Cell indicators:\n{indicatori}\n"
        "Operators: < and >.\n"
        "{pesabili}: {categorie}.\n"
        "A weight is a multiplier between {peso_min} and "
        "{peso_max} (1.0 = no change).\n"
        "Weights are RELATIVE: the scores are renormalised, so "
        "raising three of them while leaving the others at 1 amounts to penalising the "
        "latter. {una_da_proteggere} you want to protect {va_alzata}.\n"
        "At most {max_regole} rules, and for each cell the FIRST one that "
        "fires is the one that counts.\n\n"
        "Reply ONLY with one of these two JSON objects:\n"
        '{{"accept": true, "rationale": "<why the government policy is fine>"}}\n'
        '{{"accept": false, "policy": [{{"if": {{"indicator": "<name>", "op": "<|>", '
        '"value": <float>}} | null, "weights": {{"<{segnaposto}>": <float>}}}}], '
        '"rationale": "<why I am changing it>"}}\n'
    ),
    "amm_morti_nessuno": "Deaths in your district since the last round: none.\n",
    "amm_morti": (
        "Deaths in your district since the last round: {totale}{quota} \u2014 {dettaglio}.\n"
    ),
    "amm_morti_quota": " ({quota} of your colonists)",
    "amm_ruolo_neutro": (
        "You administer {celle} units of a system, and not the others. A central "
        "level has just issued the rules by which each unit divides its own work "
        "among five activities.\n\n"
    ),
    "amm_catena_neutro": (
        "The COMPLETE rules of the central level, in the order in which they hold "
        "(for each unit the FIRST one that fires is the one that counts):\n{catena}\n\n"
        "Here is what these rules produce, unit by unit, among YOURS:\n"
    ),
    "amm_scelte_neutro": (
        "You have two choices, and neither of them is the fallback:\n"
        "  ACCEPT if the central rules are already right for your units: "
        "they will be applied exactly as they are.\n"
        "  REWRITE if they are not: your chain takes the place of the central "
        "one, for your units and those only. You may start from the central "
        "rules and change their thresholds, weights or order, or write new "
        "ones: what you hand over is the ENTIRE chain of your units, not an "
        "amendment.\n"
        "Accepting rules that work is as correct as correcting wrong ones. "
        "Accepting out of inertia rules that do not act on your units is not: if "
        "above you read NO INTERVENTION, the central level is leaving you "
        "without rules, and the decision is entirely yours.\n\n"
        "Step: {step}. Occupants in your units: {popolazione}.\n"
    ),
    "amm_dati_neutro": (
        "\nMeasures of the whole system:\n{metriche}\n\n"
        "Distribution of the indicators over YOUR units (mean/std/min/max):\n"
        "{indicatori}\n\n"
        "The same indicators over the WHOLE system, for comparison:\n"
        "{indicatori_colonia}\n\n"
        "State of the occupants of your units (mean and dangerous tail):\n"
        "{stato}\n\n"
    ),
    "amm_vocabolario_neutro": (
        "Vocabulary (identical to the central level's).\n"
        "Unit indicators:\n{indicatori}\n"
        "Operators: < and >.\n"
        "{pesabili}: {categorie}.\n"
        "A weight is a multiplier between {peso_min} and "
        "{peso_max} (1.0 = no change).\n"
        "Weights are RELATIVE: the scores are renormalised, so "
        "raising three of them while leaving the others at 1 amounts to "
        "penalising the latter. {una_da_proteggere} you want to protect "
        "{va_alzata}.\n"
        "At most {max_regole} rules, and for each unit the FIRST one that "
        "fires is the one that counts.\n\n"
        "Reply ONLY with one of these two JSON objects:\n"
        '{{"accept": true, "rationale": "<why the central rules are fine>"}}\n'
        '{{"accept": false, "policy": [{{"if": {{"indicator": "<name>", "op": "<|>", '
        '"value": <float>}} | null, "weights": {{"<{segnaposto}>": <float>}}}}], '
        '"rationale": "<why I am changing them>"}}\n'
    ),
    "amm_morti_nessuno_neutro": "Losses in your units since the last round: none.\n",
    "amm_morti_neutro": (
        "Losses in your units since the last round: {totale}{quota} \u2014 {dettaglio}.\n"
    ),
    "amm_morti_quota_neutro": " ({quota} of your occupants)",
    "amm_unita": "unit",
    "colono": "colonist",
    "coloni": "colonists",
    "colonia": "colony",
    "colono_cieco": "occupant",
    "coloni_cieco": "occupants",
    "colonia_cieco": "total population",
    # --- Same three texts in English. A translation, not a rewrite: where the
    # Italian is blunt the English stays blunt, and the marker is the one the
    # instructions above tell the administrator to look for.
    "amm_cella": "cell",
    "vista_pura": "{parola} {cella}: no rule, pure preferences",
    "vista_nulla": ("{parola} {cella}: NO INTERVENTION — no rule fires, "
                    "the colonists follow their own preferences"),
    "vista_pesi_uno": ("{parola} {cella}: NO INTERVENTION — «{regola}» fires "
                       "but every weight is 1, so nothing changes"),
    "vista_regola": "{parola} {cella}: {regola}  ->  {pesi}",
    "catena_vuota": "  (no policy: the government has issued no rules)",
    "prec_accettato": ("Last time you ACCEPTED the government's policy. "
                       "The reason you gave: {motivo}"),
    "prec_riscritto": "Last time you had REWRITTEN it as: {regole}",
    "prec_a_vuoto": (" — but on none of your cells did it fire with weights other "
                     "than 1: you left them WITH NO POLICY, neither the government's "
                     "nor your own."),
    "amm_cella_neutro": "unit",
    "vista_pura_neutro": "{parola} {cella}: no rule, pure preferences",
    "vista_nulla_neutro": ("{parola} {cella}: NO INTERVENTION — no rule fires, "
                           "the occupants follow their own preferences"),
    "vista_pesi_uno_neutro": ("{parola} {cella}: NO INTERVENTION — «{regola}» fires "
                              "but every weight is 1, so nothing changes"),
    "vista_regola_neutro": "{parola} {cella}: {regola}  ->  {pesi}",
    "catena_vuota_neutro": "  (no rules: the central level has issued none)",
    "prec_accettato_neutro": ("Last time you ACCEPTED the central level's rules. "
                              "The reason you gave: {motivo}"),
    "prec_riscritto_neutro": "Last time you had REWRITTEN it as: {regole}",
    "prec_a_vuoto_neutro": (" — but on none of your units did it fire with weights "
                            "other than 1: you left them WITH NO RULES, neither the "
                            "central level's nor your own."),
}

FRASI: dict[str, dict[str, str]] = {"it": _IT, "en": _EN}

# Le due lingue devono avere le STESSE chiavi: una chiave mancante in una sola
# lingua produrrebbe un blocco vuoto in un braccio e non nell'altro, cioe' due
# prompt che differiscono per contenuto e non per lingua --- il difetto che
# questa campagna non potrebbe vedere.
assert set(_IT) == set(_EN), (
    "le due lingue divergono: "
    f"solo it {sorted(set(_IT) - set(_EN))}, solo en {sorted(set(_EN) - set(_IT))}"
)


def frasi(lingua: str) -> dict[str, str]:
    try:
        return FRASI[lingua]
    except KeyError:
        raise NotImplementedError(f"il prompt in «{lingua}» non esiste") from None
