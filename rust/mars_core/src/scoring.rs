//! Punteggio dei sei pilastri per un agente (Task C.2 del port del ciclo).
//!
//! Cinque funzioni Python -- `pillar_availability`, `pillar_priority`,
//! `compute_urgencies`, `score_pillars`, `choose_pillar` -- valgono insieme il
//! 10,42% del passo su configurazione reale, e vengono eseguite 11.941 volte su
//! 12.000 decisioni. Qui stanno come un solo calcolo, cosi' che il confine venga
//! attraversato una volta per decisione invece di cinque.
//!
//! **Il vincolo di parita' che non e' ovvio.** L'originale eleva al quadrato con
//! `x ** 2` su un `float` Python, cioe' attraverso la `pow()` di sistema.
//! Misurato: `x ** 2 != x * x` per 556 valori su un milione nel dominio vero.
//! Questo modulo usa quindi `powf(2.0)` e non una moltiplicazione, ed esiste un
//! test che confronta i due lati proprio sui valori che divergono.
//!
//! Le tabelle costanti (indici di azione per pilastro, indici di risorsa, codici
//! di missione) arrivano dal chiamante e attraversano il confine **una volta**,
//! alla costruzione. Non sono duplicate qui: la loro definizione vive lato
//! Python, e una copia locale sarebbe una divergenza silenziosa in attesa.

/// Numero di pilastri. Coincide con `pillars.N_PILLARS` e con l'ordine
/// posizionale `P_SUSTENANCE..P_EXPLORE`, su cui poggiano le formule sotto.
pub const N_PILLARS: usize = 6;

/// Tabelle costanti del punteggio, costruite una volta sola.
pub struct PillarTables {
    /// Indici di azione per pilastro, in ordine di pilastro.
    pub pillar_actions: Vec<Vec<usize>>,
    /// Indici delle azioni di costruzione.
    pub build_indices: Vec<usize>,
    /// Indice di `construction_material` in `inv`.
    pub material_index: usize,
    /// Indice di `minerals` in `inv`.
    pub minerals_index: usize,
    /// Codici della colonna `mission` che contano come missione attiva.
    pub mission_active_codes: Vec<i8>,
    /// Esponente delle urgenze, cioe' il `2` di `x ** 2` lato Python.
    ///
    /// E' un campo e non una costante letterale **di proposito**. `powf(2.0)`
    /// sembra la traduzione ovvia, ma con l'esponente costante LLVM la riscrive in
    /// `x * x` (passo `SimplifyLibCalls`), e le due non coincidono: misurato,
    /// differiscono all'ultimo bit per 556 valori su un milione nel dominio vero.
    /// Ricevendo l'esponente dal chiamante la riscrittura non e' possibile, e in
    /// piu' il numero resta definito una volta sola, lato Python.
    pub urgency_exponent: f64,
}

/// Lo stato di un agente che il punteggio legge.
pub struct AgentRow<'a> {
    /// Maschera di ammissibilita' per azione.
    pub mask: &'a [bool],
    /// Priorita' proposta dalla cella per azione.
    pub action_priority: &'a [f64],
    /// Preferenze normalizzate sui pilastri.
    pub preferences: &'a [f64],
    /// Abilita' di ruolo sui pilastri.
    pub skills: &'a [f64],
    /// Idratazione.
    pub hydration: f64,
    /// Sazieta'.
    pub satiety: f64,
    /// Salute.
    pub health: f64,
    /// Affaticamento.
    pub fatigue: f64,
    /// Indice di stress.
    pub stress: f64,
    /// Morale.
    pub morale: f64,
    /// Curiosita'.
    pub curiosity: f64,
    /// Materiale da costruzione trasportato.
    pub material: f64,
    /// Minerali trasportati.
    pub minerals: f64,
    /// Passi senz'acqua.
    pub steps_without_water: i32,
    /// Passi senza cibo.
    pub steps_without_food: i32,
    /// Codice di missione.
    pub mission: i8,
    /// Kit del fondatore riservato (stato cold, non una colonna).
    pub founder_kit_reserved: bool,
}

// Nota di metodo, perche' l'errore e' stato commesso qui. La prima stesura usava
// `powf(2.0)` e passava l'harness di parita' su 120 passi con digest identico. La
// divergenza c'era comunque: l'ha trovata `tests/core/test_decision_scoring.py`,
// che confronta i bracci proprio sui valori dove `pow` e la moltiplicazione si
// separano. Un digest di stato identico non dimostra che un valore INTERMEDIO
// coincide -- una differenza all'ultimo bit nei punteggi cambia lo stato solo
// quando arriva a ribaltare il pilastro scelto, cioe' quasi mai e non mai.

/// `min(1.0, value)` con la semantica di Python: `b if b < a else a`.
#[inline]
fn capped(value: f64) -> f64 {
    if value < 1.0 { value } else { 1.0 }
}

/// `max(first, second)` con la semantica di Python: `b if b > a else a`.
#[inline]
fn larger(first: f64, second: f64) -> f64 {
    if second > first { second } else { first }
}

/// Le sei urgenze, nell'ordine e con l'ordine di operazioni dell'originale.
#[must_use]
pub fn urgencies(row: &AgentRow<'_>, tables: &PillarTables) -> [f64; N_PILLARS] {
    let deprived = row.steps_without_water >= 2 || row.steps_without_food >= 6;
    let sustenance = capped(
        larger(1.0 - row.hydration, 1.0 - row.satiety).powf(tables.urgency_exponent)
            * 1.5
            + if deprived { 0.3 } else { 0.0 },
    );

    let resources = 1.0 - 0.5 * (row.material / 10.0 + row.minerals / 6.0);
    // Scritto per esteso invece che con `clamp`: la forma esplicita rende visibile
    // che su NaN entrambe le guardie sono false e il valore passa, che e' cio' che
    // fa `np.clip` lato Python. Con `clamp` quel comportamento andrebbe verificato
    // sulla documentazione invece che letto qui.
    #[expect(
        clippy::manual_clamp,
        reason = "la forma esplicita documenta la semantica di np.clip su NaN"
    )]
    let resources = if resources < 0.0 {
        0.0
    } else if resources > 1.0 {
        1.0
    } else {
        resources
    };

    let buildable = tables
        .build_indices
        .iter()
        .any(|&index| row.mask[index]);
    let build = 0.6 * f64::from(u8::from(buildable)) + 0.4 * capped(row.material / 10.0);

    let life = capped(
        larger(1.0 - row.health, row.fatigue).powf(tables.urgency_exponent) * 1.3
            + 0.2 * row.stress,
    );
    let social = capped(0.8 * row.stress + larger(0.0, 0.82 - row.morale));

    let mission_active = row.founder_kit_reserved
        || tables.mission_active_codes.contains(&row.mission);
    let explore = capped(
        0.25 + 0.6 * row.curiosity + if mission_active { 0.5 } else { 0.0 },
    );

    [sustenance, resources, build, life, social, explore]
}

/// Punteggi normalizzati dei sei pilastri e pilastro scelto.
///
/// `survival` spento corrisponde all'esperimento guidato dalla cella: le urgenze
/// valgono tutte 1 e lo stato vitale personale non pesa.
///
/// # Panics
///
/// Se `tables.pillar_actions` non ha esattamente [`N_PILLARS`] voci, o se un
/// indice cade fuori dagli array forniti. Il ponte `PyO3` valida a monte, cosi'
/// che l'errore dica *quale* argomento e' sbagliato.
#[must_use]
pub fn score(
    row: &AgentRow<'_>,
    tables: &PillarTables,
    survival: bool,
    u01: f64,
    greedy: bool,
) -> ([f64; N_PILLARS], usize) {
    assert_eq!(
        tables.pillar_actions.len(),
        N_PILLARS,
        "`pillar_actions` deve avere una voce per pilastro"
    );
    let urgency = if survival {
        urgencies(row, tables)
    } else {
        [1.0; N_PILLARS]
    };

    let mut raw = [0.0_f64; N_PILLARS];
    let mut total = 0.0_f64;
    for (pillar, actions) in tables.pillar_actions.iter().enumerate() {
        let mut best = 0.0_f64;
        let mut available = false;
        for &action in actions {
            if row.mask[action] {
                let priority = row.action_priority[action];
                // Lo zero e' il valore per lista VUOTA di `max(..., default=0.0)`,
                // non un minimo: una priorita' negativa ammissibile deve vincere
                // sullo zero, altrimenti il pilastro sembrerebbe piu' attraente
                // di quanto la cella dica.
                if !available || priority > best {
                    best = priority;
                }
                available = true;
            }
        }
        if available {
            let value = row.preferences[pillar] * urgency[pillar] * best * row.skills[pillar];
            raw[pillar] = value;
            total += value;
        }
    }
    if total <= 0.0 {
        return ([0.0; N_PILLARS], 0);
    }

    let mut scores = [0.0_f64; N_PILLARS];
    for (slot, value) in scores.iter_mut().zip(raw) {
        *slot = value / total;
    }

    if greedy {
        let mut chosen = 0;
        let mut best = scores[0];
        for (pillar, &value) in scores.iter().enumerate().skip(1) {
            if value > best {
                best = value;
                chosen = pillar;
            }
        }
        return (scores, chosen);
    }

    let mut cumulative = 0.0_f64;
    let mut running = [0.0_f64; N_PILLARS];
    for (slot, value) in running.iter_mut().zip(scores) {
        cumulative += value;
        *slot = cumulative;
    }
    if cumulative <= 0.0 {
        return (scores, 0);
    }
    // `np.searchsorted(..., side="right")`: quanti elementi sono <= soglia.
    let threshold = u01 * cumulative;
    let mut chosen = 0;
    for &value in &running {
        if value <= threshold {
            chosen += 1;
        } else {
            break;
        }
    }
    (scores, chosen.min(N_PILLARS - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> PillarTables {
        PillarTables {
            // Due azioni per pilastro, indici 0..11.
            pillar_actions: (0..N_PILLARS).map(|p| vec![2 * p, 2 * p + 1]).collect(),
            build_indices: vec![4, 5],
            material_index: 0,
            minerals_index: 1,
            mission_active_codes: vec![1, 2, 3],
            urgency_exponent: 2.0,
        }
    }

    fn row<'a>(mask: &'a [bool], priority: &'a [f64], pref: &'a [f64], skill: &'a [f64]) -> AgentRow<'a> {
        AgentRow {
            mask,
            action_priority: priority,
            preferences: pref,
            skills: skill,
            hydration: 1.0,
            satiety: 1.0,
            health: 1.0,
            fatigue: 0.0,
            stress: 0.0,
            morale: 1.0,
            curiosity: 0.0,
            material: 0.0,
            minerals: 0.0,
            steps_without_water: 0,
            steps_without_food: 0,
            mission: 0,
            founder_kit_reserved: false,
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "il confronto esatto E' l'oggetto del test: la parita' bit-exact"
    )]
    fn an_unavailable_pillar_scores_zero_and_the_rest_normalize_to_one() {
        let mask = [true, true, false, false, true, true, true, true, true, true, true, true];
        let priority = [1.0; 12];
        let pref = [1.0 / 6.0; 6];
        let skill = [1.0; 6];
        let (scores, _chosen) = score(&row(&mask, &priority, &pref, &skill), &tables(), true, 0.0, false);
        assert_eq!(scores[1], 0.0, "pilastro senza azioni ammissibili");
        let total: f64 = scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "somma {total}");
    }

    #[test]
    fn a_negative_admissible_priority_beats_the_empty_default() {
        // Se lo zero fosse trattato come minimo invece che come valore di lista
        // vuota, questo pilastro riceverebbe 0.0 e sembrerebbe indisponibile
        // quanto uno senza azioni: due situazioni diverse con lo stesso numero.
        let mut mask = [false; 12];
        mask[0] = true;
        mask[2] = true;
        let mut priority = [0.0; 12];
        priority[0] = -2.0;
        priority[2] = 4.0;
        let pref = [1.0; 6];
        let skill = [1.0; 6];
        let mut agent = row(&mask, &priority, &pref, &skill);
        // Con vitali perfetti l'urgenza del sostentamento e' zero e il prodotto
        // darebbe -0.0 qualunque sia la priorita': il segno non direbbe nulla.
        agent.hydration = 0.5;
        let (scores, _) = score(&agent, &tables(), true, 0.0, false);
        assert!(scores[0] < 0.0, "priorita' negativa conservata: {}", scores[0]);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "il confronto esatto E' l'oggetto del test: la parita' bit-exact"
    )]
    fn the_exponent_comes_from_the_tables_and_is_not_hardcoded() {
        // Guardiano locale della scelta spiegata sopra: se qualcuno sostituisse
        // `powf(tables.urgency_exponent)` con `x * x` o con `powf(2.0)` -- che
        // sembrano la stessa cosa e sono anche piu' veloci -- questo test cade.
        // La parita' FRA I LINGUAGGI la verifica invece
        // `tests/core/test_decision_scoring.py`, che e' l'unico posto da cui si
        // vedono entrambi i lati.
        let mask = [true; 12];
        let priority = [1.0; 12];
        let pref = [1.0; 6];
        let skill = [1.0; 6];
        let mut agent = row(&mask, &priority, &pref, &skill);
        agent.hydration = 0.5;
        agent.satiety = 1.0;
        let mut cubic = tables();
        cubic.urgency_exponent = 3.0;
        assert_eq!(urgencies(&agent, &cubic)[0], capped(0.5_f64.powf(3.0) * 1.5));
        assert_eq!(
            urgencies(&agent, &tables())[0],
            capped(0.5_f64.powf(2.0) * 1.5)
        );
    }

    #[test]
    fn survival_disabled_flattens_every_urgency_to_one() {
        let mask = [true; 12];
        let priority = [1.0; 12];
        let pref = [1.0 / 6.0; 6];
        let skill = [1.0; 6];
        let mut agent = row(&mask, &priority, &pref, &skill);
        agent.hydration = 0.0;
        let (scores, _) = score(&agent, &tables(), false, 0.0, false);
        for value in scores {
            assert!((value - 1.0 / 6.0).abs() < 1e-15, "atteso uniforme, visto {value}");
        }
    }

    #[test]
    fn the_softmax_draw_counts_the_prefixes_below_the_threshold() {
        let mask = [true; 12];
        let priority = [1.0; 12];
        let pref = [1.0 / 6.0; 6];
        let skill = [1.0; 6];
        let agent = row(&mask, &priority, &pref, &skill);
        // Punteggi uniformi: la soglia u01 cade nel pilastro floor(u01 * 6).
        for (u01, expected) in [(0.0, 0), (0.2, 1), (0.5, 3), (0.99, 5), (1.0, 5)] {
            let (_, chosen) = score(&agent, &tables(), false, u01, false);
            assert_eq!(chosen, expected, "u01={u01}");
        }
    }
}
