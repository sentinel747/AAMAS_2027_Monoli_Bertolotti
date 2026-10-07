//! Filtro batch del guardiano fisiologico (Task C.1 del port del ciclo).
//!
//! `_physiological_override`, lato Python, restituisce `None` nel 99,98% delle
//! chiamate. Questo modulo dimostra in blocco *chi* ricade in quel 99,98%, cosi'
//! che il ciclo sequenziale possa saltare la chiamata invece di eseguirla per poi
//! scoprire che non c'era nulla da fare.
//!
//! Il predicato non e' un'euristica ma una condizione **sufficiente**: tutte le
//! guardie della funzione originale devono essere false. La derivazione, in
//! particolare per l'unico ramo privo di guardia, sta nella docstring di
//! `src/core/physio_gate.py`, dove vive accanto al codice che quel ramo lo
//! esegue davvero.
//!
//! Indici di colonna, rapporti di supporto e soglie arrivano tutti dal
//! chiamante. Non sono duplicati qui di proposito: sono definiti una volta sola
//! lato Python, e una copia locale sarebbe una divergenza silenziosa in attesa di
//! accadere -- esattamente la classe di errore che lo `SCHEMA_VERSION` esiste per
//! impedire.

/// Soglie e indici di risorsa del predicato.
pub struct GatePolicy {
    /// Indice della colonna `water` in `inv`.
    pub water_index: usize,
    /// Indice della colonna `ice` in `inv`.
    pub ice_index: usize,
    /// Indice della colonna `food` in `inv`.
    pub food_index: usize,
    /// `CRITICAL_HYDRATION`.
    pub critical_hydration: f64,
    /// `CRITICAL_SATIETY`.
    pub critical_satiety: f64,
    /// `CRITICAL_HEALTH`.
    pub critical_health: f64,
    /// `WATER_RETURN_RESERVE`.
    pub water_return_reserve: f64,
    /// `FOOD_RETURN_RESERVE`.
    pub food_return_reserve: f64,
    /// `WATER_GUARD_STEPS`.
    pub water_guard_steps: i32,
    /// `FOOD_GUARD_STEPS`.
    pub food_guard_steps: i32,
}

/// Indici di struttura e rapporti abitanti-per-struttura.
pub struct SupportPolicy {
    /// Indice di `SHELTER` in `struct_count`.
    pub shelter: usize,
    /// Indice di `HABITAT`.
    pub habitat: usize,
    /// Indice di `INFIRMARY`.
    pub infirmary: usize,
    /// Indice di `GREENHOUSE`.
    pub greenhouse: usize,
    /// Indice di `SOLAR_ARRAY`.
    pub solar_array: usize,
    /// Indice di `OXYGEN_PLANT`.
    pub oxygen_plant: usize,
    /// Colonisti serviti da una serra.
    pub greenhouse_ratio: i64,
    /// Colonisti serviti da un pannello solare.
    pub solar_ratio: i64,
    /// Colonisti serviti da un impianto a ossigeno.
    pub oxygen_ratio: i64,
}

/// Colonne per-agente, indicizzate dalla riga di `AgentArrays` (non dal batch).
pub struct AgentColumns<'a> {
    /// Righe che decidono in questo passo.
    pub rows: &'a [i64],
    /// Colonna `x` completa.
    pub x: &'a [i16],
    /// Colonna `y` completa.
    pub y: &'a [i16],
    /// Colonna `hydration` completa.
    pub hydration: &'a [f64],
    /// Colonna `satiety` completa.
    pub satiety: &'a [f64],
    /// Colonna `health` completa.
    pub health: &'a [f64],
    /// Colonna `steps_without_water` completa.
    pub steps_without_water: &'a [i16],
    /// Colonna `steps_without_food` completa.
    pub steps_without_food: &'a [i16],
    /// Inventario `[capacita', risorse]` in ordine di riga.
    pub inv: &'a [f64],
    /// Numero di risorse per riga di `inv`.
    pub resource_count: usize,
}

/// Griglia dei conteggi di struttura, `[altezza, larghezza, tipi]`.
pub struct StructureGrid<'a> {
    /// Conteggi in ordine di riga.
    pub counts: &'a [i16],
    /// Larghezza della griglia.
    pub width: usize,
    /// Numero di tipi di struttura per cella.
    pub structure_count: usize,
}

/// `local_life_support_capacity` calcolata sui soli conteggi, in interi.
///
/// La formula Python usa `int(shelter + 2*habitat + 0.5*infirmary)`. Per
/// conteggi non negativi quel troncamento coincide con
/// `(2*shelter + 4*habitat + infirmary) / 2` fra interi, e restare negli interi
/// e' cio' che rende i tre bracci identici per costruzione.
#[must_use]
fn life_support_capacity(counts: &[i16], support: &SupportPolicy) -> i64 {
    let count_of = |index: usize| i64::from(counts[index]);
    let housing = (2 * count_of(support.shelter)
        + 4 * count_of(support.habitat)
        + count_of(support.infirmary))
        / 2;
    let capacity = housing
        .min(count_of(support.greenhouse) * support.greenhouse_ratio)
        .min(count_of(support.solar_array) * support.solar_ratio)
        .min(count_of(support.oxygen_plant) * support.oxygen_ratio);
    capacity.max(0)
}

/// Scrive in `out[i]` se la riga `agents.rows[i]` puo' saltare l'override.
///
/// # Panics
///
/// Se `out` non ha la stessa lunghezza di `agents.rows`, o se una riga o una
/// coordinata cade fuori dalle colonne fornite. Il ponte `PyO3` valida gli
/// intervalli prima di arrivare qui, cosi' che l'errore visto da chi chiama dica
/// *quale* array e' sbagliato invece di essere un panic.
pub fn skip_override(
    agents: &AgentColumns<'_>,
    grid: &StructureGrid<'_>,
    support: &SupportPolicy,
    policy: &GatePolicy,
    out: &mut [bool],
) {
    assert_eq!(
        out.len(),
        agents.rows.len(),
        "`out` deve avere un elemento per riga del batch"
    );
    for (slot, &row) in out.iter_mut().zip(agents.rows) {
        let index = usize::try_from(row).expect("riga negativa: validare a monte");
        let y = usize::try_from(agents.y[index]).expect("y negativa: validare a monte");
        let x = usize::try_from(agents.x[index]).expect("x negativa: validare a monte");
        let cell = (y * grid.width + x) * grid.structure_count;
        let counts = &grid.counts[cell..cell + grid.structure_count];
        let inventory = index * agents.resource_count;
        let water = agents.inv[inventory + policy.water_index];
        let ice = agents.inv[inventory + policy.ice_index];
        let food = agents.inv[inventory + policy.food_index];
        *slot = life_support_capacity(counts, support) > 0
            && i32::from(agents.steps_without_food[index]) < policy.food_guard_steps
            && i32::from(agents.steps_without_water[index]) < policy.water_guard_steps
            && agents.hydration[index] >= policy.critical_hydration
            && agents.satiety[index] >= policy.critical_satiety
            && agents.health[index] >= policy.critical_health
            && water + ice > policy.water_return_reserve
            && food > policy.food_return_reserve;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHELTER: usize = 0;
    const HABITAT: usize = 1;
    const INFIRMARY: usize = 2;
    const GREENHOUSE: usize = 3;
    const SOLAR: usize = 4;
    const OXYGEN: usize = 5;
    const KINDS: usize = 6;

    fn support() -> SupportPolicy {
        SupportPolicy {
            shelter: SHELTER,
            habitat: HABITAT,
            infirmary: INFIRMARY,
            greenhouse: GREENHOUSE,
            solar_array: SOLAR,
            oxygen_plant: OXYGEN,
            greenhouse_ratio: 7,
            solar_ratio: 7,
            oxygen_ratio: 10,
        }
    }

    fn policy() -> GatePolicy {
        GatePolicy {
            water_index: 0,
            ice_index: 1,
            food_index: 2,
            critical_hydration: 0.75,
            critical_satiety: 0.70,
            critical_health: 0.55,
            water_return_reserve: 3.2,
            food_return_reserve: 1.0,
            water_guard_steps: 2,
            food_guard_steps: 6,
        }
    }

    /// Una cella supportata, un agente in salute: il salto e' lecito.
    fn healthy_case() -> (Vec<i16>, Vec<f64>) {
        let mut counts = vec![0_i16; KINDS];
        counts[HABITAT] = 1;
        counts[GREENHOUSE] = 1;
        counts[SOLAR] = 1;
        counts[OXYGEN] = 1;
        let inventory = vec![5.0, 0.0, 5.0];
        (counts, inventory)
    }

    fn run(counts: &[i16], inventory: &[f64], vitals: (f64, f64, f64), clocks: (i16, i16)) -> bool {
        let agents = AgentColumns {
            rows: &[0],
            x: &[0],
            y: &[0],
            hydration: &[vitals.0],
            satiety: &[vitals.1],
            health: &[vitals.2],
            steps_without_water: &[clocks.0],
            steps_without_food: &[clocks.1],
            inv: inventory,
            resource_count: 3,
        };
        let grid = StructureGrid {
            counts,
            width: 1,
            structure_count: KINDS,
        };
        let mut out = [false];
        skip_override(&agents, &grid, &support(), &policy(), &mut out);
        out[0]
    }

    #[test]
    fn a_supported_healthy_agent_may_skip_the_override() {
        let (counts, inventory) = healthy_case();
        assert!(run(&counts, &inventory, (1.0, 1.0, 1.0), (0, 0)));
    }

    #[test]
    fn every_guard_alone_is_enough_to_deny_the_skip() {
        let (counts, inventory) = healthy_case();
        // Ogni caso spegne una sola guardia: se il filtro saltasse comunque,
        // salterebbe una chiamata che invece puo' restituire un'azione.
        assert!(!run(&counts, &inventory, (0.74, 1.0, 1.0), (0, 0)), "idratazione");
        assert!(!run(&counts, &inventory, (1.0, 0.69, 1.0), (0, 0)), "sazieta'");
        assert!(!run(&counts, &inventory, (1.0, 1.0, 0.54), (0, 0)), "salute");
        assert!(!run(&counts, &inventory, (1.0, 1.0, 1.0), (2, 0)), "orologio acqua");
        assert!(!run(&counts, &inventory, (1.0, 1.0, 1.0), (0, 6)), "orologio cibo");
        assert!(
            !run(&counts, &[3.2, 0.0, 5.0], (1.0, 1.0, 1.0), (0, 0)),
            "riserva d'acqua alla soglia"
        );
        assert!(
            !run(&counts, &[5.0, 0.0, 1.0], (1.0, 1.0, 1.0), (0, 0)),
            "riserva di cibo alla soglia"
        );
    }

    #[test]
    fn an_unsupported_cell_denies_the_skip() {
        let (mut counts, inventory) = healthy_case();
        // Senza serra la capacita' e' zero, e con essa cadono le due distanze di
        // supporto su cui poggia tutta la dimostrazione.
        counts[GREENHOUSE] = 0;
        assert!(!run(&counts, &inventory, (1.0, 1.0, 1.0), (0, 0)));
    }

    #[test]
    fn half_an_infirmary_does_not_round_up_to_housing() {
        // `int(0 + 0 + 0.5*1)` vale 0: una sola infermeria non ospita nessuno.
        // Il troncamento e' parte della formula, non un dettaglio di scrittura.
        let (mut counts, inventory) = healthy_case();
        counts[HABITAT] = 0;
        counts[INFIRMARY] = 1;
        assert!(!run(&counts, &inventory, (1.0, 1.0, 1.0), (0, 0)));
        counts[INFIRMARY] = 2;
        assert!(run(&counts, &inventory, (1.0, 1.0, 1.0), (0, 0)));
    }
}
