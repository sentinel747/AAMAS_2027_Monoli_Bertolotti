//! Raggiungibilita' di una cella con rientro garantito, in forma batch.
//!
//! Riproduce la semantica di `RuleBasedAgent._can_visit_cell_and_return` e delle
//! funzioni che chiama (`_support_distance_cells`, `_return_ration_requirement`)
//! in `src/agents/rule_based_agent.py`, ma calcolata una volta per step su tutta
//! la griglia invece che per ogni coppia (agente, cella candidata).
//!
//! Il profilo post-Task-0b misura 70.032 valutazioni di questo cluster ogni 30
//! step, con la distanza di supporto ricalcolata a ogni query. La riformulazione
//! a campo di distanza la riduce a due passate sulla griglia per step, piu'
//! aritmetica per coppia.
//!
//! Le due funzioni sono deliberatamente separate: il campo di distanza dipende
//! solo dal mondo (dove sono le strutture di supporto), la maschera dipende anche
//! dagli agenti. Tenerle distinte permette di ricalcolare il campo solo quando
//! cambiano le strutture, che e' molto piu' raro di un cambio di inventario.

/// Valore del campo di distanza quando non esiste alcun supporto raggiungibile.
///
/// Corrisponde al ramo `distance is None` del lato Python, che produce
/// `float("inf")` come razione richiesta. Si usa un sentinella intero invece di
/// un `Option<i32>` per cella perche' il campo attraversa il confine FFI come
/// buffer contiguo: `Option` non ha una rappresentazione garantita e
/// costringerebbe a due array paralleli.
pub const NO_SUPPORT: i32 = -1;

/// Distanza Chebyshev su una griglia il cui asse orizzontale e' periodico.
///
/// Corrisponde a `wrapped_chebyshev_distance` (`src/world/navigation.py`): la
/// longitudine si avvolge all'antimeridiano, la latitudine no.
///
/// # Panics
/// Se `width` non e' positivo: un periodo nullo o negativo non definisce alcun
/// avvolgimento, e proseguire produrrebbe silenziosamente distanze assurde.
#[must_use]
pub fn wrapped_chebyshev(
    first_x: i32,
    first_y: i32,
    second_x: i32,
    second_y: i32,
    width: i32,
) -> i32 {
    assert!(width > 0, "il periodo dell'asse x deve essere positivo");
    let direct = (first_x - second_x).abs() % width;
    let horizontal = direct.min(width - direct);
    let vertical = (first_y - second_y).abs();
    horizontal.max(vertical)
}

/// Dimensioni della griglia, con la conversione fatta una volta sola.
///
/// Esiste per evitare di ripetere `i32 <-> usize` a ogni indicizzazione: la
/// conversione e' un punto in cui un valore negativo diventerebbe un indice
/// enorme, quindi va concentrata e verificata in un posto solo.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    width: u32,
    height: u32,
}

impl Grid {
    /// Costruisce la griglia.
    ///
    /// # Panics
    /// Se una delle due dimensioni e' nulla, o se non entra in un `i32` (il tipo
    /// usato dall'aritmetica delle distanze).
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        assert!(width > 0 && height > 0, "la griglia non puo' essere vuota");
        assert!(
            i32::try_from(width).is_ok() && i32::try_from(height).is_ok(),
            "dimensioni della griglia fuori dall'intervallo di i32"
        );
        Self { width, height }
    }

    /// Numero di celle.
    #[must_use]
    pub const fn len(self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Sempre `false`: [`Grid::new`] rifiuta le griglie vuote.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        false
    }

    /// Larghezza come `i32`, per l'aritmetica dell'avvolgimento.
    #[must_use]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "Grid::new ha gia' verificato che width entri in un i32"
    )]
    const fn width_i32(self) -> i32 {
        self.width as i32
    }

    /// Indice piatto in ordine di riga, lo stesso layout C di un array `NumPy`
    /// `(height, width)`: cosi' il ponte consegna una vista senza trasporre.
    #[must_use]
    pub const fn flat_index(self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }
}

/// Riempie `out` con la distanza minima da una qualunque cella di supporto.
///
/// Quando non ci sono target ogni cella riceve [`NO_SUPPORT`]: e' il caso dei
/// mondi minimi di test che non contengono alcun insediamento, e il lato Python
/// lo tratta esplicitamente come "esplorazione libera".
///
/// # Panics
/// Se `out` non ha esattamente `grid.len()` elementi, o se i due array di
/// coordinate dei target hanno lunghezze diverse.
pub fn support_distance_field(
    grid: Grid,
    target_columns: &[i32],
    target_rows: &[i32],
    out: &mut [i32],
) {
    assert_eq!(
        target_columns.len(),
        target_rows.len(),
        "le coordinate dei target devono essere appaiate"
    );
    assert_eq!(
        out.len(),
        grid.len(),
        "il campo deve avere width*height elementi"
    );

    if target_columns.is_empty() {
        out.fill(NO_SUPPORT);
        return;
    }

    let width = grid.width_i32();
    for y in 0..grid.height {
        let row_start = grid.flat_index(0, y);
        for x in 0..grid.width {
            let mut best = i32::MAX;
            for (&target_x, &target_y) in target_columns.iter().zip(target_rows.iter()) {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "x e y sono limitati dalle dimensioni verificate in Grid::new"
                )]
                let distance = wrapped_chebyshev(x as i32, y as i32, target_x, target_y, width);
                if distance < best {
                    best = distance;
                    if best == 0 {
                        // Una struttura coincide con la cella: nessun altro
                        // target puo' fare meglio.
                        break;
                    }
                }
            }
            out[row_start + (x as usize)] = best;
        }
    }
}

/// Parametri di razionamento, identici alle costanti del lato Python.
#[derive(Debug, Clone, Copy)]
pub struct RationPolicy {
    /// Razione consumata per singolo step di viaggio (`SCOUT_RATION_PER_STEP`).
    pub per_step: f64,
    /// Margine di sicurezza in step, sommato al viaggio (`safety_steps`).
    pub safety_steps: i32,
}

impl Default for RationPolicy {
    fn default() -> Self {
        Self {
            per_step: 0.1,
            safety_steps: 2,
        }
    }
}

/// I due campi di distanza richiesti dalla decisione di rientro.
///
/// Cibo e acqua hanno insiemi di strutture di supporto diversi (la serra per il
/// cibo; serra, habitat o infermeria per l'acqua), quindi due campi distinti.
#[derive(Debug, Clone, Copy)]
pub struct SupportFields<'a> {
    /// Distanza dalla struttura di supporto alimentare piu' vicina.
    pub food: &'a [i32],
    /// Distanza dalla struttura di supporto idrico piu' vicina.
    pub water: &'a [i32],
}

/// Stato per-agente rilevante per la raggiungibilita'.
#[derive(Debug, Clone, Copy)]
pub struct AgentRations<'a> {
    /// Cibo trasportato.
    pub food: &'a [f64],
    /// Acqua trasportata, ghiaccio incluso.
    pub water: &'a [f64],
    /// Step necessari ad attraversare una cella, per agente.
    pub steps_per_cell: &'a [i32],
}

/// Coppie (agente, cella) da valutare.
///
/// Questa forma evita di calcolare l'intera matrice agenti x celle: nel percorso
/// reale ogni agente valuta solo le celle adiacenti, quindi la matrice completa
/// sarebbe due ordini di grandezza di lavoro in piu' di quello richiesto.
#[derive(Debug, Clone, Copy)]
pub struct Pairs<'a> {
    /// Indice dell'agente della coppia.
    pub agent: &'a [u32],
    /// Indice piatto della cella candidata della coppia.
    pub cell: &'a [u32],
}

/// Razione necessaria per raggiungere il supporto e tornare indietro.
///
/// `None` quando la cella non ha supporto raggiungibile, che e' il caso in cui il
/// lato Python restituisce `(inf, None)`.
fn ration_requirement(distance: i32, steps_per_cell: i32, policy: RationPolicy) -> Option<f64> {
    if distance == NO_SUPPORT {
        return None;
    }
    let travel_steps = distance * steps_per_cell;
    Some(f64::from(travel_steps + policy.safety_steps.max(0)) * policy.per_step)
}

/// Calcola la maschera "posso visitare la cella e tornare" per ogni coppia.
///
/// # Panics
/// Se gli array appaiati hanno lunghezze incoerenti.
pub fn reachability_batch(
    pairs: Pairs<'_>,
    fields: SupportFields<'_>,
    rations: AgentRations<'_>,
    policy: RationPolicy,
    out: &mut [bool],
) {
    assert_eq!(pairs.agent.len(), pairs.cell.len(), "coppie non appaiate");
    assert_eq!(
        pairs.agent.len(),
        out.len(),
        "output di lunghezza diversa dalle coppie"
    );
    assert_eq!(
        fields.food.len(),
        fields.water.len(),
        "campi di distanza incoerenti"
    );
    assert_eq!(
        rations.food.len(),
        rations.water.len(),
        "inventari non appaiati"
    );
    assert_eq!(
        rations.food.len(),
        rations.steps_per_cell.len(),
        "steps_per_cell non appaiato agli inventari"
    );

    let after_move = policy.per_step;
    for (slot, (&agent_index, &cell_index)) in out
        .iter_mut()
        .zip(pairs.agent.iter().zip(pairs.cell.iter()))
    {
        let agent = agent_index as usize;
        let cell = cell_index as usize;
        let steps_per_cell = rations.steps_per_cell[agent];

        let food_required = ration_requirement(fields.food[cell], steps_per_cell, policy);
        let water_required = ration_requirement(fields.water[cell], steps_per_cell, policy);

        *slot = match (food_required, water_required) {
            // Nessun supporto di alcun tipo: il lato Python preserva
            // esplicitamente l'esplorazione libera nei mondi minimi che non
            // contengono alcun insediamento.
            (None, None) => true,
            // Un solo tipo di supporto mancante e' invece una situazione in cui
            // il rientro non e' garantito, e va negata.
            (None, Some(_)) | (Some(_), None) => false,
            (Some(food), Some(water)) => {
                rations.food[agent] - after_move >= food
                    && rations.water[agent] - after_move >= water
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Griglia 4x3 usata dai test del campo di distanza.
    fn small_grid() -> Grid {
        Grid::new(4, 3)
    }

    fn at(grid: Grid, x: u32, y: u32) -> usize {
        grid.flat_index(x, y)
    }

    #[test]
    fn wrapped_axis_takes_the_short_way_round() {
        // Griglia larga 36: da x=1 a x=35 la via corta e' 2, non 34.
        assert_eq!(wrapped_chebyshev(1, 0, 35, 0, 36), 2);
        // La latitudine non si avvolge.
        assert_eq!(wrapped_chebyshev(0, 1, 0, 20, 36), 19);
        // Chebyshev prende il massimo dei due assi.
        assert_eq!(wrapped_chebyshev(0, 0, 3, 5, 36), 5);
    }

    #[test]
    fn empty_target_set_marks_every_cell_unsupported() {
        let grid = Grid::new(3, 2);
        let mut field = vec![0; grid.len()];
        support_distance_field(grid, &[], &[], &mut field);
        assert!(field.iter().all(|&value| value == NO_SUPPORT));
    }

    #[test]
    fn field_is_zero_on_the_target_and_grows_outward() {
        let grid = small_grid();
        let mut field = vec![0; grid.len()];
        support_distance_field(grid, &[1], &[1], &mut field);
        assert_eq!(field[at(grid, 1, 1)], 0);
        assert_eq!(field[at(grid, 1, 0)], 1);
        assert_eq!(field[at(grid, 3, 2)], 2);
    }

    #[test]
    fn field_takes_the_minimum_over_targets() {
        let grid = small_grid();
        let mut one = vec![0; grid.len()];
        let mut both = vec![0; grid.len()];
        support_distance_field(grid, &[0], &[0], &mut one);
        support_distance_field(grid, &[0, 3], &[0, 2], &mut both);
        for (left, right) in both.iter().zip(one.iter()) {
            assert!(left <= right, "aggiungere un target non puo' allontanare");
        }
        assert_eq!(both[at(grid, 3, 2)], 0);
    }

    #[test]
    fn field_wraps_across_the_antimeridian() {
        // Un target a x=0 deve risultare vicino alla colonna x=3 di una griglia
        // larga 4: e' il caso che una distanza non avvolta sbaglierebbe.
        let grid = Grid::new(4, 1);
        let mut field = vec![0; grid.len()];
        support_distance_field(grid, &[0], &[0], &mut field);
        assert_eq!(field[at(grid, 3, 0)], 1);
    }

    fn single_pair_verdict(
        food_field: i32,
        water_field: i32,
        food: f64,
        water: f64,
        steps_per_cell: i32,
    ) -> bool {
        let mut out = [false];
        reachability_batch(
            Pairs {
                agent: &[0],
                cell: &[0],
            },
            SupportFields {
                food: &[food_field],
                water: &[water_field],
            },
            AgentRations {
                food: &[food],
                water: &[water],
                steps_per_cell: &[steps_per_cell],
            },
            RationPolicy::default(),
            &mut out,
        );
        out[0]
    }

    #[test]
    fn unsupported_world_allows_free_exploration() {
        assert!(single_pair_verdict(NO_SUPPORT, NO_SUPPORT, 0.0, 0.0, 1));
    }

    #[test]
    fn half_supported_world_is_refused() {
        assert!(
            !single_pair_verdict(NO_SUPPORT, 1, 99.0, 99.0, 1),
            "un solo tipo di supporto non garantisce il rientro"
        );
        assert!(!single_pair_verdict(1, NO_SUPPORT, 99.0, 99.0, 1));
    }

    #[test]
    fn rations_gate_the_visit() {
        // distanza 3, un passo per cella, margine 2 -> (3 + 2) * 0,1 = 0,5,
        // piu' la razione consumata dallo spostamento stesso: servono 0,6.
        assert!(
            single_pair_verdict(3, 3, 0.6, 0.6, 1),
            "0,6 di scorta copre esattamente il fabbisogno"
        );
        assert!(!single_pair_verdict(3, 3, 0.59, 0.59, 1));
    }

    #[test]
    fn food_and_water_are_gated_independently() {
        assert!(
            !single_pair_verdict(3, 3, 0.6, 0.1, 1),
            "acqua insufficiente"
        );
        assert!(
            !single_pair_verdict(3, 3, 0.1, 0.6, 1),
            "cibo insufficiente"
        );
    }

    #[test]
    fn more_steps_per_cell_make_the_same_cell_unreachable() {
        assert!(single_pair_verdict(3, 3, 0.6, 0.6, 1));
        assert!(
            !single_pair_verdict(3, 3, 0.6, 0.6, 2),
            "raddoppiando i passi per cella il viaggio costa il doppio"
        );
    }

    #[test]
    fn pairs_are_evaluated_independently() {
        let mut out = [false; 3];
        reachability_batch(
            Pairs {
                agent: &[0, 1, 0],
                cell: &[0, 0, 1],
            },
            SupportFields {
                food: &[3, NO_SUPPORT],
                water: &[3, NO_SUPPORT],
            },
            AgentRations {
                food: &[0.6, 0.1],
                water: &[0.6, 0.1],
                steps_per_cell: &[1, 1],
            },
            RationPolicy::default(),
            &mut out,
        );
        assert_eq!(out, [true, false, true]);
    }
}
