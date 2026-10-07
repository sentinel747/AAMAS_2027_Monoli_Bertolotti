//! Ponte `PyO3` fra la shell Python di `MarsABM` e il kernel `mars_core`.
//!
//! Regola architetturale del piano di migrazione: qui dentro sta *solo* la
//! traduzione fra oggetti Python e fette contigue di memoria. Nessuna decisione
//! di simulazione, nessuna formula, nessuna euristica. Ogni volta che una
//! funzione di questo modulo e' tentata di "sistemare un caso particolare", il
//! caso particolare appartiene a `mars_core`, dove e' testabile senza un
//! interprete Python.
//!
//! Il ponte e' inoltre l'unico punto in cui un errore di tipo o di forma puo'
//! ancora essere intercettato con un messaggio comprensibile. Piu' a valle, un
//! array con dtype sbagliato non produce un errore: produce numeri sbagliati.
//! Per questo ogni ingresso e' validato esplicitamente e ogni messaggio dice
//! *quale* argomento, *cosa* si aspettava e *cosa* ha ricevuto.

use mars_core::physio::{
    AgentColumns, GatePolicy, StructureGrid, SupportPolicy, skip_override,
};
use mars_core::reachability::{
    AgentRations, Grid, NO_SUPPORT, Pairs, RationPolicy, SupportFields, reachability_batch,
    support_distance_field,
};
use mars_core::scoring::{
    AgentRow, N_PILLARS, PillarTables, score as score_pillars_native,
};
use numpy::ndarray::Dimension;
use numpy::{
    PyArray1, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Stringa di versione del kernel nativo.
#[pyfunction]
fn version() -> String {
    mars_core::version()
}

/// Versione dello schema di stato che questo binario si aspetta.
///
/// Il lato Python confronta questo valore con il proprio prima di scambiare
/// buffer: e' il controllo che impedisce a due binari disallineati di
/// interpretare gli stessi byte con indici di risorsa diversi.
#[pyfunction]
fn schema_version() -> u32 {
    mars_core::SCHEMA_VERSION
}

/// Verifica che un array sia contiguo, e ne restituisce la fetta in ordine di riga.
///
/// La contiguita' non e' un dettaglio: `numpy` puo' restituire viste con passo
/// diverso da uno (una colonna di una matrice, una slice con step). Leggerle come
/// se fossero contigue darebbe valori presi dalle posizioni sbagliate, in
/// silenzio. Il costo del controllo e' trascurabile rispetto al costo di scoprire
/// quel tipo di errore attraverso una divergenza numerica.
///
/// Vale per qualunque numero di dimensioni: un array `[h, w, k]` contiguo e' una
/// fetta di `h*w*k` elementi, e il kernel ricostruisce gli indici dalla forma. E'
/// il modo in cui il lato nativo puo' ricevere una colonna **intera** e fare da
/// se' la selezione delle righe, che e' la regola ricavata dal Task A.
fn contiguous<'py, T, D>(name: &str, array: &'py PyReadonlyArray<'py, T, D>) -> PyResult<&'py [T]>
where
    T: numpy::Element,
    D: Dimension,
{
    array.as_slice().map_err(|_| {
        PyTypeError::new_err(format!(
            "l'argomento `{name}` deve essere un array NumPy contiguo; \
             usare numpy.ascontiguousarray() prima di passarlo"
        ))
    })
}

fn same_length(first_name: &str, first: usize, second_name: &str, second: usize) -> PyResult<()> {
    if first == second {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "`{first_name}` e `{second_name}` devono avere la stessa lunghezza, \
         ricevuti {first} e {second}"
    )))
}

/// Campo delle distanze di supporto per l'intera griglia.
///
/// Restituisce un array `int32` di `height * width` elementi in ordine di riga,
/// gia' nella forma `(height, width)` attesa dal lato Python. Le celle senza
/// alcun supporto raggiungibile valgono `NO_SUPPORT` (-1).
#[pyfunction]
#[pyo3(signature = (width, height, target_columns, target_rows))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 estrae gli argomenti per valore: prenderli per riferimento non \
              compila, perche' l'estrattore possiede la guardia di prestito verso \
              il buffer NumPy"
)]
fn support_distance_field_py<'py>(
    py: Python<'py>,
    width: u32,
    height: u32,
    target_columns: PyReadonlyArray1<'py, i32>,
    target_rows: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err(
            "la griglia non puo' avere dimensioni nulle",
        ));
    }
    same_length(
        "target_columns",
        target_columns.len(),
        "target_rows",
        target_rows.len(),
    )?;

    let columns = contiguous("target_columns", &target_columns)?;
    let rows = contiguous("target_rows", &target_rows)?;
    let grid = Grid::new(width, height);

    let mut field = vec![0_i32; grid.len()];
    // `allow_threads` rilascia il GIL: il kernel non tocca alcun oggetto Python,
    // quindi trattenerlo bloccherebbe altri thread senza motivo. E' anche cio'
    // che rende il confronto onesto rispetto a un equivalente NumPy, che il GIL
    // lo rilascia a sua volta nelle proprie primitive.
    py.detach(|| support_distance_field(grid, columns, rows, &mut field));

    Ok(PyArray1::from_vec(py, field))
}

/// Maschera "posso visitare la cella e tornare" per coppie (agente, cella).
///
/// Ogni coppia `i` riguarda l'agente `pair_agent[i]` e la cella di indice piatto
/// `pair_cell[i]`. Gli indici vengono validati contro le lunghezze effettive: un
/// indice fuori intervallo qui e' un errore di costruzione del batch lato Python,
/// e va segnalato come tale invece di provocare un panic dentro il kernel.
#[pyfunction]
#[pyo3(signature = (
    pair_agent, pair_cell, food_field, water_field,
    agent_food, agent_water, agent_steps_per_cell,
    ration_per_step, safety_steps,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "la firma rispecchia i buffer del batch; raggrupparli in un oggetto \
              Python reintrodurrebbe proprio il costo per-chiamata che si vuole togliere"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 estrae gli argomenti per valore: vedi la nota su \
              support_distance_field_py"
)]
fn reachability_batch_py<'py>(
    py: Python<'py>,
    pair_agent: PyReadonlyArray1<'py, u32>,
    pair_cell: PyReadonlyArray1<'py, u32>,
    food_field: PyReadonlyArray1<'py, i32>,
    water_field: PyReadonlyArray1<'py, i32>,
    agent_food: PyReadonlyArray1<'py, f64>,
    agent_water: PyReadonlyArray1<'py, f64>,
    agent_steps_per_cell: PyReadonlyArray1<'py, i32>,
    ration_per_step: f64,
    safety_steps: i32,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    same_length("pair_agent", pair_agent.len(), "pair_cell", pair_cell.len())?;
    same_length(
        "food_field",
        food_field.len(),
        "water_field",
        water_field.len(),
    )?;
    same_length(
        "agent_food",
        agent_food.len(),
        "agent_water",
        agent_water.len(),
    )?;
    same_length(
        "agent_food",
        agent_food.len(),
        "agent_steps_per_cell",
        agent_steps_per_cell.len(),
    )?;

    let agents = contiguous("pair_agent", &pair_agent)?;
    let cells = contiguous("pair_cell", &pair_cell)?;
    let food = contiguous("food_field", &food_field)?;
    let water = contiguous("water_field", &water_field)?;
    let carried_food = contiguous("agent_food", &agent_food)?;
    let carried_water = contiguous("agent_water", &agent_water)?;
    let steps_per_cell = contiguous("agent_steps_per_cell", &agent_steps_per_cell)?;

    // Validazione degli indici PRIMA di entrare nel kernel: dentro `mars_core`
    // un indice fuori intervallo sarebbe un panic, che PyO3 convertirebbe in
    // `PanicException` - un messaggio che non dice quale array e' sbagliato.
    let agent_count = carried_food.len();
    let cell_count = food.len();
    if let Some(&bad) = agents.iter().find(|&&index| index as usize >= agent_count) {
        return Err(PyValueError::new_err(format!(
            "`pair_agent` contiene l'indice {bad}, fuori dai {agent_count} agenti forniti"
        )));
    }
    if let Some(&bad) = cells.iter().find(|&&index| index as usize >= cell_count) {
        return Err(PyValueError::new_err(format!(
            "`pair_cell` contiene l'indice {bad}, fuori dalle {cell_count} celle del campo"
        )));
    }

    let mut verdicts = vec![false; agents.len()];
    py.detach(|| {
        reachability_batch(
            Pairs {
                agent: agents,
                cell: cells,
            },
            SupportFields { food, water },
            AgentRations {
                food: carried_food,
                water: carried_water,
                steps_per_cell,
            },
            RationPolicy {
                per_step: ration_per_step,
                safety_steps,
            },
            &mut verdicts,
        );
    });

    Ok(PyArray1::from_vec(py, verdicts))
}

/// Attraversamento **a vuoto** del confine, con il carico di un port reale.
///
/// Task A del piano `2026-08-07-decision-loop-rust-port.md`, ed e' bloccante.
/// Riceve tutte le colonne che il ciclo decisionale legge, non calcola nulla, e
/// restituisce la somma delle loro lunghezze -- che serve solo a impedire al
/// compilatore di eliminare gli argomenti come codice morto.
///
/// Misura una cosa sola: **quanto costa passare il confine una volta per passo**,
/// validazioni comprese. Se quel costo da solo erode il tetto del port, il
/// progetto si chiude prima di scrivere la logica. E' il punto (iii) del criterio
/// di misura -- contare il costo di interfaccia prima di costruire -- applicato al
/// port stesso, dopo che il Task 3 ha mostrato quanto costi scoprirlo alla fine.
///
/// Il carico e' quello vero, non un campione: gli array di cella sono passati per
/// riferimento da `rust-numpy`, quindi la loro dimensione non dovrebbe pesare. E'
/// esattamente l'ipotesi che questa funzione serve a verificare invece che a
/// dare per buona.
#[pyfunction]
#[expect(
    clippy::too_many_arguments,
    reason = "la firma E' la misura: raggruppare gli argomenti in una struct \
              sposterebbe il costo di conversione altrove e falserebbe il numero"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 estrae gli argomenti per valore: vedi la nota su \
              support_distance_field_py"
)]
fn crossing_probe(
    rows: PyReadonlyArray1<'_, i64>,
    x: PyReadonlyArray1<'_, i16>,
    y: PyReadonlyArray1<'_, i16>,
    hydration: PyReadonlyArray1<'_, f64>,
    satiety: PyReadonlyArray1<'_, f64>,
    uniforms: PyReadonlyArray1<'_, f64>,
    inv: PyReadonlyArray2<'_, f64>,
    pref: PyReadonlyArray2<'_, f64>,
    skill: PyReadonlyArray2<'_, f64>,
    masks: PyReadonlyArray2<'_, bool>,
    priorities: PyReadonlyArray2<'_, f64>,
    quotas: PyReadonlyArray2<'_, i32>,
    struct_count: PyReadonlyArray3<'_, i16>,
    site_progress: PyReadonlyArray3<'_, f64>,
    cell_res: PyReadonlyArray3<'_, f64>,
    occupancy: PyReadonlyArray2<'_, i16>,
) -> PyResult<usize> {
    // Le stesse verifiche di contiguita' che farebbe un port vero: leggere una
    // vista non contigua come se lo fosse darebbe numeri sbagliati in silenzio.
    let rows = contiguous("rows", &rows)?;
    let x = contiguous("x", &x)?;
    let y = contiguous("y", &y)?;
    let hydration = contiguous("hydration", &hydration)?;
    let satiety = contiguous("satiety", &satiety)?;
    let uniforms = contiguous("uniforms", &uniforms)?;
    same_length("rows", rows.len(), "x", x.len())?;
    same_length("rows", rows.len(), "y", y.len())?;
    same_length("rows", rows.len(), "hydration", hydration.len())?;
    same_length("rows", rows.len(), "satiety", satiety.len())?;
    same_length("rows", rows.len(), "uniforms", uniforms.len())?;

    // Solo le dimensioni: nessuna lettura degli elementi, che e' il punto.
    Ok(rows.len()
        + inv.shape()[0]
        + pref.shape()[0]
        + skill.shape()[0]
        + masks.shape()[0]
        + priorities.shape()[0]
        + quotas.shape()[0]
        + struct_count.shape()[0]
        + site_progress.shape()[0]
        + cell_res.shape()[0]
        + occupancy.shape()[0])
}

/// Verifica che ogni indice di configurazione stia dentro la sua dimensione.
fn within(name: &str, what: &str, indices: &[usize], bound: usize) -> PyResult<()> {
    for &index in indices {
        if index >= bound {
            return Err(PyValueError::new_err(format!(
                "`{name}` contiene {index}: il massimo e' {bound} ({what})"
            )));
        }
    }
    Ok(())
}

/// Verifica che ogni riga del batch esista e stia dentro la griglia.
///
/// Separata dal corpo di `physio_gate` perche' e' la parte che produce i
/// messaggi d'errore, e leggerla insieme al predicato nasconderebbe entrambi.
fn addressable(
    rows: &[i64],
    x: &[i16],
    y: &[i16],
    capacity: usize,
    width: usize,
    height: usize,
) -> PyResult<()> {
    for &row in rows {
        let Ok(index) = usize::try_from(row) else {
            return Err(PyValueError::new_err(format!(
                "`rows` contiene la riga negativa {row}"
            )));
        };
        if index >= capacity {
            return Err(PyValueError::new_err(format!(
                "`rows` contiene la riga {row}, fuori dagli {capacity} agenti forniti"
            )));
        }
        let (Ok(cell_x), Ok(cell_y)) = (usize::try_from(x[index]), usize::try_from(y[index]))
        else {
            return Err(PyValueError::new_err(format!(
                "la riga {row} ha coordinate negative"
            )));
        };
        if cell_x >= width || cell_y >= height {
            return Err(PyValueError::new_err(format!(
                "la riga {row} sta in ({cell_x}, {cell_y}), fuori dalla \
                 griglia {width}x{height}"
            )));
        }
    }
    Ok(())
}

/// Filtro batch del guardiano fisiologico: chi puo' saltare l'override.
///
/// Task C.1. Restituisce un array di booleani lungo quanto `rows`: `True`
/// significa che `_physiological_override` restituirebbe `None` con certezza, e
/// che il ciclo sequenziale puo' non chiamarla affatto.
///
/// Le colonne arrivano **intere** e la selezione delle righe avviene qui dentro:
/// `rust-numpy` le condivide per riferimento, quindi passarle tutte non costa
/// nulla, mentre estrarne le righe lato Python costerebbe -- e' la lezione
/// misurata nel Task A, dove la preparazione degli argomenti valeva il 90,9% del
/// costo osservato di un attraversamento.
///
/// Indici di colonna, rapporti e soglie sono argomenti perche' la loro
/// definizione vive lato Python: duplicarli qui creerebbe due sorgenti di verita'
/// che possono divergere in silenzio.
#[pyfunction]
#[pyo3(signature = (
    rows, x, y, hydration, satiety, health,
    steps_without_water, steps_without_food, inv, struct_count,
    resource_index, structure_index, support_ratios, thresholds, guard_steps,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "la firma rispecchia le colonne del batch; raggrupparle in un oggetto \
              Python reintrodurrebbe il costo per-chiamata che il port vuole togliere"
)]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 estrae gli argomenti per valore: vedi la nota su \
              support_distance_field_py"
)]
fn physio_gate<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i64>,
    x: PyReadonlyArray1<'py, i16>,
    y: PyReadonlyArray1<'py, i16>,
    hydration: PyReadonlyArray1<'py, f64>,
    satiety: PyReadonlyArray1<'py, f64>,
    health: PyReadonlyArray1<'py, f64>,
    steps_without_water: PyReadonlyArray1<'py, i16>,
    steps_without_food: PyReadonlyArray1<'py, i16>,
    inv: PyReadonlyArray2<'py, f64>,
    struct_count: PyReadonlyArray3<'py, i16>,
    resource_index: (usize, usize, usize),
    structure_index: (usize, usize, usize, usize, usize, usize),
    support_ratios: (i64, i64, i64),
    thresholds: (f64, f64, f64, f64, f64),
    guard_steps: (i32, i32),
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let capacity = hydration.len();
    for (name, length) in [
        ("x", x.len()),
        ("y", y.len()),
        ("satiety", satiety.len()),
        ("health", health.len()),
        ("steps_without_water", steps_without_water.len()),
        ("steps_without_food", steps_without_food.len()),
    ] {
        same_length("hydration", capacity, name, length)?;
    }

    let inv_shape = inv.shape();
    let (inv_rows, resource_count) = (inv_shape[0], inv_shape[1]);
    same_length("hydration", capacity, "inv", inv_rows)?;
    let grid_shape = struct_count.shape();
    let (height, width, structure_count) = (grid_shape[0], grid_shape[1], grid_shape[2]);

    let rows_slice = contiguous("rows", &rows)?;
    let x_slice = contiguous("x", &x)?;
    let y_slice = contiguous("y", &y)?;
    let inv_slice = contiguous("inv", &inv)?;
    let counts_slice = contiguous("struct_count", &struct_count)?;

    // Validazione degli indici PRIMA del kernel: dentro `mars_core` sarebbero
    // panic, e un `PanicException` non dice quale array e' sbagliato.
    within("resource_index", "risorse per riga di `inv`", &[
        resource_index.0,
        resource_index.1,
        resource_index.2,
    ], resource_count)?;
    within("structure_index", "tipi di struttura per cella", &[
        structure_index.0,
        structure_index.1,
        structure_index.2,
        structure_index.3,
        structure_index.4,
        structure_index.5,
    ], structure_count)?;
    addressable(rows_slice, x_slice, y_slice, capacity, width, height)?;

    let agents = AgentColumns {
        rows: rows_slice,
        x: x_slice,
        y: y_slice,
        hydration: contiguous("hydration", &hydration)?,
        satiety: contiguous("satiety", &satiety)?,
        health: contiguous("health", &health)?,
        steps_without_water: contiguous("steps_without_water", &steps_without_water)?,
        steps_without_food: contiguous("steps_without_food", &steps_without_food)?,
        inv: inv_slice,
        resource_count,
    };
    let grid = StructureGrid {
        counts: counts_slice,
        width,
        structure_count,
    };
    let support = SupportPolicy {
        shelter: structure_index.0,
        habitat: structure_index.1,
        infirmary: structure_index.2,
        greenhouse: structure_index.3,
        solar_array: structure_index.4,
        oxygen_plant: structure_index.5,
        greenhouse_ratio: support_ratios.0,
        solar_ratio: support_ratios.1,
        oxygen_ratio: support_ratios.2,
    };
    let policy = GatePolicy {
        water_index: resource_index.0,
        ice_index: resource_index.1,
        food_index: resource_index.2,
        critical_hydration: thresholds.0,
        critical_satiety: thresholds.1,
        critical_health: thresholds.2,
        water_return_reserve: thresholds.3,
        food_return_reserve: thresholds.4,
        water_guard_steps: guard_steps.0,
        food_guard_steps: guard_steps.1,
    };

    let mut verdicts = vec![false; rows_slice.len()];
    py.detach(|| skip_override(&agents, &grid, &support, &policy, &mut verdicts));
    Ok(PyArray1::from_vec(py, verdicts))
}

/// Punteggio dei sei pilastri per una decisione, con le tabelle costanti tenute.
///
/// Task C.2. E' una **classe** e non una funzione per una ragione di misura: gli
/// indici di azione per pilastro, gli indici di risorsa e i codici di missione
/// sono costanti, e passarli a ogni chiamata li farebbe attraversare il confine
/// ~300 volte per passo senza motivo. Qui attraversano una volta, alla
/// costruzione, e restano.
///
/// Le colonne per-agente arrivano **intere** con l'indice di riga, non
/// pre-selezionate: `rust-numpy` le condivide per riferimento, quindi la
/// selezione costa meno fatta qui che lato Python. E' la regola ricavata dal
/// Task A.
#[pyclass]
struct DecisionScorer {
    tables: PillarTables,
}

#[pymethods]
impl DecisionScorer {
    #[new]
    #[pyo3(signature = (
        pillar_actions, build_indices, resource_index, mission_active_codes,
        urgency_exponent,
    ))]
    fn new(
        pillar_actions: Vec<Vec<usize>>,
        build_indices: Vec<usize>,
        resource_index: (usize, usize),
        mission_active_codes: Vec<i8>,
        urgency_exponent: f64,
    ) -> PyResult<Self> {
        if pillar_actions.len() != N_PILLARS {
            return Err(PyValueError::new_err(format!(
                "`pillar_actions` deve avere {N_PILLARS} voci, ricevute {}",
                pillar_actions.len()
            )));
        }
        Ok(Self {
            tables: PillarTables {
                pillar_actions,
                build_indices,
                material_index: resource_index.0,
                minerals_index: resource_index.1,
                mission_active_codes,
                urgency_exponent,
            },
        })
    }

    /// Punteggi normalizzati e pilastro scelto per la riga `row`.
    #[pyo3(signature = (
        row, mask, action_priority, hydration, satiety, health, fatigue, stress,
        morale, curiosity, steps_without_water, steps_without_food,
        inv, pref, skill, mission, founder_kit_reserved, u01, greedy, survival,
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "la firma rispecchia le colonne del batch; raggrupparle in un \
                  oggetto Python reintrodurrebbe il costo per-chiamata che il port \
                  vuole togliere"
    )]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 estrae gli argomenti per valore: vedi la nota su \
                  support_distance_field_py"
    )]
    fn score<'py>(
        &self,
        py: Python<'py>,
        row: usize,
        mask: PyReadonlyArray1<'py, bool>,
        action_priority: PyReadonlyArray1<'py, f64>,
        hydration: PyReadonlyArray1<'py, f64>,
        satiety: PyReadonlyArray1<'py, f64>,
        health: PyReadonlyArray1<'py, f64>,
        fatigue: PyReadonlyArray1<'py, f64>,
        stress: PyReadonlyArray1<'py, f64>,
        morale: PyReadonlyArray1<'py, f64>,
        curiosity: PyReadonlyArray1<'py, f64>,
        steps_without_water: PyReadonlyArray1<'py, i16>,
        steps_without_food: PyReadonlyArray1<'py, i16>,
        inv: PyReadonlyArray2<'py, f64>,
        pref: PyReadonlyArray2<'py, f64>,
        skill: PyReadonlyArray2<'py, f64>,
        mission: PyReadonlyArray1<'py, i8>,
        founder_kit_reserved: bool,
        u01: f64,
        greedy: bool,
        survival: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
        let mask_slice = contiguous("mask", &mask)?;
        let priority_slice = contiguous("action_priority", &action_priority)?;
        same_length("mask", mask_slice.len(), "action_priority", priority_slice.len())?;

        // Le colonne arrivano SEPARATE e non impilate in una matrice: impilarle
        // richiederebbe una copia per chiamata, oppure una cache per passo
        // giustificata da un invariante nascosto. Nove estrazioni in piu' costano
        // meno di entrambe le cose, e non aggiungono niente da tenere coerente.
        let capacity = hydration.len();
        for (name, length) in [
            ("satiety", satiety.len()),
            ("health", health.len()),
            ("fatigue", fatigue.len()),
            ("stress", stress.len()),
            ("morale", morale.len()),
            ("curiosity", curiosity.len()),
            ("steps_without_water", steps_without_water.len()),
            ("steps_without_food", steps_without_food.len()),
            ("mission", mission.len()),
        ] {
            same_length("hydration", capacity, name, length)?;
        }
        if row >= capacity {
            return Err(PyValueError::new_err(format!(
                "`row` vale {row}, fuori dagli {capacity} agenti forniti"
            )));
        }
        let inv_shape = inv.shape();
        for (name, index) in [
            ("material", self.tables.material_index),
            ("minerals", self.tables.minerals_index),
        ] {
            if index >= inv_shape[1] {
                return Err(PyValueError::new_err(format!(
                    "l'indice di `{name}` vale {index}, fuori dalle {} risorse                      per riga di `inv`",
                    inv_shape[1]
                )));
            }
        }
        if pref.shape()[1] != N_PILLARS || skill.shape()[1] != N_PILLARS {
            return Err(PyValueError::new_err(format!(
                "`pref` e `skill` devono avere {N_PILLARS} colonne, ricevute {} e {}",
                pref.shape()[1],
                skill.shape()[1]
            )));
        }
        if let Some(&bad) = self
            .tables
            .pillar_actions
            .iter()
            .flatten()
            .chain(self.tables.build_indices.iter())
            .find(|&&index| index >= mask_slice.len())
        {
            return Err(PyValueError::new_err(format!(
                "una tabella costante contiene l'indice di azione {bad}, fuori                  dalle {} azioni della maschera",
                mask_slice.len()
            )));
        }

        let inv_view = inv.as_array();
        let pref_view = pref.as_array();
        let skill_view = skill.as_array();
        let mission_slice = contiguous("mission", &mission)?;

        let preferences: Vec<f64> = (0..N_PILLARS).map(|p| pref_view[[row, p]]).collect();
        let skills: Vec<f64> = (0..N_PILLARS).map(|p| skill_view[[row, p]]).collect();
        let agent = AgentRow {
            mask: mask_slice,
            action_priority: priority_slice,
            preferences: &preferences,
            skills: &skills,
            hydration: contiguous("hydration", &hydration)?[row],
            satiety: contiguous("satiety", &satiety)?[row],
            health: contiguous("health", &health)?[row],
            fatigue: contiguous("fatigue", &fatigue)?[row],
            stress: contiguous("stress", &stress)?[row],
            morale: contiguous("morale", &morale)?[row],
            curiosity: contiguous("curiosity", &curiosity)?[row],
            material: inv_view[[row, self.tables.material_index]],
            minerals: inv_view[[row, self.tables.minerals_index]],
            steps_without_water: i32::from(contiguous("steps_without_water", &steps_without_water)?[row]),
            steps_without_food: i32::from(contiguous("steps_without_food", &steps_without_food)?[row]),
            mission: mission_slice[row],
            founder_kit_reserved,
        };

        let (scores, chosen) = score_pillars_native(&agent, &self.tables, survival, u01, greedy);
        Ok((PyArray1::from_slice(py, &scores), chosen))
    }
}

#[pymodule]
fn mars_core_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DecisionScorer>()?;
    module.add_function(wrap_pyfunction!(crossing_probe, module)?)?;
    module.add_function(wrap_pyfunction!(physio_gate, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(schema_version, module)?)?;
    module.add_function(wrap_pyfunction!(support_distance_field_py, module)?)?;
    module.add_function(wrap_pyfunction!(reachability_batch_py, module)?)?;
    module.add("SCHEMA_VERSION", mars_core::SCHEMA_VERSION)?;
    module.add("KERNEL_VERSION", mars_core::KERNEL_VERSION)?;
    module.add("NO_SUPPORT", NO_SUPPORT)?;
    Ok(())
}
