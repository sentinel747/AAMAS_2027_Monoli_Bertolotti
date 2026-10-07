//! Kernel deterministico della simulazione `MarsABM`.
//!
//! Questo crate non conosce Python. Riceve fette (`&[T]`) di dati gia' in
//! memoria contigua e scrive su fette mutabili fornite dal chiamante: nessuna
//! allocazione per agente, nessun tipo che dipenda dall'interprete. Il ponte
//! `PyO3` vive in `mars_core_py`.
//!
//! Contenuto attuale: la versione dello schema condiviso, la raggiungibilita'
//! con rientro garantito portata dal Task 4 ([`reachability`]) e il filtro batch
//! del guardiano fisiologico del Task C.1 ([`physio`]).

pub mod physio;
pub mod reachability;
pub mod scoring;

/// Versione dello schema di stato condiviso fra Python e Rust.
///
/// Il ponte confronta questo numero con quello del lato Python a ogni
/// inizializzazione: un disallineamento significa che i due lati hanno idee
/// diverse su indici di risorsa, codici azione o layout dei buffer, ed e' molto
/// meglio scoprirlo all'avvio che come una divergenza numerica allo step 400.
pub const SCHEMA_VERSION: u32 = 1;

/// Versione del kernel nativo, riportata nel manifest di ogni run.
pub const KERNEL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Descrizione del kernel nativo, per i controlli di sanita' del binding.
#[must_use]
pub fn version() -> String {
    format!("mars_core {KERNEL_VERSION} (schema v{SCHEMA_VERSION})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_reports_schema_and_package() {
        let reported = version();
        assert!(reported.contains(KERNEL_VERSION));
        assert!(reported.contains(&format!("schema v{SCHEMA_VERSION}")));
    }
}
