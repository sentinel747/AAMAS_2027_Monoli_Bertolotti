# -*- coding: utf-8 -*-
"""Le tre variabili sperimentali del prompt: la parola, la gerarchia, la lingua.

**Perche' esistono.** Sono tre obiezioni distinte al prompt usato da tutte le
campagne dell'archivio, e vanno misurate una per volta perche' una sola
esecuzione con tutte e tre cambiate non direbbe quale delle tre ha spostato il
risultato.

*La parola.* Il prompt chiama «pilastro» cio' che il modello pesa, e per giunta
non lo chiama sempre cosi': l'apertura dice «sei pilastri», il vocabolario dice
«Leve pesabili» e la meccanica dei pesi dice «le leve». Due nomi per la stessa
cosa nello stesso testo sono un difetto in se'. «Pilastro» e' inoltre una
metafora edilizia per quella che nel modello e' una **categoria di preferenza**:
il colono porta un vettore di preferenze normalizzato su di esse, e la politica
ne moltiplica i punteggi. La variante `categoria` usa una parola sola, vera, in
tutto il testo.

*La gerarchia.* Gli amministratori di distretto sanno che sopra di loro c'e' un
governo e ne leggono la politica per intero; il governatore non sa che sotto di
lui esistono amministratori che possono riscriverla. Sapendolo potrebbe scrivere
leggi piu' generali, lasciando la specializzazione al livello locale --- oppure
non cambiare niente. Non lo sappiamo, ed e' il motivo per cui e' un braccio e
non una correzione.

*La lingua.* Tutto il prompt e' in italiano mentre i modelli sono addestrati in
prevalenza su inglese. La traduzione e' una variabile, non una pulizia: fatta a
parte dalle altre due, dice quanto il risultato dipende dalla lingua e permette
di dichiarare la robustezza rispetto al prompt.

**Il contratto con la macchina non cambia mai.** I nomi degli indicatori
(`food_per_occupant`...), quelli delle categorie (`sustenance`...), gli
operatori e le chiavi del JSON di risposta sono gia' inglesi e restano
identici in ogni variante. Cambia solo la prosa che spiega. E' questo che
rende il confronto pulito: parser, validatore, applicazione vettoriale e
script di analisi non vedono la differenza.

**Il gradino cieco non e' toccato dalla parola.** Le sue etichette neutre sono
`leva_A..leva_E` (`context.py`), congelate perche' la traduzione inversa della
risposta ci si appoggia; li' la parola resta «leva» per non dire «categorie
pesabili: leva_A». La campagna incrociata gira al gradino completo, dove la
questione non si pone.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Le lingue in cui il prompt esiste.
LINGUE = ("it", "en")

#: I due nomi in gara per cio' che la politica pesa. `pilastro` e' il testo
#: storico, con la sua incoerenza inclusa: e' il braccio di riferimento e non va
#: ripulito, altrimenti non e' piu' il prompt con cui l'archivio e' stato
#: misurato.
TERMINI = ("pilastro", "categoria")


@dataclass(frozen=True)
class VariantePrompt:
    """Una casella del disegno incrociato.

    Il valore predefinito e' il prompt storico, parola per parola: qualunque
    codice che non conosca questa classe continua a produrre quel testo.
    """

    lingua: str = "it"
    termine: str = "pilastro"
    gerarchia: bool = False

    def __post_init__(self) -> None:
        if self.lingua not in LINGUE:
            raise ValueError(f"lingua sconosciuta: {self.lingua!r}; attese {LINGUE}")
        if self.termine not in TERMINI:
            raise ValueError(f"termine sconosciuto: {self.termine!r}; attesi {TERMINI}")

    @property
    def storica(self) -> bool:
        """E' il prompt con cui sono state misurate le campagne dell'archivio?"""
        return self == VariantePrompt()

    @property
    def sigla(self) -> str:
        """Il nome del braccio nel disegno incrociato, per registri e cartelle."""
        return SIGLE.get(
            (self.lingua, self.termine, self.gerarchia),
            f"{self.lingua}-{self.termine}-{'ger' if self.gerarchia else 'noger'}",
        )


#: Le sei caselle misurate, con la lettera usata in coda e nei rapporti.
SIGLE: dict[tuple[str, str, bool], str] = {
    ("it", "pilastro", False): "0",
    ("it", "categoria", False): "A",
    ("it", "categoria", True): "B",
    ("en", "pilastro", False): "F",
    ("en", "categoria", False): "C",
    ("en", "categoria", True): "D",
}


def variante_da_sigla(sigla: str) -> VariantePrompt:
    """Da «A» alla variante. Serve alle code e alla riga di comando."""
    for chiave, valore in SIGLE.items():
        if valore == sigla.strip().upper():
            lingua, termine, gerarchia = chiave
            return VariantePrompt(lingua=lingua, termine=termine, gerarchia=gerarchia)
    raise ValueError(f"sigla sconosciuta: {sigla!r}; attese {sorted(SIGLE.values())}")


# --------------------------------------------------------------- le parole
#
# Sono frasi intere e non sostantivi perche' il genere cambia con la parola:
# «ciascun pilastro» diventa «ciascuna categoria», «va alzato anche lui» diventa
# «va alzata anche lei». Sostituire il solo sostantivo produrrebbe italiano
# sbagliato dentro il prompt, che e' esattamente il genere di difetto che questa
# campagna vuole misurare.

_IT_PILASTRO = {
    # llm_arm: glossario
    "glossario_apertura": "Che cosa copre ciascun pilastro:",
    "glossario_fame": "Il pilastro della fame e della sete.",
    # llm_arm: apertura di dominio
    "uno_fra_sei": "uno fra sei pilastri",
    "i_plurale": "i pilastri",
    # llm_arm: aiuto sulla costituzione di riferimento
    "plurale_diversi": "pilastri diversi",
    # llm_arm: vocabolario
    "pesabili": "Leve pesabili",
    # llm_arm: meccanica dei pesi
    "fra_le_plurale": "fra le leve",
    "tre_plurale": "tre leve",
    "una_da_proteggere": "Una leva",
    "va_alzata": "va alzata anche lei",
    # schema JSON (solo il segnaposto, non le chiavi)
    "segnaposto": "leva",
    # administration
    "pesabili_amm": "Pilastri pesabili",
    "una_da_proteggere_amm": "Un pilastro",
    "va_alzata_amm": "va alzato anche lui",
    "segnaposto_amm": "pilastro",
}

_IT_CATEGORIA = {
    "glossario_apertura": "Che cosa copre ciascuna categoria:",
    "glossario_fame": "La categoria della fame e della sete.",
    "uno_fra_sei": "una fra sei categorie di preferenza",
    "i_plurale": "le categorie",
    "plurale_diversi": "categorie diverse",
    "pesabili": "Categorie pesabili",
    "fra_le_plurale": "fra le categorie",
    "tre_plurale": "tre categorie",
    "una_da_proteggere": "Una categoria",
    "va_alzata": "va alzata anche lei",
    "segnaposto": "categoria",
    "pesabili_amm": "Categorie pesabili",
    "una_da_proteggere_amm": "Una categoria",
    "va_alzata_amm": "va alzata anche lei",
    "segnaposto_amm": "categoria",
}

# L'inglese conserva la stessa incoerenza dell'italiano nella variante storica
# --- «pillars» nell'apertura, «levers» nella meccanica dei pesi --- perche' F e'
# la TRADUZIONE del braccio 0 e non una sua correzione: se l'inglese fosse anche
# piu' coerente, il confronto 0 contro F misurerebbe due cose insieme.
_EN_PILASTRO = {
    "glossario_apertura": "What each pillar covers:",
    "glossario_fame": "The pillar of hunger and thirst.",
    "uno_fra_sei": "one of six pillars",
    "i_plurale": "the pillars",
    "plurale_diversi": "different pillars",
    "pesabili": "Weightable levers",
    "fra_le_plurale": "among the levers",
    "tre_plurale": "three levers",
    "una_da_proteggere": "A lever",
    "va_alzata": "must be raised too",
    "segnaposto": "lever",
    "pesabili_amm": "Weightable pillars",
    "una_da_proteggere_amm": "A pillar",
    "va_alzata_amm": "must be raised too",
    "segnaposto_amm": "pillar",
}

_EN_CATEGORIA = {
    "glossario_apertura": "What each category covers:",
    "glossario_fame": "The category of hunger and thirst.",
    "uno_fra_sei": "one of six preference categories",
    "i_plurale": "the categories",
    "plurale_diversi": "different categories",
    "pesabili": "Weightable categories",
    "fra_le_plurale": "among the categories",
    "tre_plurale": "three categories",
    "una_da_proteggere": "A category",
    "va_alzata": "must be raised too",
    "segnaposto": "category",
    "pesabili_amm": "Weightable categories",
    "una_da_proteggere_amm": "A category",
    "va_alzata_amm": "must be raised too",
    "segnaposto_amm": "category",
}

PAROLE: dict[tuple[str, str], dict[str, str]] = {
    ("it", "pilastro"): _IT_PILASTRO,
    ("it", "categoria"): _IT_CATEGORIA,
    ("en", "pilastro"): _EN_PILASTRO,
    ("en", "categoria"): _EN_CATEGORIA,
}

# Le quattro tabelle devono avere le stesse chiavi: una chiave mancante
# lascerebbe un buco nel prompt di un solo braccio.
for _chiave, _tabella in PAROLE.items():
    assert set(_tabella) == set(_IT_PILASTRO), (
        f"la tabella {_chiave} diverge: "
        f"mancano {sorted(set(_IT_PILASTRO) - set(_tabella))}, "
        f"in piu' {sorted(set(_tabella) - set(_IT_PILASTRO))}"
    )


def parole(variante: VariantePrompt) -> dict[str, str]:
    """Le forme da inserire nel prompt per questa variante."""
    try:
        return PAROLE[(variante.lingua, variante.termine)]
    except KeyError:
        raise NotImplementedError(
            f"il prompt in «{variante.lingua}» con il termine «{variante.termine}» "
            "non esiste ancora"
        ) from None


# ------------------------------------------------------------- la gerarchia
#
# Il testo dichiara un fatto del disegno e non suggerisce che cosa farne: dire
# «scrivi leggi piu' generali» misurerebbe l'obbedienza a un'istruzione, non
# l'effetto di sapere.

_IT_GERARCHIA = (
    "\nSotto di te ci sono amministratori di distretto, uno per distretto, e "
    "ciascuno amministra al massimo {celle} celle. Ognuno riceve la tua politica "
    "per intero, vede che cosa produce sulle proprie celle, e sceglie: "
    "applicarla cosi' com'e', oppure sostituirla con una propria, che vale per "
    "le sue celle e solo per quelle. La tua politica e' dunque quella che vale "
    "ovunque un amministratore non decida altrimenti.\n"
)

_EN_GERARCHIA = (
    "\nBelow you there are district administrators, one per district, and "
    "each administers at most {celle} cells. Each of them receives your policy "
    "in full, sees what it produces on their own cells, and chooses: "
    "to apply it as it is, or to replace it with one of their own, valid for "
    "their cells and those only. Your policy is therefore the one that holds "
    "wherever an administrator does not decide otherwise.\n"
)

GERARCHIA: dict[str, str] = {"it": _IT_GERARCHIA, "en": _EN_GERARCHIA}


def testo_gerarchia(variante: VariantePrompt, celle_per_distretto: int) -> str:
    """Il paragrafo che dichiara al governatore l'esistenza del livello locale.

    Vuoto quando la variante non lo prevede, cosi' che il chiamante possa
    concatenarlo senza condizioni.
    """
    if not variante.gerarchia:
        return ""
    try:
        modello = GERARCHIA[variante.lingua]
    except KeyError:
        raise NotImplementedError(
            f"il paragrafo sulla gerarchia in «{variante.lingua}» non esiste ancora"
        ) from None
    return modello.format(celle=celle_per_distretto)
