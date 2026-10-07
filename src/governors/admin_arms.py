# -*- coding: utf-8 -*-
"""I bracci dell'amministratore, uno per ciascun braccio del governo.

Lo strato amministrativo deve esistere in **tutti** i bracci, non nel solo
`llm`, altrimenti il confronto fra centralizzato e decentralizzato misurerebbe
la presenza dello strato invece del modo in cui viene esercitato. Ogni braccio
riceve lo stesso identico input --- la politica del governo gia' risolta sulle
proprie celle, piu' le metriche del proprio distretto --- e ha la stessa
facolta': astenersi, o riscrivere.

    none      si astiene sempre. E' il decentramento a potere nullo, e serve
              come controllo dello strato in se'.
    random    riscrive sempre, con regole arbitrarie entro gli stessi limiti:
              isola l'effetto di *ricevere una direttiva locale*, qualunque
              essa sia.
    scripted  amministratore competente e non linguistico: interviene solo
              quando il problema del suo distretto NON e' quello della colonia,
              che e' esattamente cio' per cui un decentramento esiste.
    llm       chiede a un modello.

**Perche' lo scripted guarda la differenza e non il livello.** Un
amministratore che applicasse una soglia assoluta sarebbe un secondo governo in
piccolo, e la sua utilita' si confonderebbe con quella del primo. Quello che
solo un livello locale puo' vedere e' lo *scarto*: il mio distretto sta peggio
della media in una grandezza che il governo, guardando gli aggregati, non ha
nominato. Interviene li', e altrove tace.

**E perche' lo scarto si misura in sigma e non in percentuale.** Il criterio
non deve solo distinguere il locale dal globale: deve anche poter dire di NO.
Un "sto oltre il 25 per cento sotto la media", preso come massimo su otto
indicatori, e' vero quasi sempre per costruzione --- misurato: 98,7 per cento
delle tornate. Vedi `SIGMA_MINIME`.
"""

from __future__ import annotations

import json
import random
import time

from src.governors.arms import _RANDOM_THRESHOLDS, _seed_for
from src.governors.observation import ColonyPicture
from src.governors.policy import (
    INDICATORS,
    PILLAR_BY_NAME,
    Bounds,
    Condition,
    Policy,
    Rule,
)

#: Quale pilastro risponde a quale carenza. E' la stessa lettura che la
#: costituzione di riferimento fa, applicata al distretto.
_RISPOSTA = {
    "food_per_occupant": "sustenance",
    "water_per_occupant": "sustenance",
    "oxygen_per_occupant": "build",
    "ice_per_occupant": "sustenance",
    "material_per_occupant": "resources",
    "minerals_per_occupant": "explore",
    "power_coverage": "build",
    "structure_integrity": "build",
}

#: Quante deviazioni standard sotto la media di colonia rendono un distretto
#: un caso ANOMALO invece che semplicemente sotto la media.
#:
#: **Perche' non e' piu' una percentuale (2026-09-02).** La versione precedente
#: chiedeva uno scarto relativo di almeno il 25 per cento dalla media di
#: colonia, e sui dati veri scattava nel **98,7 per cento** delle tornate: un
#: braccio che dice sempre di si' e' inerte quanto uno che dice sempre di no, e
#: in entrambi i casi il decentramento non e' piu' attribuibile a una decisione.
#:
#: La degenerazione era STRUTTURALE e non una questione di taratura. Il
#: criterio prendeva il MASSIMO dello scarto su otto indicatori: la probabilita'
#: che almeno uno di otto sia oltre il 25 per cento sotto la media e' prossima a
#: uno, qualunque sia la soglia ragionevole. Alzare il 25 per cento avrebbe
#: spostato il numero senza togliere il difetto --- e' la stessa **costante
#: travestita da condizione** gia' trovata nella costituzione del governatore.
#:
#: Il criterio nuovo e' un test di anomalia e non di livello: il quadro porta
#: gia' la deviazione standard degli indicatori sull'intera colonia, quindi
#: "sto piu' di una sigma sotto" e' adimensionale, non dipende dall'unita'
#: dell'indicatore e non e' vero quasi ovunque per costruzione. Una sigma e' la
#: soglia convenzionale per "fuori dall'ordinario", non un valore scelto per
#: ottenere un tasso desiderato: il tasso che ne risulta si misura e si riporta
#: qualunque esso sia.
SIGMA_MINIME = 1.0


class _SenzaAttesa:
    """Chi non chiama nessuno risponde subito, e resta nello stesso protocollo.

    La tornata di distretto e' **concorrente per progetto**: gli amministratori
    deliberano tutti sulla stessa politica dello stesso tick, quindi non c'e'
    nulla che uno debba sapere da un altro, e metterli in fila moltiplicherebbe
    la latenza per il numero di distretti. `Amministrazione` li interroga
    percio' attraverso `propose_text_async`. Un braccio che non fa I/O non ha
    niente da attendere: eredita questa riga e la tornata resta omogenea, senza
    che il chiamante debba distinguere fra bracci lenti e bracci istantanei.
    """

    async def propose_text_async(self, prompt: str) -> dict:
        return self.propose_text(prompt)


class AmministratoreAssente(_SenzaAttesa):
    """Si astiene sempre: le celle restano al governo."""

    def propose_text(self, prompt: str) -> dict:
        del prompt
        return {"raw": {"accept": True, "rationale": "nessun potere amministrativo"}}


class AmministratoreCasuale(_SenzaAttesa):
    """Riscrive sempre, con regole arbitrarie entro i limiti."""

    def __init__(self, seed: int, distretto: int, bounds: Bounds) -> None:
        self._seed = int(seed) * 1000 + int(distretto)
        self._bounds = bounds
        self._chiamate = 0

    def propose_text(self, prompt: str) -> dict:
        del prompt
        self._chiamate += 1
        rng = random.Random(_seed_for(self._seed, self._chiamate))
        nomi = sorted(INDICATORS)
        pilastri = sorted(PILLAR_BY_NAME)
        regole = []
        for _ in range(rng.randint(1, 2)):
            indicatore = nomi[rng.randrange(len(nomi))]
            basso, alto = _RANDOM_THRESHOLDS[indicatore]
            regole.append(
                {
                    "if": {
                        "indicator": indicatore,
                        "op": rng.choice(("<", ">")),
                        "value": rng.uniform(basso, alto),
                    },
                    "weights": {
                        pilastri[rng.randrange(len(pilastri))]: rng.uniform(
                            self._bounds.weight_min, self._bounds.weight_max
                        )
                    },
                }
            )
        return {
            "raw": {
                "accept": False,
                "policy": regole,
                "rationale": f"amministrazione casuale, seme {self._seed}",
            }
        }


class AmministratoreScritto(_SenzaAttesa):
    """Interviene dove il distretto sta peggio della colonia, e solo li'."""

    def __init__(self, bounds: Bounds) -> None:
        self._bounds = bounds
        self.quadro: ColonyPicture | None = None

    def propose_text(self, prompt: str) -> dict:
        # Il prompt porta gia' i due quadri in JSON: si leggono da li' invece di
        # aggiungere un secondo canale, cosi' il braccio scritto vede
        # ESATTAMENTE cio' che vede quello linguistico. Se la lettura fallisce
        # l'amministratore si astiene, che e' l'esito sicuro.
        try:
            distretto = _estrai_json(prompt, "Distribuzione degli indicatori sulle TUE celle")
            colonia = _estrai_json(prompt, "Gli stessi indicatori sull'INTERA colonia")
        except (ValueError, json.JSONDecodeError):
            return {"raw": {"accept": True, "rationale": "quadro illeggibile"}}

        # **Il confronto e' con il paese, non con se stessi (2026-09-01).** La
        # prima versione guardava lo scarto interno `(media - minimo) / media`,
        # e su una qualunque cella a zero quello vale 1,0: misurato sulla run di
        # prova, 99 interventi contro 2 astensioni, cioe' un amministratore che
        # interviene sempre e non discrimina nulla. Un braccio di controllo che
        # dice sempre di si' e' inerte quanto uno che dice sempre di no: in
        # entrambi i casi il decentramento non e' piu' attribuibile a una
        # decisione. Quello che solo un livello locale puo' vedere e' che il
        # PROPRIO distretto stia peggio del paese in una grandezza che il
        # governo, guardando gli aggregati, non ha nominato.
        peggiore = None
        divario_peggiore = 0.0
        for nome, statistiche in (distretto or {}).items():
            if nome not in _RISPOSTA or not isinstance(statistiche, dict):
                continue
            riferimento = colonia.get(nome) if isinstance(colonia, dict) else None
            if not isinstance(riferimento, dict):
                continue
            sigma = float(riferimento.get("std", 0.0))
            if sigma <= 0:
                # Un indicatore uguale in tutta la colonia non ha code: nessun
                # distretto puo' esserne un caso anomalo, e dividere per zero
                # renderebbe anomalo il piu' piccolo scarto numerico.
                continue
            media_nazionale = float(riferimento.get("mean", 0.0))
            divario = (media_nazionale - float(statistiche.get("mean", 0.0))) / sigma
            if divario > divario_peggiore:
                divario_peggiore, peggiore = divario, nome

        if peggiore is None or divario_peggiore < SIGMA_MINIME:
            return {
                "raw": {
                    "accept": True,
                    "rationale": (
                        "nessun indicatore piu' di "
                        f"{SIGMA_MINIME:g} sigma sotto la media di colonia: "
                        "il distretto non e' un caso anomalo e la politica del "
                        "governo basta"
                    ),
                }
            }

        # **La soglia e' la media di colonia, non il massimo del distretto
        # (2026-09-05).** Con il massimo e `<` stretto, in un distretto di una
        # cella sola la regola era impossibile per costruzione --- e la meta'
        # dei distretti ha una cella. La riscrittura sostituiva la catena del
        # governo con una regola che non scattava mai, e la cella restava SENZA
        # alcuna politica: misurato sulla campagna del 02/09, dal 19 al 33 per
        # cento delle riscritture. La cella sotto la media nazionale e'
        # esattamente quella che ha reso il distretto anomalo: e' li' che la
        # regola deve mordere.
        soglia = float(colonia[peggiore].get("mean", 0.0))
        pilastro = _RISPOSTA[peggiore]
        peso = min(3.0, self._bounds.weight_max)
        return {
            "raw": {
                "accept": False,
                "policy": [
                    {
                        "if": {"indicator": peggiore, "op": "<", "value": soglia},
                        "weights": {pilastro: peso},
                    }
                ],
                "rationale": (
                    f"il distretto sta {divario_peggiore:.2f} sigma sotto la "
                    f"media di colonia in {peggiore}: intervengo sotto "
                    f"{soglia:.2f} con {pilastro} x{peso:g}"
                ),
            }
        }


class AmministratoreLLM:
    """Chiede a un modello, con la stessa garanzia di non sollevare mai."""

    def __init__(self, provider, model: str = "", cost_tracker=None) -> None:
        self._provider = provider
        self._model = model or getattr(provider, "model", "")
        self._cost_tracker = cost_tracker

    async def propose_text_async(self, prompt: str) -> dict:
        """Il percorso che la run usa davvero.

        E' asincrono perche' la tornata di distretto e' concorrente: le
        chiamate partono tutte insieme e il tick costa una latenza sola invece
        di una per amministratore. Il percorso sincrono qui sotto resta per la
        diagnosi a mano e per chi non ha un loop.
        """
        provider_id = getattr(self._provider, "provider_id", "sconosciuto")
        inizio = time.perf_counter()
        try:
            risposta = await self._provider.async_complete_json(prompt)
        except Exception as errore:  # noqa: BLE001
            return self._guasto(provider_id, errore, inizio)
        return self._esito(risposta, provider_id, inizio)

    def propose_text(self, prompt: str) -> dict:
        provider_id = getattr(self._provider, "provider_id", "sconosciuto")
        inizio = time.perf_counter()
        try:
            risposta = self._provider.complete_json(prompt)
        except Exception as errore:  # noqa: BLE001
            return self._guasto(provider_id, errore, inizio)
        return self._esito(risposta, provider_id, inizio)

    def _guasto(self, provider_id: str, errore, inizio: float) -> dict:
        """Un guasto del fornitore non solleva mai, e non e' una decisione.

        **`failed` e' il marcatore che lo strato legge.** Prima qui c'era
        `{"accept": True}`, e un endpoint spento entrava nel registro come
        un'accettazione: l'11 settembre 2026 un modello della campagna mista
        ha avuto 118 chiamate fallite su 118 ed e' risultato l'unico che non
        riscrive mai. Un guasto e' una condizione dell'infrastruttura e va
        contato a parte, come `malformata` lo e' per una risposta non capita.
        """
        return {
            "raw": None,
            "failed": True,
            "rationale": f"chiamata fallita: {errore}",
            "provider": provider_id,
            "model": self._model,
            "latency_s": time.perf_counter() - inizio,
        }

    def _esito(self, risposta, provider_id: str, inizio: float) -> dict:
        """La lettura della risposta, comune ai due percorsi.

        Scritta una volta sola di proposito: se contabilita' e interpretazione
        vivessero in due copie, il percorso concorrente potrebbe smettere di
        registrare i costi senza che nulla lo segnali.
        """
        latenza = time.perf_counter() - inizio
        if self._cost_tracker is not None:
            try:
                self._cost_tracker.record(
                    cost_usd=float(getattr(risposta, "cost_usd", 0.0) or 0.0),
                    success=not getattr(risposta, "error", ""),
                    provider=provider_id,
                    model=self._model,
                    tokens_in=int(getattr(risposta, "tokens_in", 0) or 0),
                    tokens_out=int(getattr(risposta, "tokens_out", 0) or 0),
                )
            except Exception:  # noqa: BLE001 - la contabilita' non ferma la run
                pass
        errore = getattr(risposta, "error", "")
        if errore:
            # Il fornitore non solleva: quando la chiamata fallisce restituisce
            # il payload di ripiego pensato per la decisione di un COLONO, che
            # e' un JSON valido, non ha la chiave `accept` e verrebbe letto
            # come accettazione. Va intercettato qui, prima del parser.
            return self._guasto(provider_id, errore, inizio)
        try:
            grezza = json.loads(getattr(risposta, "text", None))
        except (TypeError, ValueError):
            grezza = {"accept": True, "rationale": "risposta non interpretabile"}
        return {
            "raw": grezza,
            "provider": provider_id,
            "model": self._model,
            "tokens_in": int(getattr(risposta, "tokens_in", 0) or 0),
            "tokens_out": int(getattr(risposta, "tokens_out", 0) or 0),
            "latency_s": latenza,
        }


def _estrai_json(testo: str, dopo: str) -> dict:
    """Il primo oggetto JSON che segue un'intestazione nel prompt."""
    inizio = testo.index(dopo)
    apertura = testo.index("{", inizio)
    profondita, indice = 0, apertura
    while indice < len(testo):
        if testo[indice] == "{":
            profondita += 1
        elif testo[indice] == "}":
            profondita -= 1
            if profondita == 0:
                return json.loads(testo[apertura : indice + 1])
        indice += 1
    raise ValueError("oggetto JSON non chiuso")
