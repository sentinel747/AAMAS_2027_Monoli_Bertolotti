from __future__ import annotations

"""Il governatore unico, e la sua pianificazione non bloccante.

Nato (2026-08-24) dallo scheletro del consiglio, che questo modulo sostituisce:
via i mandati, via la sintesi mediana/maggioranza, via il gather su N proposte.
Resta tutto cio' che del consiglio era collaudato e non riguardava l'essere in
tanti — i due regimi, la cadenza, i mancati aggiornamenti con allarme, il
registro, il loop asincrono su thread dedicato.

**Il vincolo che ha determinato questo progetto.** Il passo costa ~0,13 s su
configurazione reale con 300 agenti; una chiamata LLM costa 1-5 s. Attendere il
governatore dentro il passo renderebbe la simulazione 10-40 volte piu' lenta.
Quindi la chiamata non sta mai sul percorso critico: `advance` avvia e ritorna,
sempre — a meno del regime bloccante, scelto in configurazione.

**Un solo punto d'ingresso.** `advance(step, picture)` fa tre cose in ordine:
se questo passo e' un confine di tick, promuove la policy del tick precedente
(o conta un mancato aggiornamento e tiene in vigore la precedente) e avvia la
nuova chiamata; poi restituisce la policy in vigore.

**Il passo di applicazione lo decide la configurazione, non la latenza.** La
policy del tick *t* entra in vigore al passo `1 + (t+1) * cadence` (regime non
bloccante) o al confine stesso (regime bloccante). Se dipendesse da quando la
risposta arriva, la riesecuzione dal registro non riprodurrebbe la run.

**Una proposta malformata e' silenzio, non una policy vuota.** `policy=None`
(rete giu', JSON rotto, fuori schema) lascia in vigore la policy precedente:
la politica di un governo non deve svanire per un errore di trascrizione. Una
policy VUOTA (zero regole) e' invece un'adozione esplicita del non-intervento,
e sostituisce la precedente.

**`misses` conta i mancati arrivi E i guasti del fornitore, non le risposte
malformate.** La chiamata che non torna in tempo e quella che torna con un 503
sono lo stesso fatto — nessuna legge e' arrivata — e la seconda non e' meno
grave per il fatto di essere veloce: e' quello il numero che dice se la
configurazione puo' riuscire, e un fornitore che rifiuta sempre e' proprio una
configurazione che non puo' riuscire. Contarne solo il timeout ha lasciato
passare per riuscite diciassette tornate consecutive di 503. Una risposta
MALFORMATA invece non conta: e' il modello che si comporta in modo erratico,
cioe' un dato sul modello, e scartare quelle run selezionerebbe via le
condizioni in cui il modello e' meno stabile. Resta visibile nel registro dalla
proposta senza policy.
"""

import asyncio
import threading

from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Policy


class Governor:
    """Un proponente, una policy in vigore, nessuna attesa non richiesta."""

    def __init__(
        self,
        proposer,
        bounds: Bounds,
        cadence_steps: int,
        recorder=None,
        wait_seconds: float = 0.0,
    ) -> None:
        if cadence_steps < 1:
            raise ValueError("cadence_steps deve essere >= 1")
        if wait_seconds < 0:
            raise ValueError("wait_seconds non puo' essere negativo")
        self._proposer = proposer
        self._bounds = bounds
        self._cadence = int(cadence_steps)
        self._recorder = recorder
        #: `0` = non attendere mai (regime originale). `> 0` = attendere la
        #: deliberazione al confine di tick, fino a questo limite in secondi.
        self._wait_seconds = float(wait_seconds)
        self._in_force: Policy | None = None
        self._pending: asyncio.Future | None = None
        self._pending_tick: int | None = None
        # Il quadro CHE HA PRODOTTO la proposta in volo, non quello del passo in
        # cui viene adottata: sono due tick diversi, e registrare il secondo
        # renderebbe il `picture_digest` del registro incoerente con la proposta
        # che accompagna.
        self._pending_picture: ColonyPicture | None = None
        self.misses = 0
        #: Mancati aggiornamenti CONSECUTIVI, e se l'allarme e' gia' stato dato.
        #: Servono a non far girare in silenzio una run che non puo' riuscire.
        self._consecutive_misses = 0
        self._alarm_given = False

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="governors", daemon=True
        )
        self._thread.start()

    # -- pianificazione, derivata dalla sola configurazione -----------------
    def tick_index(self, step: int) -> int:
        return (int(step) - 1) // self._cadence

    def is_tick_boundary(self, step: int) -> bool:
        return (int(step) - 1) % self._cadence == 0

    @property
    def blocking(self) -> bool:
        return self._wait_seconds > 0.0

    def application_step(self, tick: int) -> int:
        """Il passo in cui la policy del tick `tick` entra in vigore.

        Nel regime bloccante e' il confine che l'ha prodotta, perche' l'attesa
        si conclude dentro quel passo; nell'altro e' il confine successivo,
        perche' la risposta non e' ancora arrivata quando il passo prosegue.
        """
        if self.blocking:
            return 1 + int(tick) * self._cadence
        return 1 + (int(tick) + 1) * self._cadence

    @property
    def in_force(self):
        """La policy attualmente in vigore, senza avviare niente.

        Esiste perche' le shell chiamino `advance` SOLO ai confini di tick:
        fra un confine e l'altro la policy non puo' cambiare, e costruire il
        quadro a ogni passo per poi non usarlo costava lo 0,58% del passo —
        misurato su 300 agenti, 1,29 ms contro i ~0,03 ms della sola quota ai
        confini con cadenza 20.
        """
        return self._in_force

    def set_recorder(self, recorder) -> None:
        """Attacca (o stacca, con `None`) il registro dopo la costruzione.

        Serve perche' i due tempi non coincidono: il governatore nasce insieme
        alla shell, mentre la cartella di output arriva soltanto quando la run
        parte — nella shell GUI addirittura al primo passo.
        """
        self._recorder = recorder

    # -- unico punto d'ingresso per le shell -------------------------------
    def advance(self, step: int, picture: ColonyPicture) -> Policy | None:
        """Promuove, avvia, e restituisce la policy in vigore.

        Non attende, a meno che `wait_seconds > 0`: in quel caso il confine di
        tick attende la deliberazione e la adotta nello stesso passo.
        """
        if self._proposer is None or not self.is_tick_boundary(step):
            return self._in_force

        if self.blocking:
            return self._advance_waiting(step, picture)

        if self._pending is not None:
            if self._pending.done():
                try:
                    proposal = self._pending.result()
                except Exception:  # noqa: BLE001
                    # Un proposer che solleva non deve fermare la simulazione:
                    # vale come mancato aggiornamento, con la policy precedente
                    # ancora in vigore.
                    self._count_miss(step)
                    self._record(step, None, missed=True)
                else:
                    if getattr(proposal, "guasto_fornitore", False):
                        # Arrivata, ma senza niente dentro: il fornitore ha
                        # rifiutato. Si registra la proposta, non `None`, cosi'
                        # il motivo resta leggibile nel registro.
                        self._count_miss(step)
                        self._record(step, proposal, missed=True,
                                     miss_reason=proposal.rationale)
                    else:
                        # La serie si azzera solo qui: senza, tre mancati sparsi
                        # su una run lunga darebbero un allarme falso.
                        self._consecutive_misses = 0
                        self._adopt(proposal)
                        self._record(step, proposal, missed=False)
            else:
                # In ritardo: resta in vigore la precedente. La politica di un
                # governo non deve svanire per latenza di rete.
                self._count_miss(step)
                self._record(step, None, missed=True)
                self._pending.cancel()

        tick = self.tick_index(step)
        self._pending_tick = tick
        self._pending_picture = picture
        self._pending = asyncio.run_coroutine_threadsafe(
            self._proposer.propose(picture, self._bounds), self._loop
        )
        return self._in_force

    def _adopt(self, proposal) -> None:
        """Adotta la policy della proposta, se la proposta ne ha una.

        `None` e' silenzio: la precedente resta in vigore. Una policy vuota e'
        un'adozione esplicita del non-intervento e sostituisce la precedente.
        """
        if proposal is not None and proposal.policy is not None:
            self._in_force = proposal.policy

    def _record(
        self, step: int, proposal, missed: bool,
        miss_reason: str = "", waited_s: float = 0.0,
        picture: ColonyPicture | None = None,
    ) -> None:
        if self._recorder is None:
            return
        self._recorder.record(
            tick=self._pending_tick,
            application_step=step,
            picture=picture or self._pending_picture,
            proposal=proposal,
            policy=self._in_force,
            missed=missed,
            miss_reason=miss_reason,
            waited_s=waited_s,
        )

    def _count_miss(self, step: int) -> None:
        """Conta un mancato aggiornamento e, alla terza volta di fila, allarma.

        Tre di fila non sono sfortuna: sono una configurazione che non puo'
        riuscire. Continuare in silenzio e' il difetto piu' costoso che questo
        sottosistema abbia prodotto — una run vera ha atteso settantacinque
        volte il limite intero e ha consegnato la baseline senza un avviso.
        """
        self.misses += 1
        self._consecutive_misses += 1
        if self._consecutive_misses < 3 or self._alarm_given:
            return
        self._alarm_given = True
        if self.blocking:
            causa = (
                f"le chiamate non tornano entro wait_seconds={self._wait_seconds:.0f}s. "
                "Il provider e' piu' lento del limite: alzare il limite, oppure "
                "abbassare lo sforzo di ragionamento del modello. Verificare con "
                "scripts/check_provider.py CON GLI STESSI parametri della run"
            )
        else:
            causa = (
                f"le chiamate non tornano entro la cadenza ({self._cadence} passi). "
                "Alzare la cadenza, usare wait_seconds > 0, o un modello piu' rapido"
            )
        print(
            f"[governatore] ALLARME al passo {step}: {self._consecutive_misses} "
            f"mancati aggiornamenti consecutivi. {causa}. Finche' dura, la run "
            "coincide con la baseline.",
            flush=True,
        )

    def _advance_waiting(self, step: int, picture: ColonyPicture) -> Policy | None:
        """Il regime bloccante: interroga, attende, adotta, nello stesso passo.

        Il limite di tempo non e' opzionale: senza, un provider che non
        risponde appenderebbe la run per sempre. Allo scadere resta in vigore
        la policy precedente e il mancato aggiornamento viene contato, come
        nell'altro regime — cosi' `misses` significa la stessa cosa in
        entrambi.
        """
        tick = self.tick_index(step)
        self._pending_tick = tick
        self._pending_picture = picture
        future = asyncio.run_coroutine_threadsafe(
            self._proposer.propose(picture, self._bounds), self._loop
        )
        self._pending = future
        try:
            proposal = future.result(timeout=self._wait_seconds)
        except Exception as errore:  # noqa: BLE001 - scadenza, o un proposer che solleva
            future.cancel()
            self._pending = None
            self._count_miss(step)
            # **Il motivo, non solo il fatto.** La chiamata viene annullata
            # mentre attende, quindi il proposer non torna e non lascia ne'
            # latenza ne' errore: senza queste due voci la diagnosi richiede di
            # rileggere la configurazione e indovinare.
            self._record(
                step, None, missed=True,
                miss_reason=f"{type(errore).__name__}: attesa di "
                            f"{self._wait_seconds:.1f}s scaduta o chiamata fallita",
                waited_s=self._wait_seconds,
                picture=picture,
            )
            return self._in_force

        self._pending = None
        if getattr(proposal, "guasto_fornitore", False):
            self._count_miss(step)
            self._record(
                step, proposal, missed=True,
                miss_reason=proposal.rationale,
                waited_s=float(getattr(proposal, "latency_s", 0.0) or 0.0),
                picture=picture,
            )
            return self._in_force
        self._consecutive_misses = 0
        self._adopt(proposal)
        self._record(step, proposal, missed=False, picture=picture)
        return self._in_force

    def close(self) -> None:
        """Ferma il thread del governatore. Idempotente."""
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        if self._loop.is_running():
            # Cancellare i task ancora in volo e ATTENDERLI prima di fermare il
            # loop. Fermarlo e basta li distrugge mentre sono in esecuzione
            # ("Task was destroyed but it is pending"), e con un provider reale
            # quella e' una richiesta HTTP interrotta con la sua connessione mai
            # chiusa.
            async def _drain() -> None:
                current = asyncio.current_task()
                inflight = [task for task in asyncio.all_tasks() if task is not current]
                for task in inflight:
                    task.cancel()
                if inflight:
                    await asyncio.gather(*inflight, return_exceptions=True)

            drain = asyncio.run_coroutine_threadsafe(_drain(), self._loop)
            try:
                # Il limite di tempo evita che una chiusura resti appesa su un
                # provider che non risponde: dopo cinque secondi si procede
                # comunque, perche' una run che non termina e' peggio di una
                # connessione lasciata al sistema operativo.
                drain.result(timeout=5.0)
            except Exception:  # noqa: BLE001 - la chiusura non deve mai sollevare
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._loop.is_closed():
            self._loop.close()
