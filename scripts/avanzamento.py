# -*- coding: utf-8 -*-
"""La barra di avanzamento della coda: che cosa gira, quanto manca, che cosa si e' rotto.

**Perche' esiste.** Una coda lanciata con `nohup` scrive su un file di log che
cresce di migliaia di righe e che non si puo' guardare: chi ha lanciato la
campagna non sa se manca un'ora o quindici, e l'unico modo di scoprirlo e'
leggere a mano l'ultima riga giusta fra molte sbagliate. Qui la stessa
informazione sta in poche righe che si riscrivono da sole.

**Mostra anche i guasti, e sono la ragione principale per guardarla.** Una run
che perde tornate del governatore o che ha chiamate fallite finisce lo stesso,
scrive i suoi artifact e ha l'aria di tutte le altre: nessun errore, nessun
messaggio. Se non si guarda il registro, una campagna di venti ore puo'
riempirsi di righe inutilizzabili e accorgersene solo all'analisi. La coda si
ferma da sola alla prima (`scripts/coda.sh`), e qui si vede perche'.

**Non tocca niente.** Legge il piano, i registri e il log in sola lettura. Si
puo' avviare e fermare in qualunque momento: la coda non sa che esiste.

**Non si interrompe da solo.** Un file che non c'e' ancora, un log troncato a
meta' riga, un registro scritto mentre lo si legge: nessuno di questi casi ferma
la barra, che mostra cio' che sa e riprova. Si esce con ctrl-c.

Uso:
    python scripts/avanzamento.py
    python scripts/avanzamento.py --piano runs/piano.txt --log runs/coda.log
    python scripts/avanzamento.py --una-volta
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from scripts.qualita_run import guasti  # noqa: E402

#: L'ultima riga di avanzamento che il runner stampa dentro una run.
_PASSO = re.compile(r"step=(\d+)/(\d+)")
#: La riga con cui una coda annuncia il lavoro che sta per cominciare.
_ANNUNCIO = re.compile(r"^\[[a-z_]+\]\s+\S+\s+(.*)$")
#: Le righe di servizio: annunciano lo stato della coda, non un lavoro.
_SERVIZIO = ("piano scritto", "aspetto", "macchina libera", "si parte",
             "gia' fatta", "la macchina e'", "lucchetto", "in coda dietro")
#: Quanto dura una run quando non c'e' ancora niente da cui misurarlo.
_DURATA_ATTESA_S = 55 * 60


def _leggi_coda(percorso: Path, quanti_byte: int = 200_000) -> list[str]:
    """Le ultime righe del log, senza caricarlo tutto e senza mai sollevare."""
    try:
        with open(percorso, "rb") as f:
            f.seek(0, io.SEEK_END)
            f.seek(max(0, f.tell() - quanti_byte))
            grezzo = f.read()
    except OSError:
        return []
    return grezzo.decode("utf-8", errors="replace").splitlines()


def _righe_registro(percorso: Path) -> list[dict]:
    """Le run gia' finite, saltando la riga che si sta scrivendo adesso."""
    fatte = []
    try:
        for riga in io.open(percorso, encoding="utf-8", errors="replace"):
            riga = riga.strip()
            if not riga:
                continue
            try:
                fatte.append(json.loads(riga))
            except json.JSONDecodeError:
                continue  # l'ultima riga puo' essere a meta': sara' li' fra un attimo
    except OSError:
        pass
    return fatte


def _piano(percorso: Path) -> list[tuple[str, str, Path]]:
    """Le voci del piano: (nome, seme, cartella).

    Legge sia il formato nuovo, separato da tabulazioni e con un nome leggibile,
    sia quello vecchio a tre campi separati da spazi: una coda gia' in corso
    continua a essere seguita mentre la si sostituisce.
    """
    voci = []
    try:
        for riga in io.open(percorso, encoding="utf-8", errors="replace"):
            if not riga.strip():
                continue
            if "\t" in riga:
                pezzi = riga.rstrip("\n").split("\t")
                if len(pezzi) >= 3:
                    voci.append((pezzi[0], pezzi[2], RADICE / pezzi[1]))
            else:
                pezzi = riga.split()
                if len(pezzi) >= 3:
                    voci.append((pezzi[0], pezzi[1], RADICE / pezzi[2]))
    except OSError:
        pass
    return voci


def _cartelle(voci) -> list[Path]:
    """Le cartelle distinte del piano, nell'ordine in cui compaiono."""
    viste, fuori = set(), []
    for _, _, cartella in voci:
        if cartella not in viste:
            viste.add(cartella)
            fuori.append(cartella)
    return fuori


def _durate_per_tipo(voci) -> tuple[float, float]:
    """Quanto dura una baseline e quanto una run con LLM, in questa campagna.

    **Separate, perche' non durano lo stesso.** Diciassette minuti contro
    settanta: una media sola non descrive nessuna delle due, e siccome le
    baseline stanno in testa al piano quella media scenderebbe proprio mentre
    il lavoro rimasto diventa tutto del tipo lento.
    """
    senza, con = [], []
    for cartella in _cartelle(voci):
        for r in _righe_registro(cartella / "results.jsonl"):
            try:
                durata = float(r["wall_clock_s"])
            except (KeyError, TypeError, ValueError):
                continue
            (senza if r.get("arm") == "none" else con).append(durata)
    media_senza = sum(senza) / len(senza) if senza else _DURATA_ATTESA_S
    # Finche' non e' finita una run con LLM non si sa quanto duri: meglio la
    # baseline misurata dell'attesa a occhio, ma senza spacciarla per uguale.
    media_con = sum(con) / len(con) if con else max(media_senza, _DURATA_ATTESA_S)
    return media_senza, media_con


def _lavoratori(piano_file: Path) -> int:
    """Quante run girano insieme: tanti quanti i pezzi in cui il piano e' diviso.

    La coda con piu' lavoratori spezza il piano in `piano.1`, `piano.2`, ... e
    ne da' uno a ciascuno. Contarli e' il modo di saperlo dal disco, senza che
    il monitor debba farsi passare la configurazione della coda.
    """
    try:
        pezzi = [p for p in piano_file.parent.glob(piano_file.name + ".*")
                 if p.suffix[1:].isdigit()]
    except OSError:
        return 1
    return max(1, len(pezzi))


def _saltate(piano_file: Path) -> int:
    """Quante run la coda ha abbandonato dopo due tentativi falliti.

    Non arriveranno mai: lasciarle nel totale terrebbe la barra sotto il fondo
    per sempre e farebbe contare, nel tempo rimanente, lavoro che nessuno fara'.
    """
    # **L'etichetta, non il nome del file.** La coda chiama l'elenco
    # `saltate_<etichetta>.txt`, e l'etichetta e' cio' che resta del nome del
    # piano tolto il prefisso e l'eventuale `recupero_`. Ricavarla cosi' fa
    # funzionare il conteggio anche mentre si sta recuperando, quando il piano
    # si chiama `piano_recupero_<etichetta>.txt`.
    radice = piano_file.stem
    for prefisso in ("piano_recupero_", "piano_"):
        if radice.startswith(prefisso):
            radice = radice[len(prefisso):]
            break
    else:
        # Il piano non segue la convenzione: non si sa a quale elenco
        # corrisponda. Zero e' l'unica risposta onesta --- leggere il piano
        # stesso, com'e' successo, ne conta le righe come run abbandonate.
        return 0
    try:
        testo = io.open(piano_file.parent / f"saltate_{radice}.txt",
                        encoding="utf-8", errors="replace").read()
    except OSError:
        return 0
    return sum(1 for r in testo.splitlines() if r.strip())


def _fatte(voci) -> set[tuple[str, str]]:
    """Quali coppie (nome, seme) hanno gia' un registro."""
    per_cartella: dict[Path, set[str]] = {}
    fatte = set()
    for nome, seme, cartella in voci:
        if cartella not in per_cartella:
            per_cartella[cartella] = {
                str(r.get("seed")) for r in _righe_registro(cartella / "results.jsonl")
            }
        if seme in per_cartella[cartella]:
            fatte.add((nome, seme))
    return fatte


def _cadenza_e_passi(voci) -> tuple[int, int]:
    for cartella in _cartelle(voci):
        for r in _righe_registro(cartella / "results.jsonl"):
            try:
                return int(r["config"]["cadence_steps"]), int(r["steps"])
            except (KeyError, TypeError, ValueError):
                continue
    return 25, 1000


def _passo_dai_registri(voci) -> tuple[int, int, str]:
    """A che passo sta la run in corso, e quale run e'.

    **Perche' non si legge dal log.** Il runner stampa l'avanzamento ogni
    `--log-interval` passi, che in queste campagne vale 250: una barra che si
    muove quattro volte in un'ora non e' una barra. Il governatore invece scrive
    una riga in `governor_decisions.jsonl` a ogni tornata, cioe' ogni 25 passi,
    e quel file cresce mentre la run gira.

    **E nemmeno il NOME va letto dal log.** Prima lo era, e bastava che la run
    in corso fosse stata avviata da un'altra coda --- o dalla stessa prima di un
    riavvio --- perche' la barra dicesse «in attesa che la macchina si liberi»
    mentre una simulazione stava girando. Il registro che cresce dice tutto:
    quale cartella, quale seme, a che punto. Il log resta una comodita', non la
    fonte.
    """
    candidati = []
    for cartella in _cartelle(voci):
        try:
            for sotto in cartella.iterdir():
                registro = sotto / "governor_decisions.jsonl"
                if registro.is_file():
                    candidati.append(registro)
        except OSError:
            continue
    if not candidati:
        return 0, 0, ""
    try:
        piu_recente = max(candidati, key=lambda p: p.stat().st_mtime)
        # Una run finita ha il registro fermo: se l'ultima scrittura e' vecchia,
        # non c'e' niente in corso da mostrare.
        if time.time() - piu_recente.stat().st_mtime > 900:
            return 0, 0, ""
        tornate = sum(1 for r in io.open(piu_recente, encoding="utf-8", errors="replace")
                      if r.strip())
    except OSError:
        return 0, 0, ""
    cadenza, passi = _cadenza_e_passi(voci)

    # Il nome: la cartella del braccio e il seme, ripescati dal piano.
    cartella = piu_recente.parent.parent
    seme = ""
    coda_nome = piu_recente.parent.name
    if "seed" in coda_nome:
        seme = coda_nome.rsplit("seed", 1)[-1]
    nome = next((n for n, s, c in voci if c == cartella and s == seme), "")
    etichetta = f"{nome} seme {seme}" if nome else f"{cartella.name} seme {seme}"
    return min(passi, tornate * cadenza), passi, etichetta


def _pezzi_di_piano(piano_file: Path) -> dict:
    """Da (cartella, seme) al numero del lavoratore che ha quella riga.

    La coda con piu' lavoratori spezza il piano per BRACCIO in `piano.1`,
    `piano.2`, ...: ogni lavoratore possiede cartelle intere. Rileggere quei
    pezzi e' il solo modo di dire a chi appartiene una run in corso --- dal
    registro sul disco non si vede chi lo sta scrivendo.
    """
    mappa = {}
    for n in range(1, 17):
        pezzo = piano_file.with_name(piano_file.name + f".{n}")
        if not pezzo.exists():
            continue
        for nome, seme, cartella in _piano(pezzo):
            mappa[(cartella, seme)] = n
    return mappa


def _passo_di_una_run(sotto: Path, adesso: float, cadenza: int):
    """Il passo di una run viva, o `None` se quella cartella non sta girando.

    **Due registri, non uno.** Il braccio `solo amministratori` gira con
    `--arms none` e quindi non scrive `governor_decisions.jsonl`: cercando solo
    quello, una run in corso risultava inesistente e la barra diceva «nessuna
    simulazione in corso» mentre ne stava girando una.

    Il registro degli amministratori si preferisce anche quando ci sono
    entrambi, perche' ogni sua riga porta `step`, cioe' il passo esatto. Dal
    registro del governatore il passo si puo' solo stimare come «tornate per
    cadenza», ed e' la stima a dover fare un passo indietro quando c'e' la
    misura.

    Viva significa scritta negli ultimi 900 secondi: una run finita ha il
    registro fermo.
    """
    amministratori = sotto / "administrator_decisions.jsonl"
    try:
        if amministratori.is_file() and adesso - amministratori.stat().st_mtime <= 900:
            ultimo = 0
            for riga in io.open(amministratori, encoding="utf-8", errors="replace"):
                riga = riga.strip()
                if not riga:
                    continue
                try:
                    passo = json.loads(riga).get("step")
                except (ValueError, AttributeError):
                    continue
                if isinstance(passo, int):
                    ultimo = max(ultimo, passo)
            return ultimo
    except OSError:
        pass

    governatore = sotto / "governor_decisions.jsonl"
    try:
        if governatore.is_file() and adesso - governatore.stat().st_mtime <= 900:
            tornate = sum(1 for r in io.open(governatore, encoding="utf-8",
                                             errors="replace") if r.strip())
            return tornate * cadenza
    except OSError:
        pass
    return None


def _in_volo(voci, piano_file: Path) -> list[tuple]:
    """Ogni run che sta davvero girando: (lavoratore, nome, passo, totale).

    Stessa definizione di «in corso» usata da `_passo_dai_registri` --- un
    registro del governatore toccato negli ultimi 900 secondi --- ma restituite
    TUTTE invece che solo la piu' recente. Con due lavoratori le run in volo
    sono due, e mostrarne una sola faceva sembrare ferma meta' della coda.
    """
    di_chi = _pezzi_di_piano(piano_file)
    cadenza, passi = _cadenza_e_passi(voci)
    adesso = time.time()
    fuori = []
    for cartella in _cartelle(voci):
        try:
            sotto_cartelle = list(cartella.iterdir())
        except OSError:
            continue
        for sotto in sotto_cartelle:
            passo_ora = _passo_di_una_run(sotto, adesso, cadenza)
            if passo_ora is None:
                continue
            seme = sotto.name.rsplit("seed", 1)[-1] if "seed" in sotto.name else ""
            nome = next((n for n, sm, c in voci if c == cartella and sm == seme), "")
            etichetta = f"{nome} seme {seme}" if nome else f"{cartella.name} seme {seme}"
            fuori.append((di_chi.get((cartella, seme), 0), etichetta,
                          min(passi, passo_ora), passi))
    fuori.sort()
    return fuori


def _in_corso(righe: list[str]) -> tuple[str, int, int]:
    annuncio, passo, totale = "", 0, 0
    for riga in righe:
        m = _ANNUNCIO.match(riga.strip())
        if m:
            testo = m.group(1)
            if not testo.startswith(_SERVIZIO):
                annuncio, passo, totale = testo, 0, 0
        m = _PASSO.search(riga)
        if m:
            passo, totale = int(m.group(1)), int(m.group(2))
    return annuncio, passo, totale


def _coda_ferma(righe: list[str]) -> str:
    """La coda si e' fermata da sola? Restituisce il motivo, o stringa vuota."""
    for riga in reversed(righe[-60:]):
        if "FERMO LA CODA" in riga:
            return riga.split("]", 1)[-1].strip()
    return ""


def _barra(quota: float, larghezza: int = 34) -> str:
    quota = min(1.0, max(0.0, quota))
    pieni = int(round(quota * larghezza))
    return "#" * pieni + "." * (larghezza - pieni)


def _durata(secondi: float) -> str:
    if secondi <= 0:
        return "--"
    ore, resto = divmod(int(secondi), 3600)
    minuti = resto // 60
    return f"{ore}h{minuti:02d}m" if ore else f"{minuti}m"


def _in_corso_dal_piano(voci, fatte) -> str:
    """La prima riga non fatta, se il disco dice che qualcuno la sta scrivendo.

    Serve ai bracci senza governatore --- la baseline non tiene registro delle
    decisioni --- e a ogni run con lo strato ambientale acceso, dove l'annuncio
    della coda finisce sepolto sotto le righe del Mars Climate Database.
    """
    for nome, seme, cartella in voci:
        if (nome, seme) in fatte:
            continue
        if not cartella.exists():
            return ""
        fresco = 0.0
        for p in cartella.rglob("*"):
            try:
                fresco = max(fresco, p.stat().st_mtime)
            except OSError:
                continue
        if fresco and (time.time() - fresco) < 180:
            return f"{nome} seme {seme}"
        return ""
    return ""


def _quadro(piano_file: Path, log_file: Path) -> list[str]:
    voci = _piano(piano_file)
    righe_log = _leggi_coda(log_file)
    annuncio, passo, totale_passi = _in_corso(righe_log)

    if not voci:
        return [
            "nessun piano in " + str(piano_file),
            "la coda non e' ancora partita.",
            (f"ultima riga del log: {righe_log[-1][:90]}" if righe_log else "log vuoto."),
        ]

    fatte = _fatte(voci)
    totale, finite = len(voci), len(fatte)
    media_senza, media_con = _durate_per_tipo(voci)
    lavoratori = _lavoratori(piano_file)
    saltate = _saltate(piano_file)

    # Il registro che cresce batte il log: dice che cosa gira davvero, anche se
    # a farlo girare e' stata un'altra coda.
    # **Un annuncio che non e' nel piano non e' un annuncio.** Le righe con cui
    # la coda spiega perche' si e' fermata hanno la stessa forma di un annuncio,
    # e una di loro e' finita nella barra come nome della run in corso.
    nomi = {n for n, _, _ in voci}
    if annuncio and not any(annuncio.startswith(n) for n in nomi):
        annuncio = ""

    passo_fine, totale_fine, nome_fine = _passo_dai_registri(voci)
    if totale_fine:
        passo, totale_passi = passo_fine, totale_fine
        annuncio = nome_fine or annuncio
    elif not annuncio:
        annuncio = _in_corso_dal_piano(voci, fatte)

    quota_run = (passo / totale_passi) if totale_passi else 0.0
    # **Il lavoro rimasto si somma riga per riga, non a colpi di media.** Ogni
    # riga del piano dice gia' se e' una baseline o una run con LLM, e le due
    # durano in modo troppo diverso per essere confuse.
    lavoro = 0.0
    for nome, seme, _ in voci:
        if (nome, seme) in fatte:
            continue
        lavoro += media_senza if nome == "baseline" else media_con
    # **Le saltate restano nel conto.** Vanno rifatte --- sono state abbandonate
    # per un guasto, non perche' non servano --- quindi il loro tempo e' lavoro
    # che manca ancora.
    # Diviso i lavoratori: girano insieme, non in fila. Non si scala per il
    # pezzo gia' fatto dalle run in volo --- vale meno di una run per lavoratore
    # e non cambia una stima che si misura in ore.
    manca = max(0.0, lavoro) / lavoratori
    # Il denominatore e' il piano intero: una run saltata e' un buco, e la barra
    # deve restare indietro finche' il buco c'e'.
    quota = min(1.0, finite / max(1, totale))

    coda_riga = (f"coda   {_barra(quota)}  {finite}/{totale} run  {quota:6.1%}   "
                 f"mancano ~{_durata(manca)}")
    if lavoratori > 1:
        coda_riga += f"   ({lavoratori} insieme)"
    if saltate:
        coda_riga += f"   {saltate} DA RIFARE"
    righe = [coda_riga]
    # **Con piu' di una run in volo si mostrano tutte.** La riga «ora:» nasce
    # dal registro scritto piu' di recente: con due lavoratori saltava da una
    # run all'altra a ogni aggiornamento e l'altra meta' della coda sembrava
    # ferma. Con un lavoratore solo resta la riga di sempre.
    in_volo = _in_volo(voci, piano_file) if lavoratori > 1 else []
    if finite >= totale:
        righe.append("ora: piano completo.")
    elif len(in_volo) > 1:
        for quale, etichetta, p, tot in in_volo:
            capo = f"L{quale}" if quale else "  "
            q = (p / tot) if tot else 0.0
            righe.append(f"{capo}: {etichetta[:38]:<38} {_barra(q, 22)}  "
                         f"{p}/{tot} passi  {q:5.1%}")
    elif in_volo:
        quale, etichetta, p, tot = in_volo[0]
        q = (p / tot) if tot else 0.0
        righe.append(f"L{quale}: {etichetta[:38]:<38} {_barra(q, 22)}  "
                     f"{p}/{tot} passi  {q:5.1%}")
        for altro in range(1, lavoratori + 1):
            if altro != quale:
                righe.append(f"L{altro}: {'(nessuna run in volo)':<38}")
    elif annuncio and totale_passi:
        righe.append(f"ora: {annuncio[:40]:<40} {_barra(quota_run, 22)}  "
                     f"{passo}/{totale_passi} passi  {quota_run:5.1%}")
    elif annuncio:
        righe.append(f"ora: {annuncio[:40]:<40} (avvio...)")
    else:
        righe.append("ora: nessuna simulazione in corso; la coda aspetta.")

    # I blocchi del piano, nell'ordine in cui li esegue.
    ordine, conta = [], {}
    for nome, seme, _ in voci:
        if nome not in conta:
            conta[nome] = [0, 0]
            ordine.append(nome)
        conta[nome][1] += 1
        if (nome, seme) in fatte:
            conta[nome][0] += 1
    # A capo invece di un taglio: una riga troncata a meta' parola nasconde
    # proprio i blocchi in fondo, che sono quelli non ancora cominciati.
    pezzi = [f"{n}:{conta[n][0]}/{conta[n][1]}" for n in ordine]
    riga_corrente, prefisso = "", "blocchi: "
    for pezzo in pezzi:
        if len(prefisso) + len(riga_corrente) + len(pezzo) + 3 > 150:
            righe.append(prefisso + riga_corrente.rstrip())
            prefisso, riga_corrente = "         ", ""
        riga_corrente += pezzo + "   "
    if riga_corrente.strip():
        righe.append(prefisso + riga_corrente.rstrip())

    # I guasti. E' la riga per cui vale la pena tenere aperta la finestra.
    problemi = []
    for cartella in _cartelle(voci):
        problemi.extend(f"{cartella.name}: {p}" for p in guasti(cartella))
    fermo = _coda_ferma(righe_log)
    if problemi:
        righe.append(f"!! {len(problemi)} RUN NON COMPARABILI:")
        for p in problemi[:4]:
            righe.append(f"   - {p[:140]}")
        if len(problemi) > 4:
            righe.append(f"   ... e altre {len(problemi) - 4}")
    elif fermo:
        righe.append(f"!! coda ferma: {fermo[:140]}")
    else:
        righe.append("qualita': nessuna tornata persa, nessuna chiamata fallita.")

    righe.append(
        f"aggiornato {datetime.now().strftime('%H:%M:%S')}   "
        f"per run: baseline {_durata(media_senza)}, con LLM "
        f"{_durata(media_con)}   ctrl-c per uscire"
    )
    return righe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piano", default="runs/piano.txt")
    parser.add_argument("--log", default="runs/coda.log")
    parser.add_argument("--intervallo", type=float, default=10.0,
                        help="secondi fra due aggiornamenti (default 10)")
    parser.add_argument("--una-volta", action="store_true",
                        help="stampa una volta sola ed esce, per gli script")
    args = parser.parse_args()

    piano_file = (RADICE / args.piano).resolve()
    log_file = (RADICE / args.log).resolve()
    interattivo = sys.stdout.isatty() and not args.una_volta
    disegnate = 0

    try:
        while True:
            righe = _quadro(piano_file, log_file)
            if interattivo and disegnate:
                # Torna su e cancella: la barra resta ferma invece di scorrere.
                sys.stdout.write(f"\033[{disegnate}A")
            for riga in righe:
                sys.stdout.write(("\033[2K" + riga[:170] + "\n") if interattivo
                                 else riga + "\n")
            if interattivo and disegnate > len(righe):
                # Il quadro si e' accorciato: cancella le righe rimaste sotto.
                for _ in range(disegnate - len(righe)):
                    sys.stdout.write("\033[2K\n")
                sys.stdout.write(f"\033[{disegnate - len(righe)}A")
            sys.stdout.flush()
            disegnate = len(righe)
            if args.una_volta:
                return 0
            time.sleep(max(1.0, args.intervallo))
    except KeyboardInterrupt:
        # Uscita voluta: nessuna traccia di errore, e la coda continua a girare.
        sys.stdout.write("\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
