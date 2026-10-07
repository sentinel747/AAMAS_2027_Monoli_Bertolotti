# -*- coding: utf-8 -*-
"""Il contesto TLS con cui i provider parlano davvero con la rete.

**Perche' esiste (2026-09-01).** Su questa macchina *nessuna* chiamata HTTPS
partiva, e non per la policy del repository: un antivirus con la scansione
HTTPS attiva (AVG Web/Mail Shield) intercetta il traffico e presenta un
certificato firmato da una propria CA. Quella CA sta nel negozio di Windows —
per questo i browser funzionano — ma non nel pacchetto `certifi` che Python
usa per impostazione predefinita, e in piu' non marca *critica* l'estensione
`BasicConstraints`, cosa che OpenSSL in modalita' rigorosa rifiuta comunque.
Il risultato era, su ogni endpoint:

    CERTIFICATE_VERIFY_FAILED: Basic Constraints of CA cert not marked critical

Il fallimento e' particolarmente insidioso qui, perche' `LLMProposer` per
progetto non solleva mai: una run `llm` sarebbe durata ore, avrebbe chiamato
zero volte e sarebbe finita identica alla baseline, senza un errore.

**Che cosa fa questo modulo, e che cosa NON fa.** Carica le CA del negozio di
Windows (il file prodotto da `scripts/genera_ca_bundle.py`) *in aggiunta* a
quelle di sistema, e disattiva il solo flag `VERIFY_X509_STRICT`, che e' il
controllo di conformita' formale all'RFC — lo stesso che Windows e i browser
non applicano. **La verifica del certificato resta attiva**: la catena deve
comunque risalire a una CA fidata, l'hostname viene comunque controllato. Non
esiste in questo modulo alcun percorso che disattivi la verifica.

Se il pacchetto non c'e', o se la verifica predefinita gia' funziona (macchina
senza intercettazione), il contesto restituito e' quello di sistema e nulla
cambia.
"""

from __future__ import annotations

import ssl
from pathlib import Path

#: Il pacchetto di CA esportato dal negozio di Windows, se qualcuno lo ha
#: generato. Facoltativo per progetto: su una macchina senza intercettazione
#: non serve, e la sua assenza non e' un errore.
PACCHETTO_CA = Path(__file__).resolve().parents[2] / "configs" / "ca_bundle.pem"

_contesto: ssl.SSLContext | None = None


def _costruisci() -> ssl.SSLContext:
    contesto = ssl.create_default_context()
    if PACCHETTO_CA.exists():
        try:
            contesto.load_verify_locations(cafile=str(PACCHETTO_CA))
        except (OSError, ssl.SSLError):
            # Un pacchetto illeggibile o malformato non deve impedire di
            # partire: si resta con le sole CA di sistema, e la chiamata
            # fallira' in modo diagnosticabile invece che all'import.
            pass
    # Non e' un allentamento della fiducia: e' il controllo di conformita'
    # formale dell'estensione, che la CA di un antivirus tipicamente sbaglia e
    # che ne' Windows ne' i browser applicano.
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        contesto.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return contesto


def contesto_ssl() -> ssl.SSLContext:
    """Il contesto condiviso, costruito una volta sola."""
    global _contesto
    if _contesto is None:
        _contesto = _costruisci()
    return _contesto


def diagnosi() -> dict[str, object]:
    """Che cosa e' in uso, per farlo dire a un preflight invece di indovinarlo."""
    contesto = contesto_ssl()
    return {
        "pacchetto_ca": str(PACCHETTO_CA),
        "pacchetto_presente": PACCHETTO_CA.exists(),
        "ca_caricate": len(contesto.get_ca_certs()),
        "verifica_attiva": contesto.verify_mode == ssl.CERT_REQUIRED,
        "controllo_hostname": contesto.check_hostname,
        "x509_strict": bool(
            getattr(ssl, "VERIFY_X509_STRICT", 0) & contesto.verify_flags
        ),
    }
