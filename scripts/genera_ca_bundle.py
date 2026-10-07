# -*- coding: utf-8 -*-
"""Esporta le CA fidate di Windows in `configs/ca_bundle.pem`.

Serve solo dove un antivirus o un proxy aziendale intercetta il TLS: la sua CA
sta nel negozio di Windows (per questo i browser funzionano) ma non nel
pacchetto `certifi` che Python usa, quindi ogni chiamata HTTPS della
simulazione fallisce con `CERTIFICATE_VERIFY_FAILED`. Vedi `src/llm/tls.py`
per il ragionamento completo.

    python scripts/genera_ca_bundle.py

Il file prodotto contiene solo certificati **pubblici** — nessuna chiave
privata — ed e' quindi innocuo, ma resta fuori dal versionamento perche' e'
proprio della macchina che lo genera.

Su Linux e macOS non serve e lo script si limita a dirlo.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
USCITA = REPO / "configs" / "ca_bundle.pem"

NEGOZI = (
    r"Cert:\LocalMachine\Root",
    r"Cert:\CurrentUser\Root",
    r"Cert:\LocalMachine\CA",
    r"Cert:\CurrentUser\CA",
)

POWERSHELL = """
$out = New-Object System.Collections.ArrayList
foreach ($store in @({negozi})) {{
  try {{ $certs = Get-ChildItem $store -ErrorAction Stop }} catch {{ continue }}
  foreach ($c in $certs) {{
    [void]$out.Add(($c.Subject + '|' + [Convert]::ToBase64String($c.RawData)))
  }}
}}
$out -join \"`n\"
"""


def esporta_windows() -> int:
    comando = POWERSHELL.format(negozi=",".join(f"'{n}'" for n in NEGOZI))
    esito = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", comando],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if esito.returncode != 0:
        print(esito.stderr[:400], file=sys.stderr)
        raise SystemExit("powershell ha fallito l'esportazione")

    righe: list[str] = []
    visti: set[str] = set()
    for riga in esito.stdout.splitlines():
        if "|" not in riga:
            continue
        soggetto, b64 = riga.split("|", 1)
        b64 = b64.strip()
        if not b64 or b64 in visti:
            continue
        visti.add(b64)
        corpo = "\n".join(b64[i : i + 64] for i in range(0, len(b64), 64))
        # Un base64 non valido qui produrrebbe un PEM che OpenSSL rifiuta in
        # blocco, invalidando anche i certificati buoni: meglio scartarlo.
        try:
            base64.b64decode(b64, validate=True)
        except Exception:  # noqa: BLE001
            continue
        righe.append(f"# {soggetto.strip()}")
        righe.append("-----BEGIN CERTIFICATE-----")
        righe.append(corpo)
        righe.append("-----END CERTIFICATE-----")

    USCITA.parent.mkdir(parents=True, exist_ok=True)
    USCITA.write_text("\n".join(righe) + "\n", encoding="ascii")
    return len(visti)


def main() -> int:
    if not sys.platform.startswith("win"):
        print("Serve solo su Windows: altrove il negozio di sistema e' gia' quello di Python.")
        return 0
    quanti = esporta_windows()
    print(f"  ok   {quanti} certificati in {USCITA.relative_to(REPO)}")

    from src.llm.tls import diagnosi  # import tardivo: prima il file deve esistere

    for chiave, valore in diagnosi().items():
        print(f"       {chiave}: {valore}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
