# -*- coding: utf-8 -*-
"""Controlla le due condizioni di ammissione che si leggono dal registro.

La tesi dichiara che una run non e' comparabile se ha perso tornate del
governatore o se ha avuto chiamate fallite: nel primo caso un intervallo resta
senza legge, nel secondo il ripiego del fornitore prende il posto di una
decisione, e in entrambi la run non e' piu' la configurazione che dichiara di
essere. Le due condizioni si leggono prima di guardare l'esito, da
`governor_misses` in `results.jsonl` e da `failed_calls` in `api_usage.json`
(riportato nel campo `api` del risultato), e devono valere zero.

**Questo script esiste perche' il numero scritto in tesi dev'essere
rigenerabile.** L'insieme delle campagne riportate e' dichiarato qui sotto e non
dedotto da un glob su `runs/`, che pesca anche gli archivi superati: in
`runs/campagna_scarsa` sopravvivono per esempio serie di modelli poi rifatte in
`runs/modelli`, e due di quelle hanno davvero chiamate fallite. Tenerle fuori e'
una scelta, e va vista.

Uso:
    python scripts/audit_registri.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Le campagne i cui numeri compaiono nel capitolo dei risultati.
CAMPAGNE = {
    "riferimento, baseline": ["runs/campagna_scarsa/ctrl_none"],
    "riferimento, LLM (Gov+Amm) (tre esecuzioni)": [
        "runs/campagna_scarsa_v3/llm_completo_amm",
        "runs/campagna_scarsa_v3/llm_completo_amm_rep2",
        "runs/campagna_scarsa_v3/llm_completo_amm_rep3",
    ],
    "mondi": ["runs/mondi"],
    "famiglie di modelli": ["runs/modelli"],
    "gradini di contesto": ["runs/livelli"],
    "accentramento e decentramento": ["runs/governo"],
    "decentramento dentro la run (2 e 5 settembre)": [
        "runs/campagna_2026-09-02",
        "runs/campagna_scarsa_v2",
    ],
}


def chiamate_fallite(registro: Path, seme) -> int | None:
    """Chiamate fallite di una run, dal risultato o dal file su disco.

    Le campagne del 2 settembre sono anteriori al campo `api` dentro
    `results.jsonl`, ma il registro c'e' lo stesso: sta in `api_usage.json`
    dentro la cartella del seme. Cercarlo li' evita di dichiarare «non
    verificabile» una run che e' verificabilissima.
    """
    for cartella in sorted(registro.parent.glob(f"*seed{seme}")):
        f = cartella / "api_usage.json"
        if f.exists():
            return int(json.loads(f.read_text(encoding="utf-8")).get("failed_calls", 0))
    return None


def risultati(radice: Path):
    for p in sorted(radice.rglob("results.jsonl")):
        for riga in p.read_text(encoding="utf-8").splitlines():
            riga = riga.strip()
            if not riga:
                continue
            try:
                yield p, json.loads(riga)
            except json.JSONDecodeError:
                continue


# Parametri che devono valere lo stesso su OGNI run riportata. Se uno di questi
# variasse, il confronto fra bracci non sarebbe piu' a parita' di ingressi e gli
# indici di scelta del sito e di input terrestre andrebbero usati come controlli
# invece che restare descrittivi. (Il profilo di mappa e' escluso: e' la
# variabile della campagna dei mondi. Le celle per distretto pure: valgono zero
# quando gli amministratori sono spenti.)
PARITA = [
    ("steps", lambda r, c: r.get("steps")),
    ("agents", lambda r, c: r.get("agents")),
    ("dotazione", lambda r, c: c.get("dotazione")),
    ("cadenza", lambda r, c: c.get("cadence_steps")),
    ("resa ISRU", lambda r, c: c.get("isru_material_rate")),
    ("priorita' di costruzione", lambda r, c: c.get("development_build_priority")),
    ("piano serre ridotto", lambda r, c: c.get("piano_serre_ridotto")),
    # **`thinking` va normalizzato prima di confrontarlo.** Il registro scrive
    # `''` quando l'opzione non e' stata passata e `'off'` quando lo e' stata,
    # e le due forme sembrano una differenza di trattamento: la campagna dei
    # mondi ha la prima esecuzione a `''` e le repliche a `'off'`, il che
    # sembrerebbe dire che le repliche non sono esecuzioni identiche, e quindi
    # che lo scarto misurato su di esse non e' la varianza del modello.
    # Non e' cosi': il provider passa il livello per `normalizza()`, che manda
    # a `off` sia la stringa vuota sia l'assenza, quindi le due forme
    # producono la STESSA richiesta. Qui si confronta cio' che e' stato
    # mandato, non cio' che e' stato scritto.
    ("ragionamento", lambda r, c: (c.get("thinking") or "off") if c.get("provider") else None),
]


def main() -> None:
    totali = con_modello = 0
    guasti: list[str] = []
    valori: dict[str, dict] = {nome: {} for nome, _ in PARITA}
    for nome, percorsi in CAMPAGNE.items():
        n = m = 0
        for rel in percorsi:
            radice = ROOT / rel
            if not radice.exists():
                guasti.append(f"MANCA la cartella {rel}")
                continue
            for p, r in risultati(radice):
                n += 1
                config = r.get("config") or {}
                for parametro, leggi in PARITA:
                    chiave = repr(leggi(r, config))
                    valori[parametro][chiave] = valori[parametro].get(chiave, 0) + 1
                if not config.get("provider"):
                    continue
                m += 1
                perse = r.get("governor_misses") or 0
                fallite = (r.get("api") or {}).get("failed_calls")
                if fallite is None:
                    fallite = chiamate_fallite(p, r.get("seed"))
                if perse:
                    guasti.append(f"{p.relative_to(ROOT)} seme {r.get('seed')}: "
                                  f"{perse} tornate perse")
                if fallite:
                    guasti.append(f"{p.relative_to(ROOT)} seme {r.get('seed')}: "
                                  f"{fallite} chiamate fallite")
                if fallite is None:
                    guasti.append(f"{p.relative_to(ROOT)} seme {r.get('seed')}: "
                                  f"nessun registro delle chiamate")
        totali += n
        con_modello += m
        print(f"{nome}: {n} esecuzioni, di cui {m} con modello")

    print(f"\ntotale: {totali} esecuzioni, {con_modello} con modello")

    print("\nparita' degli ingressi:")
    for parametro, _ in PARITA:
        distinti = valori[parametro]
        segno = "UGUALE" if len(distinti) == 1 else "DIVERSO"
        print(f"  {parametro}: {segno} {distinti}")

    if guasti:
        print(f"\nNON AMMISSIBILI ({len(guasti)}):")
        for g in guasti:
            print(f"  {g}")
    else:
        print("\nTutte ammissibili: zero tornate perse, zero chiamate fallite.")


if __name__ == "__main__":
    main()
