# -*- coding: utf-8 -*-
"""Gli esiti di una campagna, ogni effetto accanto al rumore che puo' imitarlo.

**Il metro non e' la baseline, e' la ripetizione.** Il braccio D viene eseguito
due volte sugli stessi semi con la stessa configurazione: cio' che cambia fra
quelle due esecuzioni non e' un effetto, e' il campionamento del modello.
(Un solo parametro differisce, l'attesa massima di una risposta, 120 s contro
300, perche' la ripetizione e' partita dopo che il tetto era stato alzato.
Nessuna delle due esecuzioni ha perso una tornata, quindi il tetto non e' mai
stato toccato e non puo' aver cambiato nulla.) Un contrasto del disegno --- la parola, la gerarchia, la lingua, il
governo stesso --- significa qualcosa solo se supera quella differenza. Senza
questo confronto ogni numero sembra un risultato, ed e' il modo piu' facile di
pubblicare rumore.

**Il rumore si misura in valore assoluto, non in media con segno.** Errore
fatto e corretto: la media delle differenze fra due esecuzioni identiche si
annulla per compensazione e dava «16 morti», mentre le singole differenze
valevano centinaia di morti con segni opposti. Con quel metro sbagliato
qualunque effetto sembrava enorme.

**Due letture, non una.** La media dice quanto, la concordanza di segno dice se
la direzione tiene. Con cinque semi una media grossa portata da un seme solo
vale meno di una media piccola con lo stesso segno su tutti e cinque: 3/5 e'
quasi una monetina, 5/5 no. Un effetto si dichiara solo se e' insieme piu'
grande del rumore tipico e unanime nella direzione.

**Territorio e celle abitate sono due cose diverse.** `celle_totali` conta ogni
cella con infrastruttura, cioe' il territorio che la colonia occupa, ed e' la
grandezza su cui la tesi fonda la prudenza; `celle_insediate` conta le sole
celle con popolazione sopra soglia. Differiscono di un ordine di grandezza --- un
centocinquanta contro una dozzina --- e scambiarle fa sparire l'effetto piu'
grande e piu' regolare del progetto: sul territorio il governo vale meno
settantotto celle con cinque semi su cinque, sulle celle abitate meno due con
tre su cinque. Errore fatto e corretto: qui si riportano entrambe, col nome che
dice quale e' quale.

**Le nascite non stanno nel registro riassuntivo** e si ricavano dall'identita'
della popolazione: chi c'e' alla fine e' chi e' partito, piu' i nati, meno i
morti. Verificato contro il conteggio degli eventi di una run: 2578 in entrambi.

Uso:
    python scripts/esiti_campagna.py                    # la campagna vLLM
    python scripts/esiti_campagna.py runs/base_pulita   # quella Ollama
"""

from __future__ import annotations

import io
import json
import os
import sys

#: I coloni fondatori, che non sono nati durante la run.
FONDATORI = 300

#: `territorio` e' la grandezza su cui la tesi fonda la prudenza: tutte le celle
#: con infrastruttura. `celle abitate` conta solo quelle con popolazione sopra
#: soglia ed e' un'altra cosa --- differiscono di un ordine di grandezza, e
#: scambiarle fa sparire l'effetto piu' netto del progetto.
GRANDEZZE = ('vivi', 'morti', 'territorio', 'celle abitate', 'morti/nascita')

#: (titolo, braccio, braccio di riferimento). Ogni riga isola una cosa sola.
EFFETTI = [
    ('la parola (it): A meno 0', 'variante_A', 'variante_0'),
    ('la parola (en): C meno F', 'variante_C', 'variante_F'),
    ('la gerarchia (it): B meno A', 'variante_B', 'variante_A'),
    ('la gerarchia (en): D meno C', 'variante_D', 'variante_C'),
    ('la lingua: C meno A', 'variante_C', 'variante_A'),
    ('la lingua: F meno 0', 'variante_F', 'variante_0'),
    ('la lingua: D meno B', 'variante_D', 'variante_B'),
    ('governare: D meno baseline', 'variante_D', 'ctrl_none'),
    ("cecita': cieco meno D", 'cieco_due_livelli', 'variante_D'),
    ('senza amministratori: gov meno D', 'solo_governatore', 'variante_D'),
    ('senza governatore: amm meno D', 'solo_amministratori', 'variante_D'),
]

#: Le due esecuzioni della stessa configurazione, da cui esce il rumore.
RIPETIZIONE = ('variante_D_rep2', 'variante_D')

#: Sotto questo numero di coppie il braccio ripetuto non misura la varianza del
#: modello, misura una singola estrazione. Tre e' il minimo per cui «tipico» e
#: «peggiore» dicono due cose diverse.
COPPIE_MINIME = 3

#: Sotto questo numero di semi la concordanza non distingue un effetto da una
#: monetina: con due semi, «2/2» capita una volta su due.
SEMI_MINIMI = 4


def leggi(base: str, braccio: str) -> dict[int, dict]:
    """Le run di un braccio, per seme. Vuoto se il braccio non e' stato eseguito."""
    percorso = os.path.join(base, braccio, 'results.jsonl')
    fuori: dict[int, dict] = {}
    if not os.path.exists(percorso):
        return fuori
    for riga in io.open(percorso, encoding='utf-8'):
        riga = riga.strip()
        if not riga:
            continue
        d = json.loads(riga)
        nascite = d['population'] - FONDATORI + d['deaths']
        fuori[int(d['seed'])] = {
            'vivi': d['population'],
            'morti': d['deaths'],
            'territorio': (d.get('espansione') or {}).get('celle_totali', 0),
            'celle abitate': (d.get('espansione') or {}).get('celle_insediate', 0),
            'morti/nascita': d['deaths'] / nascite if nascite else 0.0,
        }
    return fuori


def differenze(base: str, a: str, b: str, chiave: str) -> list[float]:
    """Le differenze seme per seme. Appaiate: due semi diversi non si sottraggono."""
    ra, rb = leggi(base, a), leggi(base, b)
    return [ra[s][chiave] - rb[s][chiave] for s in sorted(set(ra) & set(rb))]


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else 'runs/base_qwen'
    if not os.path.isdir(base):
        print(f'campagna non trovata: {base}')
        return 1

    bracci = sorted(d for d in os.listdir(base)
                    if os.path.exists(os.path.join(base, d, 'results.jsonl')))
    print(f'CAMPAGNA {base}')
    print(f'{len(bracci)} bracci con risultati')
    print()

    print('MEDIE PER BRACCIO')
    print('=' * 74)
    print(f'{"braccio":<24}{"semi":>6}{"vivi":>8}{"morti":>8}'
          f'{"territ.":>9}{"abitate":>9}{"morti/nascita":>15}')
    for braccio in bracci:
        r = leggi(base, braccio)
        if not r:
            continue
        def media(k): return sum(v[k] for v in r.values()) / len(r)  # noqa: E704
        print(f'{braccio:<24}{len(r):>6}{media("vivi"):>8.0f}'
              f'{media("morti"):>8.0f}{media("territorio"):>9.1f}'
              f'{media("celle abitate"):>9.1f}'
              f'{media("morti/nascita"):>15.3f}')
    print()

    for chiave in GRANDEZZE:
        rumore = [abs(d) for d in differenze(base, *RIPETIZIONE, chiave)]
        if len(rumore) < COPPIE_MINIME:
            print(f'=== {chiave.upper()} ===')
            print(f'  NESSUN METRO: il braccio ripetuto ha {len(rumore)} '
                  f'coppie su {COPPIE_MINIME} necessarie.')
            print('  Senza una stima del rumore un effetto non si puo\' '
                  'dichiarare: i verdetti sono omessi apposta.')
            print()
            continue
        tipico, peggiore = sum(rumore) / len(rumore), max(rumore)
        if tipico <= 0:
            print(f'=== {chiave.upper()} ===')
            print(f'  METRO NULLO: le {len(rumore)} esecuzioni ripetute '
                  'coincidono su questa grandezza.')
            print('  Una soglia di zero renderebbe «sopra il rumore» '
                  'qualunque differenza: verdetti omessi.')
            print()
            continue
        fmt = '{:+.3f}' if chiave == 'morti/nascita' else '{:+.0f}'
        nudo = fmt.replace('+', '')

        print(f'=== {chiave.upper()} ===')
        print(f'  fra due esecuzioni ripetute: tipico {nudo.format(tipico)}, '
              f'peggiore {nudo.format(peggiore)}  (su {len(rumore)} semi)')
        print(f'  {"effetto":<36}{"media":>9}{"direz.":>9}   verdetto')
        for titolo, a, b in EFFETTI:
            diff = differenze(base, a, b, chiave)
            if not diff:
                continue
            m = sum(diff) / len(diff)
            concordi = max(sum(1 for d in diff if d > 0),
                           sum(1 for d in diff if d < 0))
            n = len(diff)
            if n < SEMI_MINIMI:
                # Non e' un verdetto piu' debole: e' l'assenza di verdetto.
                verdetto = f'troppo pochi semi ({n}), non si giudica'
            elif concordi == n and abs(m) > tipico:
                verdetto = 'REGGE'
            elif concordi == n:
                verdetto = "unanime ma sotto il rumore"
            elif abs(m) <= tipico:
                verdetto = 'dentro il rumore'
            else:
                verdetto = 'grande ma direzione instabile'
            print(f'  {titolo:<36}{fmt.format(m):>9}{concordi:>6}/{n}   {verdetto}')
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
