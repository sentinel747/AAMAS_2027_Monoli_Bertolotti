# -*- coding: utf-8 -*-
"""Il decentramento si misura DENTRO la run, non solo fra bracci.

Un braccio +amm che vince sulla baseline dice che la coppia governo+strato ha
fatto meglio del nulla; non dice se a fare la differenza sia stato il livello
locale. La misura che solo il registro per tornata permette e' questa: nei
distretti in cui l'amministratore ha riscritto, i morti per cella-passo sono
scesi di piu' che nei distretti che nello stesso tick hanno accettato? E' una
differenza nelle differenze --- prima/dopo la decisione, riscritto/accettato ---
e i test qui sotto la fissano su un caso minuscolo costruito a mano.

**Perche' morti per cella-passo e non per colono-passo.** Il registro non porta
la popolazione del distretto per tornata e il replay per eventi comprime
l'osservazione, quindi i coloni-passo per cella non si ricostruiscono. Le celle
di un distretto sono costanti fra due tornate, quindi il prima/dopo dentro lo
stesso distretto non ne risente; il confronto fra distretti di taglia diversa
si'. Va detto quando si riporta.
"""

from scripts.analisi_decentramento import (
    differenza_nelle_differenze,
    morti_per_cella_e_finestra,
    stato_del_distretto,
)

# Due tornate a cadenza 10: la prima al passo 10, la seconda al passo 20, la
# terza al passo 30. Finestra "prima" della tornata 20 = [10, 20), "dopo" = [20, 30).
TORNATE = [
    {"step": 10, "last_round": [
        {"district": 0, "accepted_government_policy": True, "rewrite_without_effect": False, "cells": [[1, 1]]},
        {"district": 1, "accepted_government_policy": True, "rewrite_without_effect": False, "cells": [[2, 2], [2, 3]]},
    ]},
    {"step": 20, "last_round": [
        {"district": 0, "accepted_government_policy": False, "rewrite_without_effect": False, "cells": [[1, 1]]},
        {"district": 1, "accepted_government_policy": True, "rewrite_without_effect": False, "cells": [[2, 2], [2, 3]]},
    ]},
    {"step": 30, "last_round": [
        {"district": 0, "accepted_government_policy": True, "rewrite_without_effect": False, "cells": [[1, 1]]},
        {"district": 1, "accepted_government_policy": True, "rewrite_without_effect": False, "cells": [[2, 2], [2, 3]]},
    ]},
]

MORTI = (
    # distretto 0: 4 morti prima della riscrittura, 1 dopo
    [{"y": 1, "x": 1, "death_step": s} for s in (11, 13, 15, 17)]
    + [{"y": 1, "x": 1, "death_step": 25}]
    # distretto 1: 2 morti prima, 2 dopo, su due celle
    + [{"y": 2, "x": 2, "death_step": 12}, {"y": 2, "x": 3, "death_step": 18}]
    + [{"y": 2, "x": 2, "death_step": 22}, {"y": 2, "x": 3, "death_step": 28}]
)


def test_lo_stato_distingue_i_tre_esiti():
    assert stato_del_distretto({"accepted_government_policy": True}) == "accetta"
    assert stato_del_distretto({"accepted_government_policy": False, "rewrite_without_effect": False}) == "riscrive"
    assert stato_del_distretto({"accepted_government_policy": False, "rewrite_without_effect": True}) == "a_vuoto"


def test_i_morti_si_contano_per_cella_dentro_la_finestra():
    conteggio = morti_per_cella_e_finestra(MORTI, [(1, 1)], 10, 20)
    assert conteggio == 4
    assert morti_per_cella_e_finestra(MORTI, [(1, 1)], 20, 30) == 1
    # Il confine destro e' escluso: un morto al passo 20 appartiene alla finestra dopo.
    assert morti_per_cella_e_finestra([{"y": 1, "x": 1, "death_step": 20}], [(1, 1)], 10, 20) == 0


def test_la_differenza_nelle_differenze_premia_la_riscrittura_che_ha_ridotto_i_morti():
    esito = differenza_nelle_differenze(TORNATE, MORTI)
    # Distretto 0 (riscrive alla tornata 20): prima 4 morti / (1 cella x 10 passi) = 0.4
    # per cella-passo, dopo 1 / 10 = 0.1 -> delta -0.3.
    # Distretto 1 (accetta): prima 2 / (2 x 10) = 0.1, dopo 2 / 20 = 0.1 -> delta 0.
    assert esito["riscrive"]["decisioni"] == 1
    assert esito["riscrive"]["delta_medio"] == -0.3
    assert esito["accetta"]["decisioni"] == 1
    assert esito["accetta"]["delta_medio"] == 0.0
    assert esito["effetto"] == -0.3


def test_la_prima_e_l_ultima_tornata_non_entrano():
    """Senza una finestra prima o dopo non c'e' differenza da fare."""
    esito = differenza_nelle_differenze(TORNATE, MORTI)
    assert esito["riscrive"]["decisioni"] + esito["accetta"]["decisioni"] + esito["a_vuoto"]["decisioni"] == 2


def test_senza_riscritture_l_effetto_non_e_definito():
    solo_accetta = [dict(t, last_round=[dict(d, accepted_government_policy=True) for d in t["last_round"]]) for t in TORNATE]
    esito = differenza_nelle_differenze(solo_accetta, MORTI)
    assert esito["riscrive"]["decisioni"] == 0
    assert esito["effetto"] is None


# --------------------------------------------- il controllo contro il ritorno alla media


def test_saltando_la_finestra_vista_il_prima_e_due_tornate_indietro():
    """L'amministratore riscrive dove i morti dell'ULTIMA finestra sono alti: quella
    finestra e' selezionata, e il dopo puo' scendere per puro ritorno alla media.
    Il controllo confronta la finestra PRIMA di quella vista con quella dopo la
    decisione: se l'effetto resta, non e' un artefatto della selezione."""
    tornate = [
        {"step": 0, "last_round": [{"district": 0, "accepted_government_policy": True, "cells": [[1, 1]]}]},
        {"step": 10, "last_round": [{"district": 0, "accepted_government_policy": True, "cells": [[1, 1]]}]},
        {"step": 20, "last_round": [{"district": 0, "accepted_government_policy": False, "cells": [[1, 1]]}]},
        {"step": 30, "last_round": [{"district": 0, "accepted_government_policy": True, "cells": [[1, 1]]}]},
    ]
    # [0,10): 2 morti (normale); [10,20): 6 morti (il picco visto); [20,30): 2 morti.
    morti = ([{"y": 1, "x": 1, "death_step": s} for s in (1, 5)]
             + [{"y": 1, "x": 1, "death_step": s} for s in (11, 12, 13, 14, 15, 16)]
             + [{"y": 1, "x": 1, "death_step": s} for s in (21, 25)])
    normale = differenza_nelle_differenze(tornate, morti)
    controllo = differenza_nelle_differenze(tornate, morti, salta_finestra_vista=True)
    # Senza controllo il picco visto fa sembrare la riscrittura efficace: 0.2 -> 0.6 -> 0.2.
    assert normale["riscrive"]["delta_medio"] == -0.4
    # Saltando la finestra vista, prima e dopo sono uguali: nessun effetto.
    assert controllo["riscrive"]["delta_medio"] == 0.0
    assert controllo["riscrive"]["decisioni"] == 1


# ------------------------------------------------- nascite e morti, non solo vivi


def test_le_nascite_si_ricavano_da_vivi_morti_e_fondatori():
    """Vivi a fine run premia chi cresce: senza nascite e morti accanto, un governo
    prudente (meno morti, meno nascite) e uno incapace hanno lo stesso numero."""
    from scripts.analisi_decentramento import demografia

    riga = {"population": 1599, "deaths": 844, "agents": 300}
    d = demografia(riga)
    assert d["nascite"] == 1599 + 844 - 300
    assert d["morti"] == 844
    assert d["morti_per_nascita"] == round(844 / (1599 + 844 - 300), 3)


def test_senza_nascite_il_rapporto_non_divide_per_zero():
    from scripts.analisi_decentramento import demografia

    d = demografia({"population": 300, "deaths": 0, "agents": 300})
    assert d["nascite"] == 0
    assert d["morti_per_nascita"] is None


# ------------------------------------------------ il denominatore per colono


def test_con_la_popolazione_nel_registro_il_denominatore_e_per_colono():
    """Registri dal 2026-09-05 sera: `population` per decisione. Il distretto 0 ha
    una cella e 10 coloni, il distretto 1 due celle e 2 coloni: per cella-passo il
    distretto 1 pesa il doppio, per colono-passo un quinto."""
    tornate = [dict(t, last_round=[dict(d, population=10 if d["district"] == 0 else 2) for d in t["last_round"]]) for t in TORNATE]
    per_cella = differenza_nelle_differenze(tornate, MORTI)
    per_colono = differenza_nelle_differenze(tornate, MORTI, per_colono=True)
    # Distretto 0 (riscrive): prima 4 morti / (10 coloni x 10 passi) = 0.04, dopo 1/100 = 0.01 -> -0.03.
    assert per_colono["riscrive"]["delta_medio"] == -0.03
    assert per_cella["riscrive"]["delta_medio"] == -0.3
    # Distretto 1 (accetta): 2/(2x10)=0.1 prima e dopo -> 0.
    assert per_colono["accetta"]["delta_medio"] == 0.0


def test_senza_popolazione_nel_registro_il_per_colono_ricade_sulle_celle():
    assert differenza_nelle_differenze(TORNATE, MORTI, per_colono=True) == differenza_nelle_differenze(TORNATE, MORTI)
