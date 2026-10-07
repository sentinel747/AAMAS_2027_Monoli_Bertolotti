import numpy as np
from src.world.resources import ResourceBundle
from src.world.structures import BUILD_COSTS, StructureType
from src.core import constants as C


def test_resource_index_matches_resourcebundle_order():
    assert C.RESOURCES == tuple(ResourceBundle.__dataclass_fields__)
    assert C.NR == len(C.RESOURCES)
    assert C.R["water"] == C.RESOURCES.index("water")


def test_structure_index_matches_enum_order():
    assert C.STRUCTURES == tuple(StructureType)
    assert C.S[StructureType.GREENHOUSE] == 3


def test_build_cost_matrix_matches_build_costs():
    for st, bundle in BUILD_COSTS.items():
        for res, value in bundle.to_dict().items():
            assert C.BUILD_COST_M[C.S[st], C.R[res]] == value


def test_struct_fx_matrix_matches_local_effects():
    # Effetti base a integrità 1.0 (efficiency=1): la matrice deve
    # riprodurre Structure.local_effect per ogni tipo.
    from src.world.structures import Structure
    for st in StructureType:
        s = Structure(type=st, x=0, y=0, integrity=1.0)
        for eff, value in s.local_effect.items():
            assert np.isclose(C.STRUCT_FX_M[C.S[st], C.E[eff]], value)


def test_action_codes_are_stable_and_mapped():
    assert C.ACT.OBSERVE == 0
    assert C.ACT_TO_ACTIONTYPE[int(C.ACT.BUILD)] == "build"  # famiglia; tipo in param
    assert C.ACT_TO_ACTIONTYPE[int(C.ACT.COMMUNICATE)] == "communicate"
