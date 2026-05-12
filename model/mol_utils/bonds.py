from rdkit.Chem.rdchem import BondType

from .constants import _BONDS1, _BONDS2, _BONDS3, _MARGIN1, _MARGIN2, _MARGIN3

_RDKIT_BOND_TYPES = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]


def get_bond_order(atom1: str, atom2: str, distance: float) -> int:
    """Return bond order (0=none, 1=single, 2=double, 3=triple).

    Ported directly from the EDM/E-NF reference implementation.
    distance must be in Angstroms; it is converted to pm internally.
    """
    dist_pm = distance * 100
    if atom1 not in _BONDS1 or atom2 not in _BONDS1[atom1]:
        return 0
    if dist_pm < _BONDS1[atom1][atom2] + _MARGIN1:
        if atom1 in _BONDS2 and atom2 in _BONDS2.get(atom1, {}):
            if dist_pm < _BONDS2[atom1][atom2] + _MARGIN2:
                if atom1 in _BONDS3 and atom2 in _BONDS3.get(atom1, {}):
                    if dist_pm < _BONDS3[atom1][atom2] + _MARGIN3:
                        return 3
                return 2
        return 1
    return 0
