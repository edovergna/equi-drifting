# QM9 atom ordering matches EncodeAtomTypesTransform: {H, C, N, O, F}
_ATOMIC_NUMS = [1, 6, 7, 8, 9]
_ATOM_NAMES = ["H", "C", "N", "O", "F"]

# Covalent radii in Ångströms (Alvarez 2008)
_COV_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
_BOND_FACTOR = 1.3  # bond exists when dist < factor * (r_i + r_j)

# Maximum valence per element (conservative: allows for ionic/charged forms)
_MAX_VALENCE = {1: 1, 6: 4, 7: 4, 8: 3, 9: 1}

# Typical (target) valence used for stability: atom is stable iff bond_count == this
_STABLE_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}

# Generic distance threshold (Å) used before atom types are known
_GENERIC_BOND_THRESHOLD = 2.0

# Bond-length thresholds (Å) for the QM9-specific heuristic inference
_H_BOND_MAX = 1.20  # terminal bond shorter than this → H (C-H ~1.09)
_CARBONYL_O_MAX = 1.30  # terminal bond in [H_BOND_MAX, this) to deg≥3 nbr → carbonyl O
_O_AVG_BOND_MAX = 1.45  # avg bond for deg-2 atoms below this → O (C-O ~1.43)
_N_AVG_BOND_MAX = 1.52  # avg bond below this (but above O threshold) → N (C-N ~1.47)

# Bond length tables (pm) for QM9 atoms — ported from the EDM/E-NF reference.
# Source: http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
_BONDS1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}
_BONDS2 = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}
_BONDS3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}
_MARGIN1, _MARGIN2, _MARGIN3 = 10, 5, 3
