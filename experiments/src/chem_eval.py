import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from model.mol_utils.validity import batch_to_validity
from model.mol_utils.stability import batch_to_stability, heavy_atom_counts

import torch
import numpy as np

def evaluate_generated_molecules(gen_pos: torch.Tensor, a_hard: torch.Tensor, batch_vec: torch.Tensor, show: bool = False):

    results = batch_to_validity(gen_pos.detach().cpu(), a_hard.detach().cpu(), batch_vec.detach().cpu())
    heavy = heavy_atom_counts(a_hard.detach().cpu(), batch_vec.detach().cpu())
    atom_stable_frac, mol_stable_frac = batch_to_stability(gen_pos.detach().cpu(), a_hard.detach().cpu(), batch_vec.detach().cpu())

    n_total = len(results)
    n_valid = sum(1 for ok, _ in results if ok)
    valid_ids = [ident for ok, ident in results if ok and ident is not None]
    uniqueness = len(set(valid_ids)) / len(valid_ids) if valid_ids else 0.0
    validity = n_valid / n_total if n_total > 0 else 0.0
    heavy_mean = float(np.mean(heavy)) if heavy else 0.0

    if show:
        print(f"Validity: {validity:.4f}, Uniqueness: {uniqueness:.4f}, Mean Heavy Atoms: {heavy_mean:.2f}")
        print(f"Atom Stability Fraction: {atom_stable_frac:.4f}, Molecule Stability Fraction: {mol_stable_frac:.4f}")
    
    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "mean_heavy_atoms": heavy_mean,
        "atom_stability_fraction": atom_stable_frac,
        "molecule_stability_fraction": mol_stable_frac,
    }
