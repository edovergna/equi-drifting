# Monitoring Guide

## What we log and what it means

### Loss

| Metric | Logged | What it is |
| --- | --- | --- |
| `train_loss` | step + epoch | Normalized drift loss on training data |
| `val_loss` | epoch | Normalized drift loss on validation data |
| `test_loss` | epoch | Same loss on test split |

**Healthy:** both `train_loss` and `val_loss` decrease together, no large gap between them.
**Red flags:** `val_loss` diverges upward from `train_loss` (overfitting); loss explodes or becomes NaN (pipeline crashes with `TrainingDivergedException`).

---

### Drift internals (`drift_train/*`, `drift_val/*`)

Per-step stats from the drift loss function. Both splits share the same keys, under `drift_train/` (per step) and `drift_val/` (per epoch).

| Key | What it means |
| --- | --- |
| `*/mean_euclidean_distance` | Mean Euclidean distance between each generated molecule and its nearest real match (after alignment). **Primary position signal.** |
| `*/std_euclidean_distance` | Spread of the Euclidean distances — high std means some molecules are much further from real than others. |
| `*/mean_spherical_distance` | Mean geodesic distance on the atom-type sphere between generated and real. **Primary type signal.** |
| `*/std_spherical_distance` | Spread of the spherical distances. |
| `*/norm_V_posit` | Mean norm of the net position drift vector (attractive minus repulsive). Near-zero means drift is balanced/dead. |
| `*/norm_V_types` | Mean tangent-space norm of the net type drift vector. |
| `*/mean_V_posit_pos` | Mean magnitude of the attractive position drift (pulling toward real molecules). |
| `*/mean_V_posit_neg` | Mean magnitude of the repulsive position drift (pushing away from outliers). |
| `*/mean_V_types_pos` | Mean tangent norm of the attractive type drift. |
| `*/mean_V_types_neg` | Mean tangent norm of the repulsive type drift. |

**Healthy:** `mean_euclidean_distance` and `mean_spherical_distance` decrease over training. `norm_V_posit` and `norm_V_types` are non-zero and stable — the drift field is active. Attractive components (`_pos`) should dominate over repulsive (`_neg`) once the generator is near real data.
**Red flags:** `norm_V_posit` or `norm_V_types` collapses to 0 (dead drift field); distances plateau and stop decreasing (generator stuck); `mean_V_posit_neg` ≫ `mean_V_posit_pos` (repulsion dominating, generator is far from real data).

---

### Geometry (`geom/*`)

Logged each training step on generated positions.

| Key | What it means |
| --- | --- |
| `geom/pos_gen_norm_mean` | Mean distance of generated atoms from the origin. Should be in a reasonable range (QM9 molecules span ~5 Å). |
| `geom/pos_gen_norm_std` | Spread of atom distances — too low means collapse. |
| `geom/max_atom_dist` | Max pairwise distance across the batch. Sanity check for explosion. |

**Healthy:** `pos_gen_norm_mean` ~1–5 Å, stable. `max_atom_dist` bounded.
**Red flags:** `pos_gen_norm_mean` blows up (positional explosion) or collapses to ~0 (mode collapse to origin).

---

### Debug (`debug/*`)

Center-of-mass norms per graph. These should be close to 0 because both generated and real molecules are centered.

| Key | What it means |
| --- | --- |
| `debug/gen_center_norm_mean` | Mean CoM norm of generated molecules. |
| `debug/real_center_norm_mean` | Mean CoM norm of real molecules (should always be ~0). |

**Red flags:** `gen_center_norm_mean` drifts large — means centering is broken somewhere in the pipeline.

---

### Gradients (`grad/*`, callback, every step)

| Key | What it means |
| --- | --- |
| `grad/total_norm` | L2 norm of all EGNN gradients before clipping. |
| `grad/max_abs` | Max absolute gradient element. |

**Healthy:** `total_norm` in the range 0.1–10, relatively stable after warm-up. Clipping kicks in occasionally but not every step.
**Red flags:** `total_norm` spikes and stays high (unstable loss landscape); `total_norm` = 0 after the first few steps (vanishing gradients, dead network).

---

### Molecule visualizations (`mol/*`, each val epoch)

| Panel | What it shows |
| --- | --- |
| `mol/random_gen` | First K generated molecules — qualitative sanity check. |
| `mol/best_gen` | K generated molecules closest to any real molecule (embedding NN). Shows what the model *can* do. |
| `mol/worst_gen` | K generated molecules furthest from real — shows where the model still fails. |
| `mol/real_ref` | Corresponding real molecules from the same batch. |
| `mol/atom_type_dist` | Bar chart of atom type fractions: generated vs real. |

**Healthy:** `best_gen` molecules start looking like plausible molecular graphs (connected, reasonable bond lengths). `atom_type_dist` gen bars approach real bars over epochs.
**Red flags:** All atoms clustered at origin; all predicted atom type = H (the model has learned a trivial solution); wildly different atom type distribution from real.

---

### Chemical validity (`chem/*`, each val epoch)

| Key | What it means |
| --- | --- |
| `chem/validity` | Fraction of generated molecules that pass RDKit sanitization. |
| `chem/uniqueness` | Of the valid molecules, fraction that are structurally unique (by SMILES). |
| `chem/heavy_atom_mean` | Mean number of non-hydrogen atoms per molecule. |
| `chem/atom_stability` | Fraction of atoms whose bond count exactly matches the target valence (H=1, C=4, N=3, O=2, F=1). |
| `chem/mol_stability` | Fraction of molecules where every atom is stable. |
| `chem/valid_smiles` | Table of the top-50 most frequent valid SMILES generated this epoch. |

Note: **stability** and **validity** measure different things. Stability is purely geometric — does the distance-based bond count match the expected valence? Validity requires the full molecule to be chemically sane (RDKit sanitization). A molecule can be stable but still fail sanitization (e.g. disconnected graph), or pass sanitization with some unusual bond orders that don't match the target valence exactly.

**Healthy:** `atom_stability` and `mol_stability` increase in early training (geometry improving), then `validity` follows as the overall structure becomes chemically coherent. `uniqueness` stays high. `heavy_atom_mean` matches QM9 (~9 heavy atoms on average).
**Red flags:** `validity` stays at 0 for many epochs (geometry is still garbage); `uniqueness` drops toward 0 (mode collapse — model generates the same molecule every time); `atom_stability` high but `validity` near 0 (geometry looks right locally but molecules aren't coherent globally); `heavy_atom_mean` near 0 (predicting mostly H).

---

## Quick triage checklist

If something looks wrong, check in this order:

1. **`debug/gen_center_norm_mean`** — is centering broken?
2. **`grad/total_norm`** — is the network actually learning?
3. **`geom/pos_gen_norm_mean`** — is geometry exploding or collapsing?
4. **`drift_train/mean_euclidean_distance`** — are positions getting closer to real?
5. **`drift_train/mean_spherical_distance`** — are atom types getting closer to real?
6. **`mol/random_gen`** — does it look like a molecule at all?
7. **`chem/validity`** — is anything chemically sensible coming out?
