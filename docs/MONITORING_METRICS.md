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

Per-temperature stats from the normalized drift loss function. Both splits share the same keys, under `drift_train/` (per step) and `drift_val/` (per epoch).

| Key | What it means |
| --- | --- |
| `*/scale_S` | Normalization scale. Should be stable once training settles. |
| `*/phi_gen_norm_mean/std` | L2 norm of generated embeddings. |
| `*/phi_real_norm_mean` | L2 norm of real embeddings (fixed EPT encoder). |
| `*/cosine_sim_to_nn` | Cosine similarity of each φ_gen to its nearest φ_real. **Primary signal.** |
| `*/nn_l2_distance` | L2 distance to nearest real embedding. |
| `*/attn_entropy_{τ}` | Entropy of attention weights at temperature τ. |
| `*/lambda_{τ}` | Per-temperature loss contribution. |
| `*/v_norm_{τ}` | Norm of the drift direction vector. |
| `*/attn_pos_mass_frac_{τ}` | Fraction of attention mass on molecules closer than the generator. Higher = generator is already near real data. |

**Healthy:** `cosine_sim_to_nn` increases toward 1. `nn_l2_distance` decreases. `attn_entropy` starts high (uniform attention, generator far from everything) and sharpens over time. `attn_pos_mass_frac` increases — means the generator is catching up to real data.
**Red flags:** `v_norm` collapses to 0 (dead drift field); `scale_S` blows up; `phi_gen_norm` diverges far from `phi_real_norm`.

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

### Embedding space (`embed/*`, histograms, each val epoch)

Logged as WandB histograms — look at the distribution shape, not just scalars.

| Key | What it means |
| --- | --- |
| `embed/cosine_sim_to_nn_hist` | Distribution of nearest-neighbor cosine sims. Starts near 0, should shift right toward 1. |
| `embed/phi_gen_norm_hist` | Distribution of generated embedding norms. |
| `embed/phi_real_norm_hist` | Distribution of real embedding norms (reference — should be stable). |

**Healthy:** `embed/cosine_sim_to_nn_hist` becomes more concentrated near 1 over epochs. Gen and real norm distributions converge.
**Red flags:** Bimodal cosine sim distribution (some modes collapsing, others not); gen norms far outside the real norm range.

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
| `chem/valid_smiles` | Table of the top-50 most frequent valid SMILES generated this epoch. |

**Healthy:** `validity` increases monotonically. `uniqueness` stays high (model isn't stuck repeating one structure). `heavy_atom_mean` matches QM9 (~9 heavy atoms on average).
**Red flags:** `validity` stays at 0 for many epochs (geometry is still garbage); `uniqueness` drops toward 0 (mode collapse — model generates the same molecule every time); `heavy_atom_mean` near 0 (predicting mostly H).

---

## Quick triage checklist

If something looks wrong, check in this order:

1. **`debug/gen_center_norm_mean`** — is centering broken?
2. **`grad/total_norm`** — is the network actually learning?
3. **`geom/pos_gen_norm_mean`** — is geometry exploding or collapsing?
4. **`drift/train/cosine_sim_to_nn`** — is the generator moving toward real data?
5. **`embed/cosine_sim_to_nn_hist`** — is it uniform (stuck) or concentrated (learning)?
6. **`mol/random_gen`** — does it look like a molecule at all?
7. **`chem/validity`** — is anything chemically sensible coming out?
