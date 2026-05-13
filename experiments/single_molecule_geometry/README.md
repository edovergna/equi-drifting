# Single-Molecule Geometry Overfit

This diagnostic checks whether the EGNN can learn positions at all in the
simplest useful setting:

1. Load one QM9 molecule.
2. Sample one fixed prior point cloud and fixed random node features once.
3. Train the EGNN from that same prior every step.
4. Optimize only geometry, ignoring atom-type prediction/loss.

The loss is MSE between centered generated coordinates and centered target
coordinates. The fixed random node features are kept because they give nodes
distinct identities; with identical features, a fully connected EGNN has much
less information for matching a specific ordered target geometry.

Example:

```bash
python experiments/single_molecule_geometry/train_single.py \
  --root data/QM9 \
  --output_dir outputs/single_molecule_geometry \
  --max_num_atoms 18 \
  --steps 5000 \
  --hidden_dim 64 \
  --num_layers 6 \
  --lr 1e-3
```

Useful outputs:

- `fixed_prior.pt`: fixed input features, positions, and edge index.
- `target.pt`: target positions and atom numbers.
- `best_model.pt`: best EGNN weights by RMSD.
- `metrics.csv`: step-wise loss/RMSD.
- `prior.xyz`, `target.xyz`, `final.xyz`: quick geometry inspection files.

To visualize the fit after training:

```bash
python experiments/single_molecule_geometry/visualize_result.py \
  --output_dir outputs/single_molecule_geometry
```

On the cluster, submit the visualization job after the training job has finished:

```bash
sbatch experiments/single_molecule_geometry/visualize_slurm.sh
```

This writes:

- `geometry_side_by_side.png`
- `geometry_overlay.png`

## Multi-Molecule Overfit

To train the same geometry-only diagnostic on a fixed batch of 100 molecules:

```bash
sbatch experiments/single_molecule_geometry/run_many_slurm.sh
```

or directly:

```bash
python experiments/single_molecule_geometry/train_many.py \
  --root data/QM9 \
  --output_dir outputs/many_molecule_geometry \
  --n_molecules 100 \
  --max_num_atoms 18 \
  --steps 10000 \
  --hidden_dim 64 \
  --num_layers 6 \
  --lr 1e-3
```

This samples one fixed prior batch once, stores it in `fixed_prior.pt`, and then
passes that exact same batch through the EGNN at every training step. It writes
`targets.pt`, `final.pt`, `best_model.pt`, `metrics.csv`, and a small set of
example XYZ triplets under `xyz_examples/`.

To visualize the first few molecules from the multi run:

```bash
sbatch experiments/single_molecule_geometry/visualize_many_slurm.sh
```

This writes `many_geometry_overlay_grid.png` and
`many_geometry_per_molecule_rmsd.csv`.
