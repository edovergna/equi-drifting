import matplotlib.pyplot as plt
import torch

def visualize_both_molecules(gen_pos: torch.Tensor, real_pos: torch.Tensor):
    gen_pos = gen_pos.view(-1, 3).detach().cpu().numpy()
    real_pos = real_pos.view(-1, 3).detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(gen_pos[:, 0],  gen_pos[:, 1],  gen_pos[:, 2],  color='red',  label='Generated', alpha=0.8)
    ax.scatter(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2], color='blue', label='Real',      alpha=0.8)

    ax.set_title('Generated vs Real Molecule')
    ax.legend()
    plt.show()

def _apply_limits(ax, x_lim, y_lim, z_lim):
    ax.set_autoscale_on(False)
    ax.set_xlim3d(*x_lim)
    ax.set_ylim3d(*y_lim)
    ax.set_zlim3d(*z_lim)


def _plot_molecule(ax, pos, types, title='', limits=None):
    """Scatter-plots a single molecule onto an existing 3-D axis."""
    if limits is not None:
        ax.set_autoscale_on(False)          # lock before any scatter call
        ax.set_xlim3d(*limits[0])
        ax.set_ylim3d(*limits[1])
        ax.set_zlim3d(*limits[2])

    pos_np = pos.view(-1, 3).detach().cpu().numpy()
    types_r = torch.round(types, decimals=0)

    for atom_type in torch.unique(types_r):
        mask = (types_r == atom_type).cpu().numpy()
        ax.scatter(pos_np[mask, 0], pos_np[mask, 1], pos_np[mask, 2],
                   label=f'Type {atom_type.item()}')

    ax.set_title(title)
    ax.legend(fontsize=7)


def visualize_progression_with_real(
    pos_list: list[torch.Tensor],
    gen_types: torch.Tensor,
    real_pos: torch.Tensor,
    real_types: torch.Tensor,
    step_indices: list[int] | None = None,
    max_steps: int = 5
):
    if step_indices is None:
        total = len(pos_list)
        step_indices = [int(i * (total - 1) / (max_steps - 1)) for i in range(max_steps)]

    # --- Compute consistent limits from ALL positions (trajectory + real) ---
    all_pos = torch.cat(
        [p.view(-1, 3) for p in pos_list] + [real_pos.view(-1, 3)]
    ).detach().cpu().numpy()

    padding = 0.5
    limits = (
        (all_pos[:, 0].min() - padding, all_pos[:, 0].max() + padding),
        (all_pos[:, 1].min() - padding, all_pos[:, 1].max() + padding),
        (all_pos[:, 2].min() - padding, all_pos[:, 2].max() + padding),
    )

    def _enforce_limits(ax):
        ax.set_autoscale_on(False)
        ax.set_xlim3d(*limits[0])
        ax.set_ylim3d(*limits[1])
        ax.set_zlim3d(*limits[2])

        ax.set_xticks([int(t) for t in range(int(limits[0][0]), int(limits[0][1]) + 1)])
        ax.set_yticks([int(t) for t in range(int(limits[1][0]), int(limits[1][1]) + 1)])
        ax.set_zticks([int(t) for t in range(int(limits[2][0]), int(limits[2][1]) + 1)])

    n_cols = len(step_indices) + 2
    fig = plt.figure(figsize=(4 * n_cols, 4))

    # --- Generated snapshots ---
    for plot_idx, step_idx in enumerate(step_indices):
        ax = fig.add_subplot(1, n_cols, plot_idx + 1, projection='3d')
        _plot_molecule(ax, pos_list[step_idx], gen_types, title=f'Step {step_idx}')
        _enforce_limits(ax)

    # --- Real molecule with atom types ---
    ax_real = fig.add_subplot(1, n_cols, n_cols - 1, projection='3d')
    _plot_molecule(ax_real, real_pos, real_types, title='Real Molecule')
    _enforce_limits(ax_real)

    # --- Overlap: final generated vs real ---
    ax_overlap = fig.add_subplot(1, n_cols, n_cols, projection='3d')
    final_pos = pos_list[-1].view(-1, 3).detach().cpu().numpy()
    real_pos_np = real_pos.view(-1, 3).detach().cpu().numpy()
    ax_overlap.scatter(final_pos[:, 0],   final_pos[:, 1],   final_pos[:, 2],   color='red',  label='Generated', alpha=0.8)
    ax_overlap.scatter(real_pos_np[:, 0], real_pos_np[:, 1], real_pos_np[:, 2], color='blue', label='Real',      alpha=0.8)
    ax_overlap.set_title('Final vs Real')
    ax_overlap.legend(fontsize=7)
    _enforce_limits(ax_overlap)

    plt.tight_layout()
    plt.show()


def visualize_both_molecules_with_atom_types(
    gen_pos, gen_types, real_pos, real_types
):
    fig = plt.figure(figsize=(12, 6))
    _plot_molecule(fig.add_subplot(121, projection='3d'), gen_pos,  gen_types,  'Generated Molecule')
    _plot_molecule(fig.add_subplot(122, projection='3d'), real_pos, real_types, 'Real Molecule')
    plt.show()


def plot_loss(loss_list):
    plt.plot(loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
