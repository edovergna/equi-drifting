
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb

from src.data import MolData, load_and_filter_data
from src.utils import center_positions_per_mol, get_sorted_mols
from src.model import EGNN
from src.align import kabsch_align, kabsch_align_pyg
from src.drifting_loss import compute_positive_field, compute_negative_field
from src.chem_eval import evaluate_generated_molecules
from src.viz import plot_loss, visualize_progression_with_real

def plot_closest_generated_molecule(
    model: EGNN,
    data: MolData,
    selected_real_mol_idx: int = 0,
):
    model.eval()
    with torch.no_grad():
        gen_pos, pos_list = model(
            data.gen.pos, data.gen.atom_types, data.gen.edge_index, return_change_in_pos=True
        )

    gen_pos = center_positions_per_mol(gen_pos, data.gen.batch)

    # Compare centered (but unaligned) gen positions against the selected real mol.
    # Aligning each gen mol to a single paired real mol before searching introduces
    # frame mismatch: gen mols paired to a different real end up in the wrong frame
    # when compared against selected_real_mol_idx.
    real_mol_flat = data.real.pos.view(data.n_real_mols, -1)[selected_real_mol_idx].unsqueeze(0)
    gen_pos_flattened = gen_pos.view(data.n_gen_mols, -1)
    _, closest_indices = get_sorted_mols(gen_pos_flattened, real_mol_flat)
    gen_selected_idx = closest_indices[0].item()

    num_atoms = data.real.num_atoms
    selected_real_pos = data.real.pos[selected_real_mol_idx * num_atoms : (selected_real_mol_idx + 1) * num_atoms]
    selected_real_types = data.real.atom_types[selected_real_mol_idx * num_atoms : (selected_real_mol_idx + 1) * num_atoms]
    gen_selected_types = data.gen.atom_types[gen_selected_idx * num_atoms : (gen_selected_idx + 1) * num_atoms]

    selected_pos_list = []
    for gen_pos_step in pos_list:
        step = gen_pos_step.view(data.n_gen_mols, -1)[gen_selected_idx].view(-1, 3)
        step = step - step.mean(dim=0)
        step = kabsch_align(step, selected_real_pos)
        selected_pos_list.append(step)

    fig = visualize_progression_with_real(
        pos_list=selected_pos_list,
        gen_types=gen_selected_types.to(torch.float32),
        real_pos=selected_real_pos,
        real_types=selected_real_types.to(torch.float32),
        max_steps=len(selected_pos_list),
        show=False,
    )
    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    # Add arguments here
    parser.add_argument("--num_atoms", type=int, default=17)
    parser.add_argument("--n_gen_mols", type=int, default=2)
    parser.add_argument("--n_real_mols", type=int, default=2)
    parser.add_argument("--num_atom_types", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--hidden_nf", type=int, default=32)
    parser.add_argument("--tanh_coord_updates", action="store_true")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--aggr_type", type=str, default="mean")
    parser.add_argument("--num_iters", type=int, default=13000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()

def main(args: argparse.Namespace):

    wandb.init(project="only-pos", config=vars(args), mode="offline" if args.offline else "online")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging prints
    print(f"Number of generated molecule: {args.n_gen_mols}")
    print(f"Number of real molecule: {args.n_real_mols}")
    print(f"Number of molecules in batch: {args.n_gen_mols * args.n_real_mols}")
    print(f"Number of atoms per molecule: {args.num_atoms}")

    # Load the dataset
    data = load_and_filter_data(args.num_atoms, args.n_real_mols, args.n_gen_mols, device)

    # Instantiate model and optimizer
    model = EGNN(
        num_atom_types=args.num_atom_types,
        num_blocks=args.num_blocks,
        hidden_nf=args.hidden_nf,
        attention=args.attention,
        tanh_coord_updates=args.tanh_coord_updates,
        aggr_type=args.aggr_type,
    )
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    real_pos_3d = data.real.pos.view(data.n_real_mols, data.gen.num_atoms, 3)

    loss_list = []
    for iter in range(args.num_iters):
        optimizer.zero_grad()

        # Forward pass + center
        gen_pos_flat = model(data.gen.pos, data.gen.atom_types, data.gen.edge_index)
        gen_pos_flat = center_positions_per_mol(gen_pos_flat, data.gen.batch)

        # Flat refers to [n_mols * num_atoms, 3], 3D refers to [n_mols, num_atoms, 3]
        gen_pos_3d = gen_pos_flat.view(data.n_gen_mols, data.gen.num_atoms, 3)

        V_pos = compute_positive_field(gen_pos_3d, real_pos_3d, sigma=8.0)
        V_neg = compute_negative_field(gen_pos_3d, sigma=8.0)

        field = V_pos - V_neg  # [N_gen, N_atoms, 3]

        # Target is the field-displaced position; detach so loss gradient only
        # flows through gen_pos_3d, not through the field computation.
        target_pos = (gen_pos_3d + field).detach()
        loss = F.mse_loss(gen_pos_3d, target_pos)

        loss.backward()
        optimizer.step()
        loss_list.append(loss.mean().item())

        field_norm = field.norm(dim=-1).mean()
        wandb.log({"loss": loss.mean().item(), "field_norm": field_norm.item()}, step=iter)

        if torch.allclose(field_norm, torch.tensor(0.0), atol=1e-6):
            print("The drifting field has become zero. Stopping training.")
            breakpoint()
            V_pos = compute_positive_field(gen_pos_3d, real_pos_3d, sigma=8.0)
            V_neg = compute_negative_field(gen_pos_3d, sigma=8.0)
            break

        if (iter + 1) % 500 == 0:
            print(f"Iter {iter}: Loss = {loss.mean().item()} | Field norm = {field_norm.item():.4f}")
            evaluate_generated_molecules(gen_pos_flat, data.gen.atom_types, data.gen.batch)

    plot_loss(loss_list)

    fig = plot_closest_generated_molecule(model, data)
    wandb.log({"closest_generated_molecule": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)