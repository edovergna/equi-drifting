import torch

# HARD-CODED FOR QM9
BOND_ORDER_1_THRESHOLDS = torch.tensor([
    [0.84, 1.19, 1.11, 1.06, 1.02],
    [1.19, 1.64, 1.57, 1.53, 1.45],
    [1.11, 1.57, 1.55, 1.50, 1.46],
    [1.06, 1.53, 1.50, 1.58, 1.52],
    [1.02, 1.45, 1.46, 1.52, 1.52],
])
BOND_ORDER_2_THRESHOLDS = torch.tensor([
    [0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.39, 1.34, 1.25, 0.00],
    [0.00, 1.34, 1.30, 1.26, 0.00],
    [0.00, 1.25, 1.26, 1.26, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00],
])
BOND_ORDER_3_THRESHOLDS = torch.tensor([
    [0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.23, 1.19, 1.16, 0.00],
    [0.00, 1.19, 1.13, 0.00, 0.00],
    [0.00, 1.16, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00],
])

ATOM_TYPICAL_VALENCE = torch.tensor([1.0, 4.0, 3.0, 2.0, 1.0])

def compute_chem_loss(pred_pos, pred_type_probs, cfg):
    batch_size, n_atoms, _ = pred_pos.shape
    eye = torch.eye(n_atoms, device=pred_pos.device, dtype=torch.bool).unsqueeze(0)

    d = torch.cdist(pred_pos, pred_pos).masked_fill(eye, float("inf"))
    clash_loss = torch.relu(cfg["clash_threshold"] - d).pow(2).sum(dim=(1, 2)).mean() / (n_atoms * (n_atoms - 1))

    p1, p2, p3 = _soft_bond_order_components(pred_pos, pred_type_probs, cfg)
    p1 = p1.masked_fill(eye, 0.0)
    p2 = p2.masked_fill(eye, 0.0)
    p3 = p3.masked_fill(eye, 0.0)
    soft_order = p1 + p2 + p3
    soft_valence = soft_order.sum(dim=-1)

    typical_valence = ATOM_TYPICAL_VALENCE.to(pred_pos.device, dtype=pred_pos.dtype)
    expected_valence = pred_type_probs @ typical_valence
    valence_excess_loss = torch.relu(soft_valence - expected_valence).pow(2).mean()

    hydrogen_prob = pred_type_probs[..., 0]
    single_degree = p1.sum(dim=-1)
    second_bond_prob = p1.topk(k=2, dim=-1).values[..., 1]
    hydrogen_valence_loss = (
        hydrogen_prob * (single_degree - 1.0).pow(2)
        + 2.0 * hydrogen_prob * second_bond_prob.pow(2)
        + hydrogen_prob * (p2 + p3).sum(dim=-1).pow(2)
    ).mean()

    total_aux = (
        cfg["lambda_clash"] * clash_loss
        + cfg["lambda_valence_excess"] * valence_excess_loss
        + cfg["lambda_hydrogen_valence"] * hydrogen_valence_loss
    )

    return total_aux, {
        "clash_loss": clash_loss,
        "valence_excess_loss": valence_excess_loss,
        "hydrogen_valence_loss": hydrogen_valence_loss,
    }


def _soft_bond_order_components(pred_pos, pred_type_probs, cfg):
    d = torch.cdist(pred_pos, pred_pos)
    temp = cfg["bond_temperature"]
    p1 = _soft_pair_compatibility(pred_type_probs, BOND_ORDER_1_THRESHOLDS) * torch.sigmoid(
        (_soft_pair_threshold(pred_type_probs, BOND_ORDER_1_THRESHOLDS) - d) / temp
    )
    p2 = _soft_pair_compatibility(pred_type_probs, BOND_ORDER_2_THRESHOLDS) * torch.sigmoid(
        (_soft_pair_threshold(pred_type_probs, BOND_ORDER_2_THRESHOLDS) - d) / temp
    )
    p3 = _soft_pair_compatibility(pred_type_probs, BOND_ORDER_3_THRESHOLDS) * torch.sigmoid(
        (_soft_pair_threshold(pred_type_probs, BOND_ORDER_3_THRESHOLDS) - d) / temp
    )
    return p1, p2, p3

def _soft_pair_threshold(pred_type_probs, thresholds):
    thresholds = thresholds.to(pred_type_probs.device, dtype=pred_type_probs.dtype)
    return torch.einsum("bik,kl,bjl->bij", pred_type_probs, thresholds, pred_type_probs)


def _soft_pair_compatibility(pred_type_probs, thresholds):
    mask = (thresholds > 0).to(pred_type_probs.device, dtype=pred_type_probs.dtype)
    return torch.einsum("bik,kl,bjl->bij", pred_type_probs, mask, pred_type_probs)