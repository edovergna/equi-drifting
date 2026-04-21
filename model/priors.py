import torch
from scipy.optimize import linear_sum_assignment
from torch.distributions import Exponential
from torch.nn.functional import one_hot, softmax


def simplex_proj(x):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    # seq has shape (batch_size, sequence_length, alphabet_size)
    Y = x.reshape(-1, x.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(x.shape)


def gaussian(n: int, d: int, std: float = 1.0, simplex_center: bool = False):
    """
    Generate a prior feature by sampling from a Gaussian distribution.
    """
    p = torch.randn(n, d) * std

    if simplex_center:
        p = p + 1 / d
    return p


def centered_normal_prior(n: int, d: int, std: float = 4.0):
    """
    Generate a prior feature by sampling from a centered normal distribution.
    """
    prior_feat = torch.randn(n, d) * std
    prior_feat = prior_feat - prior_feat.mean(dim=0, keepdim=True)
    return prior_feat


# Temporarily removing because we don't know if we want to use it
# and we don't want to spend time fixing it right now.
# def centered_normal_prior_batched_graph(
#     g: dgl.DGLGraph, node_batch_idx: torch.Tensor, std: float = 4.0
# ):

#     n = g.num_nodes()
#     prior_sample = torch.randn(n, 3, device=g.device)
#     with g.local_scope():
#         g.ndata["prior_sample"] = prior_sample
#         prior_sample = (
#             prior_sample
#             - dgl.readout_nodes(g, feat="prior_sample", op="mean")[node_batch_idx]
#         )

#     return prior_sample


def barycenter_prior(n: int, d: int, blur: float = 0.0):

    p = torch.ones(n, d) / d

    if blur != 0.0:
        p = p + torch.randn_like(p) * blur
        p = simplex_proj(p)

    return p


def biased_simplex_prior(
    n, d, vertex_prob: float = 0.75, std: float = 0.2, vertex_idx: int = 0
):
    """
    Generate samples from a simplex which are biased towards one category.
    """
    non_zero_weight = (1 - vertex_prob) / (d - 1)
    mu = torch.ones(d) * non_zero_weight
    mu[vertex_idx] = vertex_prob
    simplex_sample = mu.unsqueeze(0) + torch.randn(n, d) * std
    simplex_sample = softmax(simplex_sample / (1 / d), dim=1)
    return simplex_sample


def uniform_simplex_prior(n, d):
    """
    Generate samples from a uniform distribution on a simplex.
    """
    exp_dist = Exponential(torch.tensor(1.0))
    sample = exp_dist.sample((n, d))
    sample = sample / sample.sum(dim=1, keepdim=True)
    return sample


def sample_marginal(n: int, d: int, p: torch.Tensor, blur: float = None):
    """
    Sample from the marginal distribution of a categorical variable.

    Args:
    - n: number of samples to generate
    - d: number of categories
    - p: probability vector of shape (d,) representing the marginal distribution
    - blur: if not None, adds Gaussian noise to the one-hot samples
        and re-projects them onto the simplex to create a blurred
        version of the categorical distribution
    """
    # It looks at the global statistics of the dataset stored in p, then picks
    # n atoms based on those odds. After that it one-hot encodes those atoms.
    prior_idxs = torch.multinomial(p, n, replacement=True)
    prior_one_hot = one_hot(prior_idxs, num_classes=d).float()

    if blur is not None:
        # Model's sometimes have issues moving from integers. Adding noise
        # and re-projecting onto the simplex creates a softer version of the
        # categorical distribution helping models deal with this better.
        prior_one_hot = prior_one_hot + torch.randn_like(prior_one_hot) * blur
        # The division by (1/d) is a temperature scaling that makes the distribution
        # sharper, counteracting the effect of adding noise,
        # ensuring that the distribution remains sharp for the one-hot samples.
        prior_one_hot = softmax(prior_one_hot / (1 / d), dim=1)

    return prior_one_hot


def sample_p_c_given_a(
    n: int,
    d: int,
    atom_types: torch.Tensor,
    p_c_given_a: torch.Tensor,
    blur: float = None,
):
    """
    Sample from the conditional distribution of charges given atom type, p(c|a).

    Charges are more likely to be certain values given the atom type
    (e.g. carbons are more likely to be neutral, while oxygens are more likely to be negative).
    This prior captures this relationship by sampling charges conditioned on the atom types of the nodes.

    Look at marginal for more details on the arguments. The additional arguments are:
        - atom_types: a tensor of shape (n, num_atom_types) representing the
            atom types of the nodes, one-hot encoded
        - p_c_given_a: a tensor of shape (num_atom_types, num_charge_categories) representing
            the conditional probabilities of charges given atom types
    """
    if p_c_given_a.device != atom_types.device:
        p_c_given_a = p_c_given_a.to(atom_types.device)

    atom_type_idxs = atom_types.argmax(dim=1)
    charge_idxs = torch.multinomial(
        p_c_given_a[atom_type_idxs], 1, replacement=True
    ).squeeze(-1)

    charge_simplex = one_hot(charge_idxs, num_classes=d).float()

    if blur is not None:
        charge_simplex = charge_simplex + torch.randn_like(charge_simplex) * blur
        charge_simplex = softmax(charge_simplex / (1 / d), dim=1)

    return charge_simplex


def ctmc_masked_prior(n: int, d: int):
    """
    Sample from a Continuous Time Markov Chain (CTMC) masked prior.
    All samples are assigned the mask token at t=0, which is equivalent to
    assigning every sample to the same ``I don't know category''.
    """
    p = torch.full((n,), fill_value=d)
    p = one_hot(p, num_classes=d + 1).float()
    return p


def align_prior(
    prior_feat: torch.Tensor,
    dst_feat: torch.Tensor,
    permutation=False,
    rigid_body=False,
    n_alignments: int = 1,
):
    """
    Aligns a prior feature to a destination feature.
    """
    for _ in range(n_alignments):
        if permutation:
            # solve assignment problem
            cost_mat = torch.cdist(dst_feat, prior_feat, p=2)
            _, prior_idx = linear_sum_assignment(cost_mat)

            # reorder prior to according to optimal assignment
            prior_feat = prior_feat[prior_idx]

        if rigid_body:
            # perform rigid alignment
            prior_feat = rigid_alignment(prior_feat, dst_feat)

    return prior_feat


def rigid_alignment(x_0, x_1, pre_centered=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Alignment of two point clouds using the Kabsch algorithm.
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    d = x_0.shape[1]
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    # remove COM from data and record initial COM
    if pre_centered:
        x_0_mean = torch.zeros(1, d)
        x_1_mean = torch.zeros(1, d)
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=0, keepdim=True)
        x_1_mean = x_1.mean(dim=0, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix
    H = x_0_c.T.mm(x_1_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    if pre_centered:
        t = torch.zeros(1, d)
    else:
        t = x_1_mean - R.mm(x_0_mean.T).T  # has shape (1, D)

    # apply rotation to x_0_c
    x_0_aligned = x_0_c.mm(R.T)

    # move x_0_aligned to its original frame
    x_0_aligned = x_0_aligned + x_0_mean

    # apply the translation
    x_0_aligned = x_0_aligned + t

    return x_0_aligned


def batched_rigid_alignment(x_0, x_1, pre_centered=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Alignment of two point clouds using the Kabsch algorithm.
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    print(
        "WARNING: batched_rigid_alignment is currently broken (gives incorrect results)"
    )
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    if len(x_0.shape) == 2:
        n, d = x_0.shape
        b = 1
        x_0 = x_0.unsqueeze(0)
        x_1 = x_1.unsqueeze(0)

    elif len(x_0.shape) == 3:
        b, n, d = x_0.shape

    # remove COM from data and record initial COM
    if pre_centered:
        x_0_mean = torch.zeros(b, 1, d)
        x_1_mean = torch.zeros(b, 1, d)
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=1, keepdim=True)
        x_1_mean = x_1.mean(dim=1, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix
    # x_0_c has shape (b, n, d) as does x_1_c
    # H shold have shape (b, d, d)
    # below is the line for the unbatched version, followed by the batched version
    # H = x_0_c.T.mm(x_1_c)
    H = torch.einsum("bnd,bnm->bdm", x_0_c, x_1_c)

    U, S, V = torch.svd(H)
    # Rotation matrix
    # U and V both have shape (b, d, d)
    # R should have shape (b, d, d)
    # below is the line for the unbatched version, followed by the batched version
    # R = V.mm(U.T)
    R = torch.einsum("bxy,bjk->bxj", V, U)

    # Translation vector
    if pre_centered:
        t = torch.zeros(b, 1, d)
    else:
        # R has shape (b, d, d)
        # x_0_mean has shape (b, 1, d)
        # t = x_1_mean - R.mm(x_0_mean.T).T # has shape (b, 1, D)
        t = x_1_mean - torch.einsum("bxy,bjk->bjy", R, x_0_mean)

    # apply rotation to x_0_c
    # x_0_c has shape (b, n, d)
    # R has shape (b, d, d)
    # x_0_aligned should have shape (b, n, d)
    # below is the line for the unbatched version, followed by the batched version
    # x_0_aligned = x_0_c.mm(R.T)
    x_0_aligned = torch.einsum("bxy,bjk->bxk", x_0_c, R)

    # move x_0_aligned to its original frame
    x_0_aligned = x_0_aligned + x_0_mean

    # apply the translation
    x_0_aligned = x_0_aligned + t

    return x_0_aligned


TRAIN_PRIOR_REGISTER = {
    "centered-normal": centered_normal_prior,
    "uniform-simplex": uniform_simplex_prior,
    "biased-simplex": biased_simplex_prior,
    "marginal": sample_marginal,
    "c-given-a": sample_p_c_given_a,
    "gaussian": gaussian,
    "barycenter": barycenter_prior,
    "ctmc": ctmc_masked_prior,
}

INFERENCE_PRIOR_REGISTER = {
    "centered-normal": centered_normal_prior_batched_graph,
    "uniform-simplex": uniform_simplex_prior,
    "biased-simplex": biased_simplex_prior,
    "marginal": sample_marginal,
    "c-given-a": sample_p_c_given_a,
    "gaussian": gaussian,
    "barycenter": barycenter_prior,
    "ctmc": ctmc_masked_prior,
}


@torch.no_grad()
def coupled_node_prior(dst_dict: dict, prior_config: dict):
    prior_dict = {}

    for feat in dst_dict.keys():

        # get the prior configuration for this feature
        feat_prior_config = prior_config[feat]

        # get destination features (t=1)
        dst_feat = dst_dict[feat]

        # sample prior
        prior_fn = TRAIN_PRIOR_REGISTER[feat_prior_config["type"]]
        n, d = dst_feat.shape
        args = [n, d]

        # if sampling the charges conditioned on atom type, we need to pass the atom types to the prior function
        # note that this behavior is dependent on "a" being encountered in this loop before "c"
        if feat == "c" and feat_prior_config["type"] == "c-given-a":
            args.append(prior_dict["a"])

        prior_feat = prior_fn(*args, **feat_prior_config["kwargs"])

        # align prior to destination if necessary
        if feat_prior_config["align"]:

            if feat == "x":
                rigid_body = True
            else:
                rigid_body = False

            prior_feat = align_prior(
                prior_feat, dst_feat, permutation=True, rigid_body=rigid_body
            )

        prior_dict[feat] = prior_feat

    return prior_dict


def edge_prior(
    upper_edge_mask: torch.Tensor,
    edge_prior_config: dict,
    explicit_aromaticity: bool = False,
):

    n_upper_edges = upper_edge_mask.sum().item()
    prior_fn = TRAIN_PRIOR_REGISTER[edge_prior_config["type"]]

    n_non_mask_bond_types = 5 if explicit_aromaticity else 4
    upper_edge_prior = prior_fn(
        n_upper_edges, n_non_mask_bond_types, **edge_prior_config["kwargs"]
    )

    edge_prior = torch.zeros(upper_edge_mask.shape[0], upper_edge_prior.shape[1])
    edge_prior[upper_edge_mask] = upper_edge_prior
    edge_prior[~upper_edge_mask] = upper_edge_prior
    return edge_prior
