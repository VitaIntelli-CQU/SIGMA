from scipy import stats
import warnings
import torch
from tqdm import tqdm
from sigma.neural.architecture import SIGMA
from sigma.neural.layers import SAGEConv_Decoder, SAGEConv_Encoder
from sigma.training.objectives import (
    configure_deterministic_backend as _configure_deterministic_backend,
    cross_modal_contrastive_loss,
    laplacian_regularization,
    modality_entropy_regularization,
    set_training_seed as _set_training_seed,
)
import torch.nn.functional as F
import numpy as np


def train_SIGMA(
    features,
    edges,
    triplet_samples_list,
    weights=[1, 1],
    emb_dim=64,
    n_epochs=500,
    lr=0.0001,
    weight_decay=1e-5,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    window_size=20,
    slope=0.0001,
    Conv_Encoder=SAGEConv_Encoder,
    Conv_Decoder=SAGEConv_Decoder,
    margin=0.5,
    return_loss=False,
    laplacian_alpha=0,
    gate_hidden_dim=None,
    modality_dropout=0.0,
    entropy_alpha=0.0,
    return_attention=False,
    seed=None,
    strict_reproducibility=False,
    contrastive_alpha=0.16,
    contrastive_temperature=0.5,
    alpha=0.05,
):
    """
    Train SIGMA with sample-wise dynamic fusion.

    Parameters
    ----------
    features : list of torch.Tensor
        Node feature matrices for each modality.
    edges : list of torch.LongTensor
        Graph connectivity for each modality.
    triplet_samples_list : list of tuple
        Triplets of (anchors, positives, negatives) indices.
    weights : list of float, default=[1, 1]
        Loss weights ordered as:
        [reconstruction weights..., triplet weights...].
    emb_dim : int, default=64
        Shared latent embedding size.
    n_epochs : int, default=500
        Number of training epochs.
    lr : float, default=0.0001
        Adam learning rate.
    weight_decay : float, default=1e-5
        Adam weight decay.
    device : torch.device, optional
        Target device.
    window_size : int, default=20
        Window size used for early stopping slope detection.
    slope : float, default=0.0001
        Minimum absolute slope threshold for continuing training.
    Conv_Encoder : class, default=SAGEConv_Encoder
        Graph encoder class.
    Conv_Decoder : class, default=SAGEConv_Decoder
        Graph decoder class.
    margin : float, default=0.5
        Margin for triplet loss.
    return_loss : bool, default=False
        Whether to return loss history.
    laplacian_alpha : float, default=0
        Weight of Laplacian regularization.
    gate_hidden_dim : int, optional
        Hidden size inside the gate MLP.
    modality_dropout : float, default=0.0
        Probability of masking a modality during training.
    entropy_alpha : float, default=0.0
        Weight of anti-collapse entropy regularization on modality weights.
    return_attention : bool, default=False
        Whether to also return the final modality weights after training.
    seed : int, optional
        If provided, re-seed Python, NumPy, and PyTorch at the start of this
        training call so repeated runs in the same notebook session start from
        the same RNG state.
    strict_reproducibility : bool, default=False
        If True, enforce deterministic backend flags and run on CPU to avoid
        CUDA/PyG non-determinism during graph aggregation.
    contrastive_alpha : float, default=0.1
        Weight of the cross-modal contrastive (InfoNCE) loss. Set to 0 to
        disable. Higher values enforce stronger alignment between modalities.
    contrastive_temperature : float, default=0.1
        Temperature parameter tau for the InfoNCE loss. Smaller values make
        the distribution sharper and impose stronger alignment pressure.
    """
    if seed is not None:
        _set_training_seed(seed)

    deterministic_device = _configure_deterministic_backend(
        strict_reproducibility=strict_reproducibility
    )
    if deterministic_device is not None and device.type != deterministic_device.type:
        warnings.warn(
            "strict_reproducibility=True switches SIGMA training to CPU "
            "because PyG graph aggregation on CUDA can remain non-deterministic.",
            RuntimeWarning,
        )
        device = deterministic_device

    hidden_dims = [x.shape[1] for x in features] + [emb_dim]

    model = SIGMA(
        hidden_dims=hidden_dims,
        device=device,
        Conv_Encoder=Conv_Encoder,
        Conv_Decoder=Conv_Decoder,
        gate_hidden_dim=gate_hidden_dim,
    )

    features = [x.to(device) for x in features]
    edges = [edge.to(device) for edge in edges]
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_list = []

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, x_rec, modality_weights, projected_modalities = model(
            features, edges, return_projections=True, alpha=alpha
        )

        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_loss = 0
        for i, (anchors, positives, negatives) in enumerate(triplet_samples_list):
            anchor_arr = z[anchors]
            positive_arr = z[positives]
            negative_arr = z[negatives]

            tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
            w = weights[len(weights) // 2 + i]
            tri_loss += w * tri_output

        rec_loss = 0
        for i, (feature, x_r) in enumerate(zip(features, x_rec)):
            rec_output = F.mse_loss(feature, x_r)
            w = weights[i]
            rec_loss += w * rec_output

        loss = rec_loss + tri_loss
        con_loss = torch.tensor(0.0, device=device)

        if contrastive_alpha != 0:
            con_loss = cross_modal_contrastive_loss(
                projected_modalities, temperature=contrastive_temperature
            )
            loss += contrastive_alpha * con_loss

        if laplacian_alpha != 0:
            loss += laplacian_alpha * laplacian_regularization(z, edges[0])

        entropy_loss = torch.tensor(0.0, device=device)

        if epoch > window_size and epoch % 10 == 0:
            if epoch < int(n_epochs * 0.4):
                continue
            x_axis = np.arange(window_size)
            res1 = stats.linregress(x_axis, [i[1] for i in loss_list[-window_size:]])
            res2 = stats.linregress(x_axis, [i[2] for i in loss_list[-window_size:]])
            tri_flat = abs(res1.slope) < slope and res1.slope != 0
            rec_flat = abs(res2.slope) < slope and res2.slope != 0
            if tri_flat and rec_flat:
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"tri_slope={res1.slope:.6f}, rec_slope={res2.slope:.6f}"
                )
                break

        loss_list.append(
            (
                loss.item(),
                tri_loss.item(),
                rec_loss.item(),
                entropy_loss.item(),
                con_loss.item(),
            )
        )
        loss.backward()
        optimizer.step()

    return model if not return_loss else (model, loss_list)


def fit_sigma_embedding(
    adata,
    features,
    edges,
    triplet_samples_list,
    embedding_key="SIGMA",
    move_model_to=None,
    forward_kwargs=None,
    **train_kwargs,
):
    """
    Train SIGMA and store the resulting embedding in an AnnData object.
    """
    model = train_SIGMA(
        features=features,
        edges=edges,
        triplet_samples_list=triplet_samples_list,
        **train_kwargs,
    )
    if move_model_to is not None:
        model = model.to(move_model_to)

    if forward_kwargs is None:
        embedding = model(features, edges)[0]
    else:
        embedding = model(features, edges, **forward_kwargs)[0]

    adata.obsm[embedding_key] = embedding.cpu().detach().numpy()
    return model


