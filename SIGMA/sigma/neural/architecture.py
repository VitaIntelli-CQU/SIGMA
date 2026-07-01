import torch
from torch import nn
from sigma.neural.layers import (
    SAGEConv_Encoder,
    SAGEConv_Decoder,
    GCNConv_Encoder,
    GCNConv_Decoder,
    GCN2Conv_Encoder,
    GCN2Conv_Decoder,
    GATConv_Encoder,
    GATConv_Decoder,
    GraphConv_Encoder,
    GraphConv_Decoder,
)


class SIGMA(torch.nn.Module):
    """
    SIGMA variant with sample-wise dynamic fusion.

    Each modality is first encoded independently, projected into a shared
    latent space, and then fused with sample-specific softmax weights learned
    by a lightweight gating network. The fused embedding is decoded back to
    each modality for reconstruction.

    Parameters
    ----------
    hidden_dims : list of int
        List specifying input dimensions of each modality and shared hidden
        dimension. Example: [in_dim_mod1, in_dim_mod2, ..., latent_dim].
    device : torch.device
        Device to place model modules on.
    Conv_Encoder : class
        Encoder architecture (default: SAGEConv_Encoder).
    Conv_Decoder : class
        Decoder architecture (default: SAGEConv_Decoder).
    gate_hidden_dim : int, optional
        Hidden size used by the modality gate MLP. Defaults to half of the
        latent dimension, with a minimum of 1.
    """

    def __init__(
        self,
        hidden_dims,
        device,
        Conv_Encoder=SAGEConv_Encoder,
        Conv_Decoder=SAGEConv_Decoder,
        gate_hidden_dim=None,
    ):
        super(SIGMA, self).__init__()
        out_dim = hidden_dims[-1]
        num_modalities = len(hidden_dims) - 1
        gate_hidden_dim = gate_hidden_dim or max(out_dim//2, 1)

        self.encoders = self._build_encoders(hidden_dims, out_dim, device, Conv_Encoder)
        self.proj_layers = self._build_projection_layers(out_dim, num_modalities, device)
        self.gate_layers = self._build_gate_layers(out_dim, gate_hidden_dim, num_modalities, device)
        self.decoders = self._build_decoders(hidden_dims, out_dim, device, Conv_Decoder)
        self.context_mlp = self._build_context_mlp(out_dim)

    def _build_encoders(self, hidden_dims, out_dim, device, Conv_Encoder):
        return nn.ModuleList(
            [Conv_Encoder(in_dim, out_dim).to(device) for in_dim in hidden_dims[:-1]]
        )

    def _build_projection_layers(self, out_dim, num_modalities, device):
        return nn.ModuleList(
            [nn.Linear(out_dim, out_dim).to(device) for _ in range(num_modalities)]
        )

    def _build_gate_layers(self, out_dim, gate_hidden_dim, num_modalities, device):
        return nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(out_dim, gate_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gate_hidden_dim, 1),
                ).to(device)
                for _ in range(num_modalities)
            ]
        )

    def _build_decoders(self, hidden_dims, out_dim, device, Conv_Decoder):
        return nn.ModuleList(
            [Conv_Decoder(out_dim, in_dim).to(device) for in_dim in hidden_dims[:-1]]
        )

    def _build_context_mlp(self, out_dim):
        return nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, features, edge_indexs, return_attention=False, return_projections=False,alpha=0.05):
        """
        Forward pass of the SIGMA model.

        Parameters
        ----------
        features : list of torch.Tensor
            Node features for each modality. Each tensor shape:
            [num_nodes, in_dim_mod].
        edge_indexs : list of torch.LongTensor
            Graph connectivity for each modality. Each tensor shape:
            [2, num_edges].
        return_attention : bool, default=False
            Whether to also return sample-wise modality weights.
        return_projections : bool, default=False
            Whether to also return projected modality embeddings.
            Required when using cross-modal contrastive loss.

        Returns
        -------
        z : torch.Tensor
            Latent shared representation of shape [num_nodes, latent_dim].
        x_rec : list of torch.Tensor
            Reconstructed features for each modality.
        modality_weights : torch.Tensor, optional
            Sample-wise softmax-normalized modality weights of shape
            [num_nodes, num_modalities]. Only returned when
            `return_attention=True`.
        projected_modalities : list of torch.Tensor, optional
            Projected embeddings for each modality, before dynamic fusion.
            Only returned when `return_projections=True`.
        """
        encoded_modalities = [
            encoder(feature, edge_index)
            for encoder, feature, edge_index in zip(self.encoders, features, edge_indexs)
        ]

        projected_modalities = [
            proj(modality_embedding)
            for proj, modality_embedding in zip(self.proj_layers, encoded_modalities)
        ]

        gate_scores = [
            gate(modality_embedding)
            for gate, modality_embedding in zip(self.gate_layers, projected_modalities)
        ]

        
        gate_scores = torch.cat(gate_scores, dim=1)
        modality_weights = torch.softmax(gate_scores, dim=1)

        z = torch.zeros_like(projected_modalities[0])
        for modality_idx, modality_embedding in enumerate(projected_modalities):
            z = z + modality_weights[:, modality_idx:modality_idx + 1] * modality_embedding
        
        z_mean = torch.stack(encoded_modalities, dim=0).mean(dim=0)
        z_context = self.context_mlp(z_mean)
        z = z + z_context * alpha
        

        x_rec = [
            decoder(z, edge_index)
            for decoder, edge_index in zip(self.decoders, edge_indexs)
        ]

        if return_projections:
            return z, x_rec, modality_weights, projected_modalities
        if return_attention:
            return z, x_rec, modality_weights
        return z, x_rec

