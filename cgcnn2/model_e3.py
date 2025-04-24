"""Equivariant CGCNN (E3CGCNN) using e3nn.

This module defines:
- `E3ConvLayer`: pair‑wise E(3)‑equivariant convolution with residual skip,
  batch‑norm and Softplus, closely matching the original CGCNN behaviour.
- `CrystalGraphE3ConvNet`: CGCNN backbone where every convolution is replaced
  by `E3ConvLayer`.  If `pos` (Cartesian coordinates) are not supplied the
  network gracefully falls back to invariant messages (behaves like CGCNN).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, SphericalHarmonics


class E3ConvLayer(nn.Module):
    """Single equivariant convolutional layer with residual connection."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int, lmax: int = 2):
        super().__init__()
        # scalar irreps for node features
        self.irreps_scalar = Irreps(f"{atom_fea_len}x0e")
        # real spherical harmonics basis Y_lm up to lmax
        self.sh = SphericalHarmonics(lmax, normalize=True)
        sh_dim = self.sh.irreps_out.dim

        # radial MLP: neighbour radial features → coefficient per Y_lm
        self.radial_mlp = nn.Sequential(
            nn.Linear(nbr_fea_len, nbr_fea_len),
            nn.Softplus(),
            nn.Linear(nbr_fea_len, sh_dim),
        )

        # tensor product: (scalar ⊗ Y_lm) → scalar
        self.tp = FullyConnectedTensorProduct(
            self.irreps_scalar,      # h_j
            self.sh.irreps_out,      # W_ij
            self.irreps_scalar,      # output scalars
        )

        # post‑aggregation normalisation + non‑linearity
        self.bn = nn.BatchNorm1d(atom_fea_len)
        self.act = nn.Softplus()

    def forward(
        self,
        atom_fea: torch.Tensor,            # (N, C)
        nbr_fea: torch.Tensor,             # (N, M, F_r)
        nbr_idx: torch.LongTensor,         # (N, M)
        pos: torch.Tensor | None = None,   # (N, 3) optional
    ) -> torch.Tensor:
        N, M, _ = nbr_fea.shape
        src = nbr_idx.view(-1)  # neighbour indices j
        dst = (
            torch.arange(N, device=atom_fea.device)
            .view(-1, 1)
            .expand(N, M)
            .reshape(-1)
        )  # central atom indices i

        h_j = atom_fea[src]  # (N*M, C)
        R = self.radial_mlp(nbr_fea.view(N * M, -1))  # (N*M, sh_dim)

        # geometric tensor Y_lm
        if pos is not None:
            Y = self.sh(pos[src] - pos[dst])  # (N*M, sh_dim)
        else:
            Y = torch.ones_like(R)            # invariant fallback

        W = R * Y
        msg = self.tp(h_j, W)                # (N*M, C)

        # scatter mean aggregation to central atoms
        agg = scatter(msg, dst, dim=0, dim_size=N, reduce="mean")

        # residual + BN + non‑linearity
        out = self.act(self.bn(atom_fea + agg))
        return out


class CrystalGraphE3ConvNet(nn.Module):
    """CGCNN architecture with E(3)‑equivariant convolutions."""

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        classification: bool = False,
    ) -> None:
        super().__init__()
        self.classification = classification

        # input embedding to hidden scalar space
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # stack of equivariant conv layers
        self.convs = nn.ModuleList(
            [E3ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )

        # crystal‑level pooling → fully connected
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        self.fc_out = nn.Linear(h_fea_len, 2 if classification else 1)
        if classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(
        self,
        atom_fea: torch.Tensor,                    # (N, C0)
        nbr_fea: torch.Tensor,                     # (N, M, F_r)
        nbr_idx: torch.LongTensor,                 # (N, M)
        crystal_atom_idx: list[torch.LongTensor],  # length = n_crystals
        pos: torch.Tensor | None = None,           # (N, 3) optional
    ):
        x = self.embedding(atom_fea)
        for conv in self.convs:
            x = conv(x, nbr_fea, nbr_idx, pos)

        # average pooling per crystal structure
        crys_fea = torch.stack([x[idx].mean(dim=0) for idx in crystal_atom_idx], dim=0)
        h = self.softplus(self.conv_to_fc(crys_fea))

        if hasattr(self, "fcs"):
            for fc, sp in zip(self.fcs, self.softpluses):
                h = sp(fc(h))

        if self.classification:
            h = self.dropout(h)
            out = self.logsoftmax(self.fc_out(h))
        else:
            out = self.fc_out(h)

        return out, h
