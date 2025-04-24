"""E3CGCNN model (pair‑wise) built with e3nn.
Fixed compatibility: `pos` is optional; if coordinates are missing, the layer
falls back to invariant messages (Y=1) so the original CGCNN dataloader keeps
working, though without true equivariance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, SphericalHarmonics


class E3ConvLayer(nn.Module):
    """Single pair‑wise equivariant (or invariant) convolution using e3nn."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int, lmax: int = 2):
        super().__init__()
        # input features are scalars (0e) repeated `atom_fea_len` times
        self.irreps_in = Irreps(f"{atom_fea_len}x0e")
        # real spherical harmonics basis up to degree `lmax`
        self.sh = SphericalHarmonics(lmax, normalize=True)
        sh_dim = self.sh.irreps_out.dim  # number of Y_lm components

        # radial network maps neighbor radial features → coefficients for each Y_lm
        self.radial_mlp = nn.Sequential(
            nn.Linear(nbr_fea_len, nbr_fea_len),
            nn.Softplus(),
            nn.Linear(nbr_fea_len, sh_dim),
        )

        # tensor product combines neighbour features and geometric tensor, projecting
        # back to scalar irreps (0e)
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,        # h_j irreps
            self.sh.irreps_out,    # W_ij irreps
            self.irreps_in,        # result 0e
        )

    def forward(
        self,
        atom_fea: torch.Tensor,          # shape (N, C)
        nbr_fea: torch.Tensor,           # shape (N, M, F_r)
        nbr_idx: torch.LongTensor,       # shape (N, M)
        pos: torch.Tensor | None = None, # shape (N, 3) (optional)
    ) -> torch.Tensor:
        N, M, _ = nbr_fea.shape
        # flatten neighbour list indices j and i (dst)
        src = nbr_idx.view(-1)                                    # (N*M,)
        dst = torch.arange(N, device=atom_fea.device).unsqueeze(1).expand(N, M).reshape(-1)

        # gather neighbour features h_j and flatten radial features
        h_j = atom_fea[src]
        R = self.radial_mlp(nbr_fea.view(N * M, -1))              # (N*M, sh_dim)

        # geometry tensor Y_lm
        if pos is not None:
            Y = self.sh(pos[src] - pos[dst])                      # (N*M, sh_dim)
        else:
            # no coordinates provided ⇒ revert to invariant (CGCNN‑style) messages
            Y = torch.ones_like(R)

        W = R * Y                                                 # (N*M, sh_dim)
        msg = self.tp(h_j, W)                                     # (N*M, C)

        # scatter aggregate messages back to central atom i (=dst)
        out = scatter(msg, dst, dim=0, dim_size=N, reduce="mean")
        return out


class CrystalGraphE3ConvNet(nn.Module):
    """CGCNN architecture where each conv is e3nn‑based (falls back if `pos` missing)."""

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

        # initial embedding from element one‑hot/hand‑crafted features → hidden scalars
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [E3ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )

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
        atom_fea: torch.Tensor,                   # (N, C0)
        nbr_fea: torch.Tensor,                    # (N, M, F_r)
        nbr_idx: torch.LongTensor,                # (N, M)
        crystal_atom_idx: list[torch.LongTensor], # list of len N_crystals
        pos: torch.Tensor | None = None,          # (N, 3) optional
    ):
        x = self.embedding(atom_fea)
        for conv in self.convs:
            x = conv(x, nbr_fea, nbr_idx, pos)

        # average pooling over atoms belonging to each crystal structure
        crys = torch.stack([x[idx].mean(dim=0) for idx in crystal_atom_idx], dim=0)
        h = self.softplus(self.conv_to_fc(crys))

        if hasattr(self, "fcs"):
            for fc, sp in zip(self.fcs, self.softpluses):
                h = sp(fc(h))

        out = self.fc_out(h)
        if self.classification:
            out = self.logsoftmax(out)
        return out, h
