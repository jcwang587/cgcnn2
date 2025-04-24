""" e3cgcnn model using e3nn equivariant convolutions (fixed SH attributes). """

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, SphericalHarmonics


class E3ConvLayer(nn.Module):
    """One equivariant convolutional layer built with e3nn (pairwise)."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int, lmax: int = 2):
        super().__init__()

        # each atom carries `atom_fea_len` scalar channels (0e)
        self.irreps_in = Irreps(f"{atom_fea_len}x0e")

        # real spherical harmonics up to L
        self.sh = SphericalHarmonics(lmax, normalize=True)
        sh_dim = self.sh.irreps_out.dim  # dimension of Y_lm block

        # map radial neighbour features → coefficients per Y_lm component
        self.radial_mlp = nn.Sequential(
            nn.Linear(nbr_fea_len, nbr_fea_len),
            nn.Softplus(),
            nn.Linear(nbr_fea_len, sh_dim),
        )

        # tensor product: (h_j ⊗ W_ij) → scalar channels
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,        # neighbour features (0e)
            self.sh.irreps_out,    # geometry irreps (0e+1o+…)
            self.irreps_in,        # output back to scalars
        )

    def forward(
        self,
        atom_fea: torch.Tensor,          # (N, C)
        nbr_fea: torch.Tensor,           # (N, M, F_r)
        nbr_idx: torch.LongTensor,       # (N, M)
        pos: torch.Tensor,               # (N, 3)
    ) -> torch.Tensor:
        N, M, _ = nbr_fea.shape

        # gather neighbour features h_j
        h_j = atom_fea[nbr_idx.view(-1)].view(N * M, -1)
        # radial coefficients
        R = self.radial_mlp(nbr_fea.view(N * M, -1))  # (N*M, sh_dim)

        # spherical harmonics Y_lm(r̂_ij)
        dst = torch.arange(N, device=atom_fea.device).unsqueeze(1).expand(N, M).reshape(-1)
        src = nbr_idx.view(-1)
        Y = self.sh(pos[src] - pos[dst])  # (N*M, sh_dim)

        W = R * Y  # fuse radial & angular

        # tensor product and aggregate
        msg = self.tp(h_j, W)
        atom_out = scatter(msg, dst, dim=0, dim_size=N, reduce="mean")
        return atom_out


class CrystalGraphE3ConvNet(nn.Module):
    """CGCNN backbone with E(3)-equivariant convolutional layers."""

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
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_idx: torch.LongTensor,
        crystal_atom_idx: list[torch.LongTensor],
        pos: torch.Tensor,
    ):
        x = self.embedding(atom_fea)
        for conv in self.convs:
            x = conv(x, nbr_fea, nbr_idx, pos)

        # average pool per crystal
        crys = torch.stack([x[idx].mean(dim=0) for idx in crystal_atom_idx], dim=0)
        h = self.softplus(self.conv_to_fc(crys))

        if hasattr(self, "fcs"):
            for fc, sp in zip(self.fcs, self.softpluses):
                h = sp(fc(h))

        out = self.fc_out(h)
        if self.classification:
            out = self.logsoftmax(out)
        return out, h
