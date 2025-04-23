""" e3cgcnn model using e3nn equivariant convolutions. """

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, SphericalHarmonics

class E3ConvLayer(nn.Module):
    """
    Equivariant convolutional layer using e3nn.
    """
    def __init__(self, atom_fea_len: int, nbr_fea_len: int, lmax: int = 2):
        super().__init__()
        # Scalar irreps for atom features
        self.irreps_in = Irreps(f"{atom_fea_len}x0e")
        # Use same irreps for spherical harmonics
        self.sh = SphericalHarmonics(lmax, self.irreps_in, normalize=True)
        # Radial MLP: map neighbor radial features to spherical weights
        self.radial_mlp = nn.Sequential(
            nn.Linear(nbr_fea_len, nbr_fea_len),
            nn.Softplus(),
            nn.Linear(nbr_fea_len, self.sh.irreps_out.dim),
        )
        # Tensor product to combine features and spherical harmonics
        self.tp = FullyConnectedTensorProduct(self.irreps_in, self.sh.irreps_out, self.irreps_in)
        self.scale = (1.0)**0.5

    def forward(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor, nbr_idx: torch.LongTensor, pos: torch.Tensor = None):
        """
        atom_fea: [N, atom_fea_len]
        nbr_fea: [N, M, nbr_fea_len] - radial features (e.g. GaussianDistance)
        nbr_idx: [N, M] - neighbor indices
        pos: [N, 3] - atomic coordinates for spherical harmonics (optional)
        """
        N, M, _ = nbr_fea.shape
        # Gather neighbor features
        neigh = atom_fea[nbr_idx.view(-1)].view(N, M, -1)
        neigh = neigh.view(N*M, -1)
        radial = nbr_fea.view(N*M, -1)
        # Spherical harmonics
        if pos is not None:
            idx_center = torch.arange(N, device=atom_fea.device).unsqueeze(1).expand(-1, M).reshape(-1)
            rel_vec = pos[nbr_idx.view(-1)] - pos[idx_center]
            Y = self.sh(rel_vec)
        else:
            Y = torch.ones(radial.shape[0], self.sh.irreps_out.dim, device=atom_fea.device)
        # Radial weights
        R = self.radial_mlp(radial)
        W = Y * R
        # Message tensor product
        msg = self.tp(neigh, W)
        # Aggregate messages
        idx_i = nbr_idx.view(-1)
        out = scatter(msg, idx_i, dim=0, dim_size=N, reduce="mean")
        return out * self.scale

class CrystalGraphE3ConvNet(nn.Module):
    """
    CGCNN modified to use e3nn equivariant convolutions.
    """
    def __init__(self, orig_atom_fea_len: int, nbr_fea_len: int, atom_fea_len: int=64,
                 n_conv: int=3, h_fea_len: int=128, n_h: int=1, classification: bool=False):
        super().__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # e3nn convolutional layers
        self.convs = nn.ModuleList([
            E3ConvLayer(atom_fea_len, nbr_fea_len)
            for _ in range(n_conv)
        ])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len, 2 if classification else 1)
        if classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_idx, crystal_atom_idx, pos=None):
        # atom_fea: [N, orig_atom_fea_len]
        # nbr_fea: [N, M, nbr_fea_len], nbr_idx: [N, M]
        x = self.embedding(atom_fea)
        for conv in self.convs:
            x = conv(x, nbr_fea, nbr_idx, pos)
        # Pool to crystal level
        pooled = [x[idx].mean(dim=0) for idx in crystal_atom_idx]
        crys = torch.stack(pooled, dim=0)
        h = self.softplus(self.conv_to_fc(crys))
        if hasattr(self, 'fcs'):
            for fc, sp in zip(self.fcs, self.softpluses):
                h = sp(fc(h))
        out = self.fc_out(h)
        if self.classification:
            out = self.logsoftmax(out)
        return out, h
