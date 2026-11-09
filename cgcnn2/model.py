import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    Crystal graph convolution: updates the features of every atom from its
    neighbors and the bonds to them.

    Args:
        atom_fea_len (int): Number of hidden atom features.
        nbr_fea_len (int): Number of bond features.
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int) -> None:
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus = nn.Softplus()

    def forward(
        self,
        atom_in_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        `N` is the number of atoms in the batch and `M` the number of neighbors per atom.

        Args:
            atom_in_fea (torch.Tensor): shape (N, atom_fea_len), atom features before the convolution.
            nbr_fea (torch.Tensor): shape (N, M, nbr_fea_len), bond features of each atom's neighbors.
            nbr_fea_idx (torch.LongTensor): shape (N, M), indices of each atom's neighbors.

        Returns:
            torch.Tensor: shape (N, atom_fea_len), atom features after the convolution.
        """
        n_atoms, n_nbr = nbr_fea_idx.shape
        # fc_full acts on the concatenation [atom_i, atom_j, bond_ij]. Applying its
        # three weight blocks separately gives the same result while the two atom
        # terms are computed once per atom instead of once per neighbor pair, and
        # the (N, M, 2 * atom_fea_len + nbr_fea_len) concatenation never exists.
        # index_select gathers the neighbor term with a deterministic backward.
        w_self, w_nbr, w_bond = self.fc_full.weight.split(
            [self.atom_fea_len, self.atom_fea_len, self.nbr_fea_len], dim=1
        )
        nbr_term = F.linear(atom_in_fea, w_nbr).index_select(0, nbr_fea_idx.reshape(-1))
        gated_fea = (
            F.linear(atom_in_fea, w_self, self.fc_full.bias).unsqueeze(1)
            + nbr_term.view(n_atoms, n_nbr, -1)
            + F.linear(nbr_fea, w_bond)
        )
        gated_fea = self.bn1(gated_fea.view(-1, 2 * self.atom_fea_len)).view(
            n_atoms, n_nbr, 2 * self.atom_fea_len
        )
        nbr_filter, nbr_core = gated_fea.chunk(2, dim=2)
        nbr_summed = (torch.sigmoid(nbr_filter) * self.softplus(nbr_core)).sum(dim=1)
        return self.softplus(atom_in_fea + self.bn2(nbr_summed))


class CrystalGraphConvNet(nn.Module):
    """
    Crystal graph convolutional neural network for predicting a material property.

    Args:
        orig_atom_fea_len (int): Number of atom features in the input.
        nbr_fea_len (int): Number of bond features.
        atom_fea_len (int): Number of hidden atom features in the convolutional layers.
        n_conv (int): Number of convolutional layers.
        h_fea_len (int): Number of hidden features after pooling.
        n_h (int): Number of hidden layers after pooling.
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.fcs = nn.ModuleList(
            [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
        )
        self.fc_out = nn.Linear(h_fea_len, 1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
        crystal_atom_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        `N` is the number of atoms in the batch, `M` the number of neighbors per atom
        and `B` the number of crystals.

        Args:
            atom_fea (torch.Tensor): shape (N, orig_atom_fea_len), atom features from the atom type.
            nbr_fea (torch.Tensor): shape (N, M, nbr_fea_len), bond features of each atom's neighbors.
            nbr_fea_idx (torch.LongTensor): shape (N, M), indices of each atom's neighbors.
            crystal_atom_idx (torch.LongTensor): shape (N,), the crystal of each atom, sorted ascending.

        Returns:
            out (torch.Tensor): shape (B, 1), the predicted property.
            crys_fea (torch.Tensor): shape (B, h_fea_len), the crystal features before the output layer.
        """
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.softplus(self.pooling(atom_fea, crystal_atom_idx))
        crys_fea = self.softplus(self.conv_to_fc(crys_fea))
        for fc in self.fcs:
            crys_fea = self.softplus(fc(crys_fea))
        return self.fc_out(crys_fea), crys_fea

    def pooling(
        self, atom_fea: torch.Tensor, crystal_atom_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean-pool atom features into crystal features.

        Args:
            atom_fea (torch.Tensor): shape (N, atom_fea_len), the atom embeddings of the batch.
            crystal_atom_idx (torch.LongTensor): shape (N,), the crystal of each atom, sorted ascending.

        Returns:
            torch.Tensor: shape (B, atom_fea_len), the mean embedding of each crystal.
        """
        n_crystals = int(crystal_atom_idx[-1]) + 1
        index = crystal_atom_idx.unsqueeze(1).expand(-1, atom_fea.shape[1])
        return atom_fea.new_zeros(n_crystals, atom_fea.shape[1]).scatter_reduce_(
            0, index, atom_fea, reduce="mean", include_self=False
        )
