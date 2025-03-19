"""
This module implements the Crystal Graph Convolutional Neural Network (CGCNN) for predicting
material properties based on their crystal structures.

Classes:
    ConvLayer: Convolutional layer for graph data.
    MaskedConvLayer: Convolutional layer with masking for padding indices.
    CrystalGraphConvNet: CGCNN model for predicting material properties.
    Normalizer: Utility class for normalizing tensors.

Usage:
    Define your model by creating an instance of CrystalGraphConvNet with the desired parameters.
    Use the Normalizer class to normalize your target properties during training.
"""

import torch
import torch.nn as nn
from typing import Dict


class ConvLayer(nn.Module):
    """
    Convolutional layer for graph data.

    Performs a convolutional operation on graphs, updating atom features based on their neighbors.
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize the ConvLayer.

        Args:
            atom_feature_len (int): Number of atom hidden features.
            neighbor_feature_len (int): Number of bond (neighbor) features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out, crys_fea

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert (
            sum([len(idx_map) for idx_map in crystal_atom_idx])
            == atom_fea.data.shape[0]
        )
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)


class Normalizer:
    """
    Normalizes a PyTorch tensor and allows restoring it later.

    This class keeps track of the mean and standard deviation of a tensor and provides methods
    to normalize and denormalize tensors using these statistics.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Initialize the Normalizer with a sample tensor to calculate mean and standard deviation.

        Args:
            tensor (torch.Tensor): Sample tensor to compute mean and standard deviation.
        """
        self.mean: torch.Tensor = torch.mean(tensor)
        self.std: torch.Tensor = torch.std(tensor)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor using the stored mean and standard deviation.

        Args:
            tensor (torch.Tensor): Tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor using the stored mean and standard deviation.

        Args:
            normed_tensor (torch.Tensor): Normalized tensor to denormalize.

        Returns:
            torch.Tensor: Denormalized tensor.
        """
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Returns the state dictionary containing the mean and standard deviation.

        Returns:
            Dict[str, torch.Tensor]: State dictionary.
        """
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Loads the mean and standard deviation from a state dictionary.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dictionary containing 'mean' and 'std'.
        """
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
