import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class ConvLayer(nn.Module):
    """
    Convolutional layer for graph data.

    Performs a convolutional operation on graphs, updating atom features based on their neighbors.
    """

    def __init__(self, atom_feature_len: int, neighbor_feature_len: int):
        """
        Initialize the ConvLayer.

        Args:
            atom_feature_len (int): Number of atom hidden features.
            neighbor_feature_len (int): Number of bond (neighbor) features.
        """
        super(ConvLayer, self).__init__()
        self.atom_feature_len = atom_feature_len
        self.neighbor_feature_len = neighbor_feature_len

        self.fc_full = nn.Linear(
            2 * self.atom_feature_len + self.neighbor_feature_len,
            2 * self.atom_feature_len,
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.batch_norm1 = nn.BatchNorm1d(2 * self.atom_feature_len)
        self.batch_norm2 = nn.BatchNorm1d(self.atom_feature_len)
        self.softplus2 = nn.Softplus()

    def forward(
        self,
        atom_in_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward pass for the convolutional layer.

        Args:
            atom_in_fea (torch.Tensor): Input atom features with shape (N, atom_feature_len),
                where N is the total number of atoms in the batch.
            nbr_fea (torch.Tensor): Neighbor (bond) features with shape (N, M, neighbor_feature_len),
                where M is the maximum number of neighbors.
            nbr_fea_idx (torch.LongTensor): Indices of M neighbors for each atom, shape (N, M).

        Returns:
            torch.Tensor: Updated atom features after convolution, shape (N, atom_feature_len).
        """
        N, M = nbr_fea_idx.shape  # Number of atoms, max number of neighbors

        # Get neighbor atom features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # Shape: (N, M, atom_feature_len)

        # Concatenate atom features with neighbor features and bond features
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_feature_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )  # Shape: (N, M, 2 * atom_feature_len + neighbor_feature_len)

        # Apply fully connected layer
        total_gated_fea = self.fc_full(total_nbr_fea)

        # Batch normalization
        total_gated_fea = self.batch_norm1(
            total_gated_fea.view(-1, 2 * self.atom_feature_len)
        ).view(N, M, 2 * self.atom_feature_len)

        # Split into filter and core
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # Aggregate over neighbors
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)

        # Batch normalization
        nbr_sumed = self.batch_norm2(nbr_sumed)

        # Update atom features
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class MaskedConvLayer(nn.Module):
    """
    Convolutional layer for graph data with masking for padding indices.

    This layer handles variable-sized neighbor lists with padding indices.
    """

    def __init__(self, atom_feature_len: int, neighbor_feature_len: int):
        """
        Initialize the MaskedConvLayer.

        Args:
            atom_feature_len (int): Number of atom hidden features.
            neighbor_feature_len (int): Number of bond (neighbor) features.
        """
        super(MaskedConvLayer, self).__init__()
        self.atom_feature_len = atom_feature_len
        self.neighbor_feature_len = neighbor_feature_len

        self.fc_full = nn.Linear(
            2 * self.atom_feature_len + self.neighbor_feature_len,
            2 * self.atom_feature_len,
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.batch_norm1 = nn.BatchNorm1d(2 * self.atom_feature_len)
        self.batch_norm2 = nn.BatchNorm1d(self.atom_feature_len)
        self.softplus2 = nn.Softplus()

    def forward(
        self,
        atom_in_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward pass for the masked convolutional layer.

        Args:
            atom_in_fea (torch.Tensor): Input atom features with shape (N, atom_feature_len),
                where N is the total number of atoms in the batch.
            nbr_fea (torch.Tensor): Neighbor (bond) features with shape (N, M, neighbor_feature_len),
                where M is the maximum number of neighbors.
            nbr_fea_idx (torch.LongTensor): Indices of M neighbors for each atom, shape (N, M).

        Returns:
            torch.Tensor: Updated atom features after convolution, shape (N, atom_feature_len).
        """
        N, M = nbr_fea_idx.shape  # Number of atoms, max number of neighbors

        # Create a mask for valid neighbor indices (assuming padding index is 0)
        mask = (nbr_fea_idx != 0).unsqueeze(-1).float()  # Shape: (N, M, 1)

        # Get neighbor atom features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :] * mask  # Shape: (N, M, atom_feature_len)

        # Zero out features corresponding to padding indices
        nbr_fea = nbr_fea * mask  # Shape: (N, M, neighbor_feature_len)

        # Concatenate atom features with neighbor features and bond features
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_feature_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )  # Shape: (N, M, 2 * atom_feature_len + neighbor_feature_len)

        # Apply fully connected layer
        total_gated_fea = self.fc_full(total_nbr_fea)

        # Batch normalization
        total_gated_fea = self.batch_norm1(
            total_gated_fea.view(-1, 2 * self.atom_feature_len)
        ).view(N, M, 2 * self.atom_feature_len)

        # Split into filter and core
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter) * mask
        nbr_core = self.softplus1(nbr_core) * mask

        # Aggregate over neighbors
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)

        # Batch normalization
        nbr_sumed = self.batch_norm2(nbr_sumed)

        # Update atom features
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN) for predicting material properties.

    This model takes a crystal graph as input and outputs predictions for material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        classification: bool = False,
    ):
        """
        Initialize the CrystalGraphConvNet.

        Args:
            orig_atom_fea_len (int): Number of atom features in the input.
            nbr_fea_len (int): Number of bond features.
            atom_fea_len (int, optional): Number of hidden atom features in the convolutional layers. Default is 64.
            n_conv (int, optional): Number of convolutional layers. Default is 3.
            h_fea_len (int, optional): Number of hidden features after pooling. Default is 128.
            n_h (int, optional): Number of hidden layers after pooling. Default is 1.
            classification (bool, optional): If True, model will perform classification. Default is False.
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification

        # Atom embedding layer
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Convolutional layers
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )

        # Fully connected layers after pooling
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_activation = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.activations = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.LongTensor,
        crystal_atom_idx: List[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CGCNN.

        Args:
            atom_fea (torch.Tensor): Atom features from atom type, shape (N, orig_atom_fea_len).
            nbr_fea (torch.Tensor): Bond features of each atom's neighbors, shape (N, M, nbr_fea_len).
            nbr_fea_idx (torch.LongTensor): Indices of neighbors for each atom, shape (N, M).
            crystal_atom_idx (List[torch.LongTensor]): Mapping from crystal index to atom indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output predictions and crystal features.
        """
        # Initial atom feature embedding
        atom_fea = self.embedding(atom_fea)  # Shape: (N, atom_fea_len)

        # Convolutional layers
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        # Pooling to obtain crystal features
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # Fully connected layers
        crys_fea = self.conv_to_fc_activation(self.conv_to_fc(crys_fea))

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, "fcs") and hasattr(self, "activations"):
            for fc, activation in zip(self.fcs, self.activations):
                crys_fea = activation(fc(crys_fea))

        # Output layer
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)

        return out, crys_fea

    def pooling(
        self,
        atom_fea: torch.Tensor,
        crystal_atom_idx: List[torch.LongTensor],
    ) -> torch.Tensor:
        """
        Pool atom features to obtain crystal features.

        Args:
            atom_fea (torch.Tensor): Atom feature vectors, shape (N, atom_fea_len).
            crystal_atom_idx (List[torch.LongTensor]): Mapping from crystal index to atom indices.

        Returns:
            torch.Tensor: Pooled crystal features, shape (number of crystals, atom_fea_len).
        """
        assert sum([len(idx) for idx in crystal_atom_idx]) == atom_fea.size(0)

        # Mean pooling
        pooled_features = [
            torch.mean(atom_fea[idx], dim=0, keepdim=True) for idx in crystal_atom_idx
        ]
        return torch.cat(pooled_features, dim=0)


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
