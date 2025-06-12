import atexit
import csv
import functools
import json
import os
import random
import shutil
import tempfile
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal properties.

    Args:
        dataset_list (list of tuples): List of tuples for each data point. Each tuple contains:
        atom_fea (torch.Tensor): shape (n_i, atom_fea_len) Atom features for each atom in the crystal
        nbr_fea (torch.Tensor): shape (n_i, M, nbr_fea_len) Bond features for each atom's M neighbors
        nbr_fea_idx (torch.LongTensor): shape (n_i, M) Indices of M neighbors of each atom
        target (torch.Tensor): shape (1, ) Target value for prediction
        cif_id (str or int) Unique ID for the crystal

    Returns:
        batch_atom_fea (torch.Tensor): shape (N, orig_atom_fea_len) Atom features from atom type
        batch_nbr_fea (torch.Tensor): shape (N, M, nbr_fea_len) Bond features of each atom's M neighbors
        batch_nbr_fea_idx (torch.LongTensor): shape (N, M) Indices of M neighbors of each atom
        crystal_atom_idx (list of torch.LongTensor): length N0 Mapping from the crystal idx to atom idx
        batch_target (torch.Tensor): shape (N, 1) Target value for prediction
        batch_cif_ids (list of str or int): Unique IDs for each crystal
    """

    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id in dataset_list:
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
        ),
        torch.stack(batch_target, dim=0),
        batch_cif_ids,
    )


class GaussianDistance:
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Args:
            dmin (float): Minimum interatomic distance (center of the first Gaussian).
            dmax (float): Maximum interatomic distance (center of the last Gaussian).
            step (float): Spacing between consecutive Gaussian centers.
            var (float, optional): Variance of each Gaussian. If None, defaults to step.
        """

        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Project each scalar distance onto a set of Gaussian basis functions.

        Args:
            distances (np.ndarray): An array of interatomic distances.

        Returns:
            expanded_distance (np.ndarray): An array where the last dimension contains the Gaussian basis values for each input distance.
        """

        expanded_distance = np.exp(
            -((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2
        )
        return expanded_distance


class AtomInitializer:
    """
    Base class for initializing the vector representation for atoms.
    Use one `AtomInitializer` per dataset.
    """

    def __init__(self, atom_types):
        """
        Initialize the atom types and embedding dictionary.

        Args:
            atom_types (set): A set of unique atom types in the dataset.
        """
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        """
        Get the vector representation for an atom type.

        Args:
            atom_type (str): The type of atom to get the vector representation for.
        """
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary for the atom initializer.

        Args:
            state_dict (dict): The state dictionary to load.
        """
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self) -> dict:
        """
        Get the state dictionary for the atom initializer.

        Returns:
            dict: The state dictionary.
        """
        return self._embedding

    def decode(self, idx: int) -> str:
        """
        Decode an index to an atom type.

        Args:
            idx (int): The index to decode.

        Returns:
            str: The decoded atom type.
        """
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Args:
        elem_embedding_file (str): The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super().__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files.

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Args:
        root_dir (str): The path to the root directory of the dataset
        max_num_nbr (int): The maximum number of neighbors while constructing the crystal graph
        radius (float): The cutoff radius for searching neighbors
        dmin (float): The minimum distance for constructing GaussianDistance
        step (float): The step size for constructing GaussianDistance
        cache_size (int | None): The size of the lru cache for the dataset. Default is None.
        random_seed (int): Random seed for shuffling the dataset

    Returns:
        atom_fea (torch.Tensor): shape (n_i, atom_fea_len)
        nbr_fea (torch.Tensor): shape (n_i, M, nbr_fea_len)
        nbr_fea_idx (torch.LongTensor): shape (n_i, M)
        target (torch.Tensor): shape (1, )
        cif_id (str or int): Unique ID for the crystal
    """

    def __init__(
        self,
        root_dir,
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        cache_size=None,
        random_seed=123,
    ):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), "root_dir does not exist!"
        id_prop_file = os.path.join(self.root_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), "id_prop.csv does not exist!"
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        atom_init_file = os.path.join(self.root_dir, "atom_init.json")
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self._raw_load_item = self._load_item_fast
        self.cache_size = cache_size
        self._configure_cache()

    def set_cache_size(self, cache_size: Optional[int]) -> None:
        """
        Change the LRU-cache capacity on the fly.

        Args:
            cache_size (int | None): The size of the cache to set, None for unlimited size. Default is None.
        """
        self.cache_size = cache_size
        if hasattr(self._cache_load, "cache_clear"):
            self._cache_load.cache_clear()
        self._configure_cache()

    def clear_cache(self) -> None:
        """
        Clear the current cache.
        """
        if hasattr(self._cache_load, "cache_clear"):
            self._cache_load.cache_clear()

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        return self._cache_load(idx)

    def _configure_cache(self) -> None:
        """
        Wrap `_raw_load_item` with an LRU cache.
        """
        if self.cache_size is None:
            self._cache_load = functools.lru_cache(maxsize=None)(self._raw_load_item)
        elif self.cache_size <= 0:
            self._cache_load = self._raw_load_item
        else:
            self._cache_load = functools.lru_cache(maxsize=self.cache_size)(
                self._raw_load_item
            )

    def _load_item(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + ".cif"))
        atom_fea = np.vstack(
            [
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ]
        )
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    "{} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius.".format(cif_id),
                    stacklevel=2,
                )
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id

    def _load_item_fast(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + ".cif"))
        atom_fea = np.vstack(
            [
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ]
        )
        atom_fea = torch.Tensor(atom_fea)
        center_idx, neigh_idx, _images, dists = crystal.get_neighbor_list(self.radius)
        n_sites = len(crystal)
        bucket = [[] for _ in range(n_sites)]
        for c, n, d in zip(center_idx, neigh_idx, dists):
            bucket[c].append((n, d))
        bucket = [sorted(lst, key=lambda x: x[1]) for lst in bucket]
        nbr_fea_idx, nbr_fea = [], []
        for lst in bucket:
            if len(lst) < self.max_num_nbr:
                warnings.warn(
                    f"{cif_id} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius.",
                    stacklevel=2,
                )
            idxs = [t[0] for t in lst[: self.max_num_nbr]]
            dvec = [t[1] for t in lst[: self.max_num_nbr]]
            pad = self.max_num_nbr - len(idxs)
            nbr_fea_idx.append(idxs + [0] * pad)
            nbr_fea.append(dvec + [self.radius + 1.0] * pad)
        nbr_fea_idx = torch.as_tensor(np.array(nbr_fea_idx), dtype=torch.long)
        nbr_fea = self.gdf.expand(np.array(nbr_fea))
        nbr_fea = torch.Tensor(nbr_fea)
        target = torch.tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


class CIFData_NoTarget(Dataset):
    """
    The CIFData_NoTarget dataset is a wrapper for a dataset where the crystal
    structures are stored in the form of CIF files.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Args:
        root_dir (str): The path to the root directory of the dataset
        max_num_nbr (int): The maximum number of neighbors while constructing the crystal graph
        radius (float): The cutoff radius for searching neighbors
        dmin (float): The minimum distance for constructing GaussianDistance
        step (float): The step size for constructing GaussianDistance
        random_seed (int): Random seed for shuffling the dataset

    Returns:
        atom_fea (torch.Tensor): shape (n_i, atom_fea_len)
        nbr_fea (torch.Tensor): shape (n_i, M, nbr_fea_len)
        nbr_fea_idx (torch.LongTensor): shape (n_i, M)
        target (torch.Tensor): shape (1, )
        cif_id (str or int): Unique ID for the crystal
    """

    def __init__(
        self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=123
    ):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), "root_dir does not exist!"
        id_prop_data = []
        for file in os.listdir(root_dir):
            if file.endswith(".cif"):
                id_prop_data.append(file[:-4])
        id_prop_data = [(cif_id, 0) for cif_id in id_prop_data]
        id_prop_data.sort(key=lambda x: x[0])
        self.id_prop_data = id_prop_data
        random.seed(random_seed)
        atom_init_file = os.path.join(self.root_dir, "atom_init.json")
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + ".cif"))
        atom_fea = np.vstack(
            [
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ]
        )
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    "{} not find enough neighbors to build graph. "
                    "If it happens frequently, consider increase "
                    "radius.".format(cif_id),
                    stacklevel=2,
                )
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr))
                    + [self.radius + 1.0] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[: self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[: self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


def full_set_split(
    full_set_dir: str,
    train_ratio: float,
    valid_ratio: float,
    train_force_dir: str | None = None,
    random_seed: int = 0,
):
    """
    Split the full set into train, valid, and test sets into a temporary directory.

    Args:
        full_set_dir (str): The path to the full set
        train_ratio (float): The ratio of the training set
        valid_ratio (float): The ratio of the validation set
        train_force_dir (str): The path to the forced training set. Adding this will no longer keep the original split ratio.
        random_seed (int): The random seed for the split

    Returns:
        train_dir (str): The path to a temporary directory containing the train set
        valid_dir (str): The path to a temporary directory containing the valid set
        test_dir (str): The path to a temporary directory containing the test set
    """
    df = pd.read_csv(
        os.path.join(full_set_dir, "id_prop.csv"),
        header=None,
        names=["cif_id", "property"],
    )

    rng = np.random.RandomState(random_seed)
    df_shuffle = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    n_total = len(df_shuffle)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_df = df_shuffle[:n_train]
    valid_df = df_shuffle[n_train : n_train + n_valid]
    test_df = df_shuffle[n_train + n_valid :]

    temp_train_dir = tempfile.mkdtemp()
    temp_valid_dir = tempfile.mkdtemp()
    temp_test_dir = tempfile.mkdtemp()

    atexit.register(shutil.rmtree, temp_train_dir, ignore_errors=True)
    atexit.register(shutil.rmtree, temp_valid_dir, ignore_errors=True)
    atexit.register(shutil.rmtree, temp_test_dir, ignore_errors=True)

    splits = {
        temp_train_dir: train_df,
        temp_valid_dir: valid_df,
        temp_test_dir: test_df,
    }

    for temp_dir, df in splits.items():
        for cif_id in df["cif_id"]:
            src = os.path.join(full_set_dir, f"{cif_id}.cif")
            dst = os.path.join(temp_dir, f"{cif_id}.cif")
            shutil.copy(src, dst)

    train_df.to_csv(
        os.path.join(temp_train_dir, "id_prop.csv"), index=False, header=False
    )
    valid_df.to_csv(
        os.path.join(temp_valid_dir, "id_prop.csv"), index=False, header=False
    )
    test_df.to_csv(
        os.path.join(temp_test_dir, "id_prop.csv"), index=False, header=False
    )

    shutil.copy(os.path.join(full_set_dir, "atom_init.json"), temp_train_dir)
    shutil.copy(os.path.join(full_set_dir, "atom_init.json"), temp_valid_dir)
    shutil.copy(os.path.join(full_set_dir, "atom_init.json"), temp_test_dir)

    if train_force_dir is not None:
        df_force = pd.read_csv(
            os.path.join(train_force_dir, "id_prop.csv"),
            header=None,
            names=["cif_id", "property"],
        )

        train_df = pd.concat([train_df, df_force])
        train_df.to_csv(
            os.path.join(temp_train_dir, "id_prop.csv"), index=False, header=False
        )

        for cif_id in df_force["cif_id"]:
            src = os.path.join(train_force_dir, f"{cif_id}.cif")
            dst = os.path.join(temp_train_dir, f"{cif_id}.cif")
            shutil.copy(src, dst)

    return temp_train_dir, temp_valid_dir, temp_test_dir
