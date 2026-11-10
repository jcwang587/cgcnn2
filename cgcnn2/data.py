import atexit
from collections.abc import Callable
import csv
import functools
import json
import os
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
import torch
from torch.utils.data import Dataset

Graph = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Item = tuple[Graph, torch.Tensor, str]


def collate_pool(dataset_list: list[Item]):
    """
    Collate a list of dataset items into one batch.

    Args:
        dataset_list (list): Items `((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)` as
            returned by `CIFData`, with `n_i` atoms and `M` neighbors per atom.

    Returns:
        tuple: `((atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, cif_ids)` where

            - atom_fea (torch.Tensor): shape (N, orig_atom_fea_len), all atoms of the batch
            - nbr_fea (torch.Tensor): shape (N, M, nbr_fea_len), bond features of each atom's neighbors
            - nbr_fea_idx (torch.LongTensor): shape (N, M), neighbor indices into the batch
            - crystal_atom_idx (torch.LongTensor): shape (N,), the crystal of each atom, sorted ascending
            - target (torch.Tensor): shape (B, 1)
            - cif_ids (list[str]): the ID of each crystal
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target, batch_cif_ids = [], [], []
    base_idx = 0
    for crystal_i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(
        dataset_list
    ):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        crystal_atom_idx.append(torch.full((n_i,), crystal_i, dtype=torch.long))
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (
        (
            torch.cat(batch_atom_fea),
            torch.cat(batch_nbr_fea),
            torch.cat(batch_nbr_fea_idx),
            torch.cat(crystal_atom_idx),
        ),
        torch.stack(batch_target),
        batch_cif_ids,
    )


def batch_to_device(batch, device):
    """
    Move the batch tensors produced by `collate_pool` to a device.

    Args:
        batch (tuple[torch.Tensor, ...]): (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        device (torch.device | str): Target device.

    Returns:
        tuple[torch.Tensor, ...]: The same tensors on `device`.
    """
    return tuple(t.to(device) for t in batch)


class GaussianDistance:
    """
    Expands interatomic distances (in angstrom) in a Gaussian basis.

    Args:
        dmin (float): Center of the first Gaussian.
        dmax (float): Center of the last Gaussian.
        step (float): Spacing between consecutive centers.
        var (float | None): Variance of each Gaussian. Defaults to `step`.
    """

    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = step if var is None else var

    def expand(self, distances):
        """
        Project distances onto the Gaussian basis.

        Args:
            distances (np.ndarray): Interatomic distances of any shape.

        Returns:
            np.ndarray: `distances.shape + (len(filter),)` basis values.
        """
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomInitializer:
    """
    Atom feature vectors read from an `atom_init.json` file, a dictionary mapping
    atomic numbers to feature vectors.

    Args:
        elem_embedding_file (str): Path to the JSON file.
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        self._embedding = {
            int(number): np.array(vector, dtype=float)
            for number, vector in elem_embedding.items()
        }

    def get_atom_fea(self, atom_type):
        """
        Return the feature vector of an atomic number.

        Args:
            atom_type (int): Atomic number.
        """
        return self._embedding[atom_type]


def _crystal_graph(crystal, cif_id, ari, gdf, max_num_nbr, radius):
    """
    Build the crystal-graph tensors for a single structure.

    Uses pymatgen's C-backed `get_neighbor_list` and vectorized numpy
    bucketing/padding instead of per-atom Python loops.

    Args:
        crystal (pymatgen.core.structure.Structure): The crystal structure.
        cif_id (str): Unique ID of the crystal, used in warnings.
        ari (AtomInitializer): Atom feature initializer.
        gdf (GaussianDistance): Gaussian distance expander.
        max_num_nbr (int): Maximum number of neighbors per atom.
        radius (float): Cutoff radius for the neighbor search.

    Returns:
        atom_fea (torch.Tensor): shape (n_i, atom_fea_len)
        nbr_fea (torch.Tensor): shape (n_i, M, nbr_fea_len)
        nbr_fea_idx (torch.LongTensor): shape (n_i, M)
    """
    atom_fea = np.vstack(
        [ari.get_atom_fea(number) for number in crystal.atomic_numbers]
    )
    center_idx, neigh_idx, _images, dists = crystal.get_neighbor_list(radius)
    n_sites = len(crystal)
    order = np.lexsort((dists, center_idx))
    center_idx, neigh_idx, dists = center_idx[order], neigh_idx[order], dists[order]
    counts = np.bincount(center_idx, minlength=n_sites)
    if (counts < max_num_nbr).any():
        warnings.warn(
            f"{cif_id} not find enough neighbors to build graph. "
            "If it happens frequently, consider increase "
            "radius.",
            stacklevel=2,
        )
    starts = np.cumsum(counts) - counts
    slot = np.arange(center_idx.shape[0]) - np.repeat(starts, counts)
    keep = slot < max_num_nbr
    nbr_fea_idx = np.zeros((n_sites, max_num_nbr), dtype=np.int64)
    nbr_dist = np.full((n_sites, max_num_nbr), radius + 1.0)
    nbr_fea_idx[center_idx[keep], slot[keep]] = neigh_idx[keep]
    nbr_dist[center_idx[keep], slot[keep]] = dists[keep]
    atom_fea = torch.as_tensor(atom_fea, dtype=torch.float32)
    nbr_fea = torch.as_tensor(gdf.expand(nbr_dist), dtype=torch.float32)
    nbr_fea_idx = torch.as_tensor(nbr_fea_idx, dtype=torch.long)
    return atom_fea, nbr_fea, nbr_fea_idx


class CIFData(Dataset):
    """
    Crystal structures stored as CIF files in one directory.

    - `atom_init.json`: the initialization vector of each element.
    - `ID.cif`: one CIF file per crystal, where `ID` is the unique ID of the crystal.
    - `id_prop.csv` (optional): two columns, the ID of each crystal and the value of its
      target property. Without it every CIF file in the directory is loaded with a NaN
      target, which is all that prediction needs.

    Parsed crystal graphs are kept in a per-instance LRU cache, so each CIF file is
    parsed once per process.

    Args:
        root_dir (str): The dataset directory.
        max_num_nbr (int): Maximum number of neighbors per atom in the crystal graph.
        radius (float): Cutoff radius for the neighbor search.
        dmin (float): Center of the first Gaussian of the distance expansion.
        step (float): Spacing of the Gaussian centers.
        cache_size (int | None): Number of parsed crystals kept in memory. None (default)
            keeps all of them, 0 disables the cache.

    Each item is `((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)`:

        atom_fea (torch.Tensor): shape (n_i, atom_fea_len)
        nbr_fea (torch.Tensor): shape (n_i, M, nbr_fea_len)
        nbr_fea_idx (torch.LongTensor): shape (n_i, M)
        target (torch.Tensor): shape (1,)
        cif_id (str): the ID of the crystal
    """

    _cache_load: Callable[[int], Item]

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        cache_size=None,
    ):
        assert os.path.isdir(root_dir), f"{root_dir} is not a directory"
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        id_prop_file = os.path.join(root_dir, "id_prop.csv")
        if os.path.isfile(id_prop_file):
            with open(id_prop_file) as f:
                self.id_prop_data = [
                    (row[0], float(row[1])) for row in csv.reader(f) if row
                ]
        else:
            self.id_prop_data = sorted(
                (name[:-4], float("nan"))
                for name in os.listdir(root_dir)
                if name.endswith(".cif")
            )
        self.ari = AtomInitializer(os.path.join(root_dir, "atom_init.json"))
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        self.cache_size = cache_size
        self._configure_cache()

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, index):
        return self._cache_load(index)

    def __getstate__(self):
        # The lru_cache wrapper around a bound method is not picklable; drop
        # it so instances survive spawn-based DataLoader workers.
        state = self.__dict__.copy()
        state.pop("_cache_load", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._configure_cache()

    def _configure_cache(self) -> None:
        if self.cache_size is not None and self.cache_size <= 0:
            self._cache_load = self._load_item
        else:
            self._cache_load = functools.lru_cache(maxsize=self.cache_size)(
                self._load_item
            )

    def _load_item(self, index: int) -> Item:
        cif_id, target = self.id_prop_data[index]
        crystal = Structure.from_file(os.path.join(self.root_dir, f"{cif_id}.cif"))
        graph = _crystal_graph(
            crystal, cif_id, self.ari, self.gdf, self.max_num_nbr, self.radius
        )
        return graph, torch.tensor([target], dtype=torch.float32), cif_id


def full_set_split(
    full_set_dir: str,
    train_ratio: float,
    valid_ratio: float,
    train_force_dir: str | None = None,
    random_seed: int = 0,
):
    """
    Split a dataset directory into train, valid and test directories.

    The CIF files are copied into three temporary directories that are removed when
    the process exits. The test set receives whatever the two ratios leave.

    Args:
        full_set_dir (str): The dataset to split.
        train_ratio (float): Fraction of the dataset used for training.
        valid_ratio (float): Fraction of the dataset used for validation.
        train_force_dir (str | None): A dataset directory whose structures are all added
            to the training set on top of the split.
        random_seed (int): Seed of the shuffle.

    Returns:
        tuple[str, str, str]: The train, valid and test directories.
    """
    df = pd.read_csv(
        os.path.join(full_set_dir, "id_prop.csv"),
        header=None,
        names=["cif_id", "property"],
    )
    rng = np.random.default_rng(random_seed)
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    n_train, n_valid = int(len(df) * train_ratio), int(len(df) * valid_ratio)
    splits = [
        [(df[:n_train], full_set_dir)],
        [(df[n_train : n_train + n_valid], full_set_dir)],
        [(df[n_train + n_valid :], full_set_dir)],
    ]
    if train_force_dir is not None:
        df_force = pd.read_csv(
            os.path.join(train_force_dir, "id_prop.csv"),
            header=None,
            names=["cif_id", "property"],
        )
        splits[0].append((df_force, train_force_dir))

    temp_dirs = []
    for parts in splits:
        temp_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
        for part, source_dir in parts:
            for cif_id in part["cif_id"]:
                shutil.copy(os.path.join(source_dir, f"{cif_id}.cif"), temp_dir)
        pd.concat([part for part, _ in parts]).to_csv(
            os.path.join(temp_dir, "id_prop.csv"), index=False, header=False
        )
        shutil.copy(os.path.join(full_set_dir, "atom_init.json"), temp_dir)
        temp_dirs.append(temp_dir)
    return tuple(temp_dirs)
