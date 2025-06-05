import io
import json
import os
import warnings

import numpy as np
import torch
from line_profiler import LineProfiler
from pymatgen.core import Structure


class AtomInitializer(object):
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
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class GaussianDistance(object):
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


def main():
    cif_id = "llto.cif"
    target = 1
    radius = 8
    max_num_nbr = 12
    ari = AtomCustomJSONInitializer("../examples/data/sample-regression/atom_init.json")
    crystal = Structure.from_file(cif_id)
    atom_fea = np.vstack(
        [ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]
    )
    atom_fea = torch.Tensor(atom_fea)

    # --- fast neighbour search ---
    center_idx, neigh_idx, _images, dists = crystal.get_neighbor_list(radius)  # new API

    # Bucket neighbours by centre atom, sort by distance
    n_sites = len(crystal)
    bucket = [[] for _ in range(n_sites)]
    for c, n, d in zip(center_idx, neigh_idx, dists):
        bucket[c].append((n, d))
    bucket = [sorted(lst, key=lambda x: x[1]) for lst in bucket]

    # --- pad / truncate to max_num_nbr exactly as before ---
    nbr_fea_idx, nbr_fea = [], []
    for lst in bucket:
        if len(lst) < max_num_nbr:
            warnings.warn(
                f"{cif_id} did not have enough neighbours; consider increasing radius.",
                stacklevel=2,
            )
        # take first max_num_nbr entries, pad if needed
        idxs = [t[0] for t in lst[:max_num_nbr]]
        dvec = [t[1] for t in lst[:max_num_nbr]]
        pad = max_num_nbr - len(idxs)
        nbr_fea_idx.append(idxs + [0] * pad)
        nbr_fea.append(dvec + [radius + 1.0] * pad)

    nbr_fea_idx = torch.as_tensor(np.array(nbr_fea_idx), dtype=torch.long)
    nbr_fea = torch.as_tensor(
        GaussianDistance(0, radius, 0.2).expand(np.array(nbr_fea))
    )
    target = torch.tensor([float(target)])

    print(atom_fea.shape)
    print(nbr_fea.shape)
    print(nbr_fea_idx.shape)
    print(target.shape)


if __name__ == "__main__":

    # ---- set up the profiler ----
    lp = LineProfiler()
    lp_wrapper = lp(main)  # profile ONLY main(); add more functions if you like

    # ---- run the code ----
    lp_wrapper()

    # ---- capture the text stats that line_profiler prints ----
    s = io.StringIO()
    lp.print_stats(stream=s, output_unit=1e-6)  # μs resolution
    stats_text = s.getvalue()

    # ---- parse -> sort -> print ----
    lines = []
    for row in stats_text.splitlines():
        try:
            parts = row.split(None, 5)
            if len(parts) == 6 and parts[0].isdigit():
                _, ncalls, tottime, _, _, code = parts
                lines.append((float(tottime), code.strip()))
        except ValueError:
            pass

    lines.sort(reverse=True, key=lambda x: x[0])  # highest-time first

    print("\n=== Lines in main() sorted by TOTAL time (µs) ===")
    for t, code in lines:
        print(f"{t:>10.6f}  {code}")
