import argparse
import csv
import glob
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatviz import density_hexbin
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .data import CIFData, collate_pool
from .model import CrystalGraphConvNet


def output_id_gen() -> str:
    """
    Generates a unique output identifier based on current date and time.

    Returns:
        str: A string in the format 'output_mmdd_HHMM' representing the current date and time.
    """

    now = datetime.now()
    timestamp = now.strftime("%m%d_%H%M")
    folder_name = f"output_{timestamp}"

    return folder_name


def id_prop_gen(cif_dir: str) -> None:
    """Generates a CSV file containing IDs and properties of CIF files.

    Args:
        cif_dir (str): Directory containing the CIF files.
    """

    cif_list = glob.glob(f"{cif_dir}/*.cif")

    id_prop_cif = pd.DataFrame(
        {
            "id": [os.path.basename(cif).split(".")[0] for cif in cif_list],
            "prop": [0 for _ in range(len(cif_list))],
        }
    )

    id_prop_cif.to_csv(
        f"{cif_dir}/id_prop.csv",
        index=False,
        header=False,
    )


def get_lr(optimizer: torch.optim.Optimizer) -> list[float]:
    """
    Extracts learning rates from a PyTorch optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to extract learning rates from.

    Returns:
        list[float]: A list of learning rates, one for each parameter group in the optimizer.
    """

    return [param_group["lr"] for param_group in optimizer.param_groups]


def extract_fea(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Extracts learned feature representations from a trained model.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): The device ('cuda' or 'cpu') to send tensors to.

    Returns:
        tuple[torch.Tensor, torch.Tensor, list[str]]: A tuple containing:
            - torch.Tensor: Extracted features
            - torch.Tensor: Targets
            - list[str]: CIF IDs
    """

    crys_fea_list, target_list, cif_id_list = [], [], []

    with torch.no_grad():
        for inputs, target, cif_id in loader:
            inputs = [
                item.to(device) if torch.is_tensor(item) else item for item in inputs
            ]
            target = target.to(device)

            _, crys_fea = model(*inputs)

            crys_fea_list.append(crys_fea)
            target_list.append(target)
            cif_id_list.append(cif_id)

    crys_fea = torch.cat(crys_fea_list, dim=0)
    target = torch.cat(target_list, dim=0)

    return crys_fea, target, cif_id_list


def cgcnn_test(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    plot_file: str = "parity_plot.svg",
    results_file: str = "results.csv",
    axis_limits: list[float] | None = None,
    **kwargs: Any,
) -> None:
    """
    Load a trained CGCNN model and test it on a dataset.

    Args:
        model (torch.nn.Module): The trained CGCNN model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): The device ('cuda' or 'cpu') where the model will be run.
        plot_file (str, optional): File path for saving the parity plot. Defaults to 'parity_plot.svg'.
        results_file (str, optional): File path for saving results as CSV. Defaults to 'results.csv'.
        axis_limits (list, optional): Limits for x and y axes of the parity plot. Defaults to None.
        **kwargs: Additional keyword arguments:
            xlabel (str): x-axis label for the parity plot. Defaults to "Actual".
            ylabel (str): y-axis label for the parity plot. Defaults to "Predicted".
    """

    # Extract optional plot labels from kwargs, with defaults
    xlabel = kwargs.get("xlabel", "Actual")
    ylabel = kwargs.get("ylabel", "Predicted")

    model.eval()
    targets_list = []
    outputs_list = []
    cif_ids = []

    with torch.no_grad():
        for input_batch, target, cif_id in loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_batch
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [idx_map.to(device) for idx_map in crystal_atom_idx]
            target = target.to(device)
            output, _ = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

            targets_list.extend(target.cpu().numpy().ravel().tolist())
            outputs_list.extend(output.cpu().numpy().ravel().tolist())
            cif_ids.extend(cif_id)

    mse = mean_squared_error(targets_list, outputs_list)
    r2 = r2_score(targets_list, outputs_list)
    print(f"MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    # Save results to CSV
    sorted_rows = sorted(zip(cif_ids, targets_list, outputs_list), key=lambda x: x[0])
    with open(results_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["cif_id", "Actual", "Predicted"])
        writer.writerows(sorted_rows)
    print(f"Prediction results have been saved to {results_file}")

    # Create parity plot
    fig, ax = plt.subplots(figsize=(8, 6))
    df = pd.DataFrame({"Actual": targets_list, "Predicted": outputs_list})

    ax = density_hexbin(
        x="Actual",
        y="Predicted",
        df=df,
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        best_fit_line=False,
        gridsize=40,
    )
    ax.set_aspect("auto")
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(plot_file, format="svg")
    print(f"Parity plot has been saved to {plot_file}")
    plt.close()

    # If axis limits are provided, save the csv file with the specified limits
    if axis_limits:
        results_file = (
            results_file.split(".")[0]
            + "_axis_limits_"
            + str(axis_limits[0])
            + "_"
            + str(axis_limits[1])
            + ".csv"
        )
        plot_file = (
            plot_file.split(".")[0]
            + "_axis_limits_"
            + str(axis_limits[0])
            + "_"
            + str(axis_limits[1])
            + ".svg"
        )

        df = df[
            (df["Actual"] >= axis_limits[0])
            & (df["Actual"] <= axis_limits[1])
            & (df["Predicted"] >= axis_limits[0])
            & (df["Predicted"] <= axis_limits[1])
        ]

        df.to_csv(
            results_file,
            index=False,
        )

        # Create parity plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax = density_hexbin(
            x="Actual",
            y="Predicted",
            df=df,
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            best_fit_line=False,
            gridsize=40,
        )
        ax.set_aspect("auto")
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig(plot_file, format="svg")
        print(f"Parity plot has been saved to {plot_file}")
        plt.close()


def cgcnn_calculator(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    verbose: int,
) -> tuple[list[float], list[torch.Tensor]]:
    """
    Load a trained CGCNN model and generate predictions and features from the last layer.

    Args:
        model (torch.nn.Module): The trained CGCNN model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): The device ('cuda' or 'cpu') where the model will be run.
        verbose (int): The verbosity level of the output.

    Returns:
        tuple: A tuple containing:
            - list: Model predictions
            - list: Crystal features from the last layer
    """

    model.eval()
    targets_list = []
    outputs_list = []
    crys_feas_list = []
    index = 0

    with torch.no_grad():
        for input, target, cif_id in loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [idx_map.to(device) for idx_map in crystal_atom_idx]
            target = target.to(device)

            output, crys_fea = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

            targets_list.extend(target.cpu().numpy().ravel().tolist())
            outputs_list.extend(output.cpu().numpy().ravel().tolist())
            crys_feas_list.append(crys_fea.cpu().numpy())

            index += 1

            # Extract the actual values from cif_id and output tensor
            cif_id_value = cif_id[0] if cif_id and isinstance(cif_id, list) else cif_id
            prediction_value = output.item() if output.numel() == 1 else output.tolist()

            if verbose >= 3:
                print(
                    "index:",
                    index,
                    "| cif id:",
                    cif_id_value,
                    "| prediction:",
                    prediction_value,
                )

    return outputs_list, crys_feas_list


def cgcnn_pred(
    model_path: str,
    all_set: str,
    verbose: int = 3,
    cuda: bool = False,
    num_workers: int = 0,
) -> tuple[list[float], list[torch.Tensor]]:
    """
    Load a trained CGCNN model and generate predictions.

    Args:
        model_path (str): Path to the file containing the trained model parameters.
        all_set (str): Path to the directory containing all CIF files for the dataset.
        verbose (int, optional): Verbosity level of the output. Defaults to 3.
        cuda (bool, optional): Whether to use CUDA. Defaults to False.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - list: Model predictions
            - list: Features from the last layer

    Raises:
        FileNotFoundError: If the model file is not found.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"=> No model params found at '{model_path}'")

    total_dataset = CIFData(all_set)

    checkpoint = torch.load(
        model_path,
        map_location=lambda storage, loc: storage if not cuda else None,
        weights_only=False,
    )
    structures, _, _ = total_dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model_args = argparse.Namespace(**checkpoint["args"])
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=model_args.atom_fea_len,
        n_conv=model_args.n_conv,
        h_fea_len=model_args.h_fea_len,
        n_h=model_args.n_h,
    )
    if cuda:
        model.cuda()

    normalizer = Normalizer(torch.zeros(3))
    normalizer.load_state_dict(checkpoint["normalizer"])
    model.load_state_dict(checkpoint["state_dict"])

    if verbose >= 3:
        print(
            f"=> Loaded model from '{model_path}' (epoch {checkpoint['epoch']}, validation error {checkpoint['best_mae_error']})"
        )

    device = "cuda" if cuda else "cpu"
    model.to(device).eval()

    full_loader = DataLoader(
        total_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pool,
        pin_memory=cuda,
    )

    pred, last_layer = cgcnn_calculator(model, full_loader, device, verbose)

    return pred, last_layer


def unique_structures_clean(dataset_dir, delete_duplicates=False):
    """
    Checks for duplicate (structurally equivalent) structures in a directory
    of CIF files using pymatgen's StructureMatcher and returns the count
    of unique structures.

    Parameters
    ----------
    dataset_dir: str
        The path to the dataset containing CIF files.
    delete_duplicates: bool
        Whether to delete the duplicate structures, default is False.

    Returns
    -------
    grouped: list
        A list of lists, where each sublist contains structurally equivalent
        structures.
    """
    cif_files = [f for f in os.listdir(dataset_dir) if f.endswith(".cif")]

    structures = []
    for filename in cif_files:
        full_path = os.path.join(dataset_dir, filename)
        structure = Structure.from_file(full_path)
        structures.append(structure)

    matcher = StructureMatcher()
    grouped = matcher.group_structures(structures)

    if delete_duplicates:
        for group in grouped:
            if len(group) > 1:
                for structure in group[1:]:
                    os.remove(os.path.join(dataset_dir, structure.filename))

    return grouped


class Normalizer:
    """
    Normalizes a PyTorch tensor and allows restoring it later.

    This class keeps track of the mean and standard deviation of a tensor and provides methods
    to normalize and denormalize tensors using these statistics.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
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

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns the state dictionary containing the mean and standard deviation.

        Returns:
            dict[str, torch.Tensor]: State dictionary.
        """
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Loads the mean and standard deviation from a state dictionary.

        Args:
            state_dict (dict[str, torch.Tensor]): State dictionary containing 'mean' and 'std'.
        """
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
