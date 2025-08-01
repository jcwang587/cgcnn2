from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import random
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader

import cgcnn2

from .data import CIFData_NoTarget, collate_pool
from .model import CrystalGraphConvNet

# ----------------------------------------------------------------------
# Global variables
# ----------------------------------------------------------------------

PLOT_RC_PARAMS: dict[str, float | int] = {
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.linewidth": 1.5,
    "xtick.major.size": 5,
    "xtick.major.width": 1.5,
    "xtick.minor.size": 3,
    "xtick.minor.width": 1,
    "ytick.major.size": 5,
    "ytick.major.width": 1.5,
    "ytick.minor.size": 3,
    "ytick.minor.width": 1,
    "legend.fontsize": 18,
    "legend.frameon": False,
}


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def setup_logging() -> None:
    """
    Sets up logging for the project.
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)

    logging.info(f"cgcnn2 version: {cgcnn2.__version__}")
    logging.info(f"cuda version: {torch.version.cuda}")
    logging.info(f"torch version: {torch.__version__}")


def get_local_version() -> str:
    """
    Retrieves the version of the project from the pyproject.toml file.

    Returns:
        version (str): The version of the project.
    """
    project_root = Path(__file__).parents[2]
    toml_path = project_root / "pyproject.toml"
    try:
        with toml_path.open("rb") as f:
            data = tomllib.load(f)
            version = data["project"]["version"]
        return version
    except Exception:
        return "unknown"


def output_id_gen() -> str:
    """
    Generates a unique output identifier based on current date and time.

    Returns:
        folder_name (str): A string in format 'output_mmdd_HHMM' for current date/time.
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
        learning_rates (list[float]): A list of learning rates for each parameter group.
    """

    learning_rates = []
    for param_group in optimizer.param_groups:
        learning_rates.append(param_group["lr"])
    return learning_rates


def metrics_text(
    df: pd.DataFrame,
    metrics: list[str] = ["mae", "r2"],
    metrics_precision: str = "3f",
    unit: str | None = None,
    unit_scale: float = 1.0,
) -> str:
    """
    Create a text string containing the metrics and their values.

    Args:
        df (pd.DataFrame): DataFrame containing the true and pred values.
        metrics (list[str]): A list of metrics to be displayed in the plot.
        metrics_precision (str): Format string for the metrics.
        unit (str | None): Unit of the property.
        unit_scale (float): Scale factor for the unit.
    Returns:
        text (str): A text string containing the metrics and their values.
    """

    values: dict[str, float] = {}
    for m in metrics:
        m_lower = m.lower()
        if m_lower == "mae":
            values["MAE"] = np.mean(np.abs(df["true"] - df["pred"])) * unit_scale
        elif m_lower == "mse":
            values["MSE"] = np.mean((df["true"] - df["pred"]) ** 2) * unit_scale
        elif m_lower == "rmse":
            values["RMSE"] = (
                np.sqrt(np.mean((df["true"] - df["pred"]) ** 2)) * unit_scale
            )
        elif m_lower == "r2":
            values["R^2"] = 1 - np.sum((df["true"] - df["pred"]) ** 2) / np.sum(
                (df["true"] - df["true"].mean()) ** 2
            )
        else:
            raise ValueError(f"Unsupported metric: {m}")

    text_lines: list[str] = []
    for name, val in values.items():
        if unit and name == "MSE":
            unit_str = rf"\,\mathrm{{{unit}}}^2"
        elif unit and name != "R^2":
            unit_str = rf"\,\mathrm{{{unit}}}"
        else:
            unit_str = ""

        if name == "R^2":
            latex_name = r"R^2"
        else:
            latex_name = rf"\mathrm{{{name}}}"

        if name == "R^2":
            text_lines.append(rf"${latex_name}: {val:.3f}{unit_str}$")
        else:
            text_lines.append(rf"${latex_name}: {val:.{metrics_precision}}{unit_str}$")
    text = "\n".join(text_lines)

    return text


def plot_hexbin(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes | None = None,
    metrics: list[str] = ["mae", "r2"],
    metrics_precision: str = "3f",
    unit: str | None = None,
    unit_scale: float = 1.0,
    subfigure_label: str | None = None,
    out_png: str | None = None,
) -> None:
    """
    Create a hexbin plot and save it to a file.

    Args:
        df (pd.DataFrame): DataFrame containing the true and pred values.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        ax (plt.Axes | None): Axes object to plot the hexbin on.
        metrics (list[str]): A list of strings to be displayed in the plot.
        metrics_precision (str): Format string for the metrics.
        unit (str | None): Unit of the property.
        unit_scale (float): Scale factor for the unit.
        subfigure_label (str | None): Label for the subfigure.
        out_png (str | None): Path of the PNG file in which to save the hexbin plot.

    """

    with plt.rc_context(PLOT_RC_PARAMS):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        else:
            ax.get_figure()

        hb = ax.hexbin(
            x="true",
            y="pred",
            data=df,
            gridsize=40,
            cmap="viridis",
            bins="log",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Keep axes square
        ax.set_box_aspect(1)

        # Get the current axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])

        # Plot y = x reference line (grey dashed)
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="grey",
            linewidth=2,
        )

        # Restore the original limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # add density colorbar put inside the plot
        cax = inset_axes(
            ax, width="3.5%", height="70%", loc="lower right", borderpad=0.5
        )
        plt.colorbar(hb, cax=cax)
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")

        # Compute requested metrics
        text = metrics_text(df, metrics, metrics_precision, unit, unit_scale)

        if subfigure_label is not None:
            text = f"{subfigure_label}\n{text}"

        ax.text(
            0.025,
            0.975,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        if out_png is not None:
            plt.savefig(out_png, format="png", dpi=300, bbox_inches="tight")


def plot_scatter(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes | None = None,
    true_types: list[str] = ["true_train", "true_valid", "true_test"],
    pred_types: list[str] = ["pred_train", "pred_valid", "pred_test"],
    colors: Sequence[str] = (
        "#137DC5",
        "#FACF39",
        "#BF1922",
        "#F7E8D3",
        "#B89FDC",
        "#0F0C08",
    ),
    legend_labels: list[str] | None = None,
    metrics: list[str] = ["mae", "r2"],
    metrics_precision: str = "3f",
    unit: str | None = None,
    unit_scale: float = 1.0,
    subfigure_label: str | None = None,
    out_png: str | None = None,
) -> None:
    """
    Create a scatter plot and save it to a file.

    Args:
        df (pd.DataFrame): DataFrame containing the true and pred values.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        ax (plt.Axes | None): Axes object to plot the scatter on.
        true_types (list[str]): A list of true data types to be displayed in the plot.
        pred_types (list[str]): A list of pred data types to be displayed in the plot.
        colors (Sequence[str]): A list of colors to be used for the data types.
            Default palette is adapted from
            [Looka 2025](https://looka.com/blog/logo-color-trends/) with six colors.
        legend_labels (list[str] | None): A list of labels for the legend.
        metrics (list[str]): Metrics to display in the plot.
        metrics_precision (str): Format string for the metrics.
        unit (str | None): Unit of the property.
        unit_scale (float): Scale factor for the unit.
        subfigure_label (str | None): Label for the subfigure.
        out_png (str | None): Path of the PNG file in which to save the scatter plot.

    """

    with plt.rc_context(PLOT_RC_PARAMS):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        else:
            ax.get_figure()

        for true_type, pred_type in zip(true_types, pred_types):
            ax.scatter(
                x=true_type,
                y=pred_type,
                data=df,
                c=colors[true_types.index(true_type) % len(colors)],
                alpha=0.5,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Keep axes square
        ax.set_box_aspect(1)

        # Get the current axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        axis_min = min(xlim[0], ylim[0])
        axis_max = max(xlim[1], ylim[1])

        # Plot y = x reference line (grey dashed)
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            color="grey",
            linewidth=2,
        )

        # Restore the original limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Convert test data for metrics calculation
        df_metrics = df.rename(
            columns={
                "true_test": "true",
                "pred_test": "pred",
            }
        )

        # Compute requested metrics
        text = metrics_text(df_metrics, metrics, metrics_precision, unit, unit_scale)

        if subfigure_label is not None:
            text = f"{subfigure_label}\n{text}"

        ax.text(
            0.025,
            0.975,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

        if legend_labels is not None:
            if len(legend_labels) != len(true_types):
                raise ValueError(
                    f"legend_labels length ({len(legend_labels)}) must match number of data series ({len(true_types)})"
                )
            ax.legend(legend_labels, loc="lower right")

        if out_png is not None:
            plt.savefig(out_png, format="png", dpi=300, bbox_inches="tight")


def plot_convergence(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes | None = None,
    y2label: str | None = None,
    ylabel_precision: str = "3f",
    y2label_precision: str = "3f",
    colors: Sequence[str] = ("#137DC5", "#BF1922"),
    xtick_rotation: float = 0,
    subfigure_label: str | None = None,
    out_png: str | None = None,
) -> None:
    """
    Create a convergence plot and save it to a file.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics values.
        xlabel (str): Label for the x-axis (epochs)
        ylabel (str): Label for the y-axis (metric)
        ax (plt.Axes | None): Axes object to plot the convergence on.
        y2label (str | None): Label for the y2-axis (metric)
        ylabel_precision (str): Format string for the y-axis label.
        y2label_precision (str): Format string for the y2-axis label.
        colors (Sequence[str]): Colors for the lines.
        xtick_rotation (float): Rotation of the x-axis tick labels.
        subfigure_label (str | None): Label for the subfigure.
        out_png (str | None): Path of the PNG file in which to save the convergence plot.

    """

    with plt.rc_context(PLOT_RC_PARAMS):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        else:
            fig = ax.get_figure()

        x = df[xlabel]
        y = df[ylabel]

        # Primary line (left y‑axis)
        (ln1,) = ax.plot(x, y, label=ylabel, color=colors[0])

        lines = [ln1]
        labels = [ylabel]

        # Optional secondary line (right y‑axis)
        if y2label is not None:
            y2 = df[y2label]
            ax2 = ax.twinx()

            (ln2,) = ax2.plot(x, y2, linestyle="--", label=y2label, color=colors[1])

            lines.append(ln2)
            labels.append(y2label)

            y1_lim = ax.get_ylim()
            y2_lim = ax2.get_ylim()

            ax.set_yticks(np.linspace(y1_lim[0], y1_lim[1], 6))
            ax2.set_yticks(np.linspace(y2_lim[0], y2_lim[1], 6))

            ax.yaxis.set_major_formatter(
                mticker.FormatStrFormatter(f"%.{ylabel_precision}")
            )
            ax2.yaxis.set_major_formatter(
                mticker.FormatStrFormatter(f"%.{y2label_precision}")
            )

            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

            ax.legend(lines, labels, loc="center right")

            ax.spines["left"].set_color(colors[0])
            ax2.spines["left"].set_visible(False)
            ax2.spines["right"].set_color(colors[1])
            ax.spines["right"].set_visible(False)

            ax.tick_params(axis="y", colors=colors[0], which="both")
            ax2.tick_params(axis="y", colors=colors[1], which="both")

        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        ax.tick_params(axis="x", rotation=xtick_rotation)

        ax.grid(True, which="major", alpha=0.3)

        if subfigure_label is not None:
            ax.text(
                0.025,
                0.975,
                subfigure_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
            )

        if out_png is not None:
            fig.savefig(out_png, dpi=300, bbox_inches="tight")


def cgcnn_test(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    results_file: str = "results.csv",
    plot_file: str = "parity_plot.png",
    axis_limits: list[float] | None = None,
    **kwargs: Any,
) -> None:
    """
    This function takes a pre-trained CGCNN model and a test dataset, runs
    inference to generate predictions, creates a parity plot comparing pred
    versus true values, and writes the results to a CSV file.

    Args:
        model (torch.nn.Module): The pre-trained CGCNN model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): The device ('cuda' or 'cpu') where the model will be run.
        results_file (str, optional): File path for saving results as CSV.
        plot_file (str, optional): File path for saving the parity plot.
        axis_limits (list, optional): Limits for x-axis (true values) of the parity plot.
        **kwargs: Additional keyword arguments:
            xlabel (str): x-axis label for the parity plot.
            ylabel (str): y-axis label for the parity plot.

    Notes:
        This function is intended for use in a command-line interface, providing
        direct output of results. For programmatic downstream analysis, consider
        using the API functions instead, i.e. cgcnn_pred and cgcnn_descriptor.
    """

    # Extract optional plot labels from kwargs
    xlabel = kwargs.get("xlabel", "true")
    ylabel = kwargs.get("ylabel", "pred")

    model.eval()
    targets_list = []
    outputs_list = []
    cif_ids = []

    with torch.inference_mode():
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

    targets_array = np.array(targets_list)
    outputs_array = np.array(outputs_list)

    # MSE and R2 Score
    mse = np.mean((targets_array - outputs_array) ** 2)
    ss_res = np.sum((targets_array - outputs_array) ** 2)
    ss_tot = np.sum((targets_array - np.mean(targets_array)) ** 2)
    r2 = 1 - ss_res / ss_tot
    logging.info(f"MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    # Save results to CSV
    sorted_rows = sorted(zip(cif_ids, targets_list, outputs_list), key=lambda x: x[0])
    with open(results_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["cif_id", "true", "pred"])
        writer.writerows(sorted_rows)
    logging.info(f"Prediction results have been saved to {results_file}")

    # Create parity plot
    df_full = pd.DataFrame({"true": targets_list, "pred": outputs_list})
    plot_hexbin(df_full, xlabel, ylabel, out_png=plot_file)
    logging.info(f"Hexbin plot has been saved to {plot_file}")

    # If axis limits are provided, save the csv file with the specified limits
    if axis_limits:
        df_clip = df_full[
            (df_full["true"] >= axis_limits[0]) & (df_full["true"] <= axis_limits[1])
        ]
        clipped_file = plot_file.replace(
            ".png", f"_axis_limits_{axis_limits[0]}_{axis_limits[1]}.png"
        )
        plot_hexbin(df_clip, xlabel, ylabel, out_png=clipped_file)
        logging.info(
            f"Hexbin plot clipped to {axis_limits} on true has been saved to {clipped_file}"
        )


def cgcnn_descriptor(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    verbose: int,
) -> tuple[list[float], list[torch.Tensor]]:
    """
    This function takes a pre-trained CGCNN model and a dataset, runs inference
    to generate predictions and features from the last layer, and returns the
    predictions and features. It is not necessary to have target values for the
    pred set.

    Args:
        model (torch.nn.Module): The trained CGCNN model.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): The device ('cuda' or 'cpu') where the model will be run.
        verbose (int): The verbosity level of the output.

    Returns:
        tuple: A tuple containing:
            - list: Model predictions
            - list: Crystal features from the last layer

    Notes:
        This function is intended for use in programmatic downstream analysis,
        where the user wants to continue downstream analysis using predictions or
        features (descriptors) generated by the model. For the command-line interface,
        consider using the cgcnn_pr script instead.
    """

    model.eval()
    targets_list = []
    outputs_list = []
    crys_feas_list = []
    index = 0

    with torch.inference_mode():
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

            # Extract the true values from cif_id and output tensor
            cif_id_value = cif_id[0] if cif_id and isinstance(cif_id, list) else cif_id
            prediction_value = output.item() if output.numel() == 1 else output.tolist()

            if verbose >= 10:
                logging.info(
                    f"index: {index} | cif id: {cif_id_value} | prediction: {prediction_value}"
                )

    return outputs_list, crys_feas_list


def cgcnn_pred(
    model_path: str,
    full_set: str,
    verbose: int = 101,
    cuda: bool = False,
    num_workers: int = 0,
) -> tuple[list[float], list[torch.Tensor]]:
    """
    This function takes the path to a pre-trained CGCNN model and a dataset,
    runs inference to generate predictions, and returns the predictions. It is
    not necessary to have target values for the pred set.

    Args:
        model_path (str): Path to the file containing the pre-trained model parameters.
        full_set (str): Path to the directory containing all CIF files for the dataset.
        verbose (int): Verbosity level of the output.
        cuda (bool): Whether to use CUDA.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: A tuple containing:
            - list: Model predictions
            - list: Features from the last layer

    Notes:
        This function is intended for use in programmatic downstream analysis,
        where the user wants to continue downstream analysis using predictions or
        features (descriptors) generated by the model. For the command-line interface,
        consider using the cgcnn_pr script instead.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"=> No model params found at '{model_path}'")

    total_dataset = CIFData_NoTarget(full_set)

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

    if verbose >= 100:
        print_checkpoint_info(checkpoint, model_path)

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

    pred, last_layer = cgcnn_descriptor(model, full_loader, device, verbose)

    return pred, last_layer


def unique_structures_clean(dataset_dir, delete_duplicates=False):
    """
    Checks for duplicate (structurally equivalent) structures in a directory
    of CIF files using pymatgen's StructureMatcher and returns the count
    of unique structures.

    Args:
        dataset_dir (str): The path to the dataset containing CIF files.
        delete_duplicates (bool): Whether to delete the duplicate structures.

    Returns:
        grouped (list): A list of lists, where each sublist contains structurally equivalent structures.
    """
    cif_files = [f for f in os.listdir(dataset_dir) if f.endswith(".cif")]
    structures = []
    filenames = []

    for fname in cif_files:
        full_path = os.path.join(dataset_dir, fname)
        structures.append(Structure.from_file(full_path))
        filenames.append(fname)

    id_to_fname = {id(s): fn for s, fn in zip(structures, filenames)}

    matcher = StructureMatcher()
    grouped = matcher.group_structures(structures)

    grouped_fnames = [[id_to_fname[id(s)] for s in group] for group in grouped]

    if delete_duplicates:
        for file_group in grouped_fnames:
            # keep the first file, delete the rest
            for dup_fname in file_group[1:]:
                os.remove(os.path.join(dataset_dir, dup_fname))

    return grouped_fnames


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


def print_checkpoint_info(checkpoint: dict[str, Any], model_path: str) -> None:
    """
    Prints the checkpoint information.

    Args:
        checkpoint (dict[str, Any]): The checkpoint dictionary.
        model_path (str): The path to the model file.
    """
    epoch = checkpoint.get("epoch", "N/A")
    mse = checkpoint.get("best_mse_error")
    mae = checkpoint.get("best_mae_error")

    metrics = []
    if mse is not None:
        metrics.append(f"MSE={mse:.4f}")
    if mae is not None:
        metrics.append(f"MAE={mae:.4f}")

    metrics_str = ", ".join(metrics) if metrics else "N/A"

    logging.info(
        f"=> Loaded model from '{model_path}' (epoch {epoch}, validation {metrics_str})"
    )


def seed_everything(seed: int) -> None:
    """
    Seeds the random number generators for Python, NumPy, PyTorch, and PyTorch CUDA.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
