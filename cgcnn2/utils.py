from collections.abc import Sequence
import glob
import logging
import os
import random
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.typing import RcKeyType
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import cgcnn2

from .data import CIFData, batch_to_device, collate_pool
from .model import CrystalGraphConvNet

PLOT_RC_PARAMS: dict[RcKeyType, Any] = {
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
# Setup
# ----------------------------------------------------------------------


def setup_logging() -> None:
    """
    Log to stdout with timestamps and record the cgcnn2, torch and CUDA versions.
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


def seed_everything(seed: int) -> None:
    """
    Seed the random number generators of Python, NumPy and PyTorch (CPU and CUDA).

    Args:
        seed (int): The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def id_prop_gen(cif_dir: str) -> None:
    """
    Write an `id_prop.csv` listing every CIF file in a directory with the target 0.

    Args:
        cif_dir (str): Directory containing the CIF files.
    """
    cif_ids = [
        os.path.basename(cif).split(".")[0] for cif in glob.glob(f"{cif_dir}/*.cif")
    ]
    pd.DataFrame({"id": cif_ids, "prop": [0] * len(cif_ids)}).to_csv(
        f"{cif_dir}/id_prop.csv", index=False, header=False
    )


# ----------------------------------------------------------------------
# Checkpoints and models
# ----------------------------------------------------------------------


def load_checkpoint(model_path: str | os.PathLike[str]) -> dict:
    """
    Load a cgcnn2 checkpoint onto the CPU and log its epoch and validation error.

    Checkpoints hold only tensors and plain Python values, so they are read with
    `weights_only=True`.

    Args:
        model_path (str): Path to the checkpoint file.

    Returns:
        dict: The checkpoint, with the keys `epoch`, `state_dict`, `args` and
            `best_mae_error` (`best_mse_error` in checkpoints trained with 0.6).
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    metrics = [
        f"{key[5:8].upper()}={checkpoint[key]:.4f}"
        for key in ("best_mse_error", "best_mae_error")
        if key in checkpoint
    ]
    logging.info(
        f"=> Loaded checkpoint '{model_path}' (epoch {checkpoint.get('epoch', 'N/A')}, "
        f"validation {', '.join(metrics) or 'N/A'})"
    )
    return checkpoint


def input_feature_lengths(dataset: CIFData) -> tuple[int, int]:
    """
    Return `(orig_atom_fea_len, nbr_fea_len)` of a dataset, read from its first item.

    Args:
        dataset (CIFData): The dataset.
    """
    (atom_fea, nbr_fea, _), _, _ = dataset[0]
    return atom_fea.shape[-1], nbr_fea.shape[-1]


def model_from_checkpoint(checkpoint: dict, dataset: CIFData) -> CrystalGraphConvNet:
    """
    Build the model described by a checkpoint's `args` and load its weights.

    Args:
        checkpoint (dict): A checkpoint from `load_checkpoint`.
        dataset (CIFData): A dataset built with the same `atom_init.json` and neighbor
            settings as the training data; it provides the input feature lengths.

    Returns:
        CrystalGraphConvNet: The model, on the CPU.
    """
    orig_atom_fea_len, nbr_fea_len = input_feature_lengths(dataset)
    args = checkpoint["args"]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args["atom_fea_len"],
        n_conv=args["n_conv"],
        h_fea_len=args["h_fea_len"],
        n_h=args["n_h"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def target_statistics(dataset: CIFData) -> tuple[float, float]:
    """
    Mean and standard deviation of the targets of a dataset.

    Args:
        dataset (CIFData): The dataset, usually the training set.
    """
    targets = np.array([target for _, target in dataset.id_prop_data])
    mean, std = float(targets.mean()), float(targets.std())
    if not std > 0:  # NaN without targets, 0 when they are all equal
        raise ValueError(
            "The targets are missing or all equal, so they cannot be standardized."
        )
    return mean, std


def standardize_output(
    model: CrystalGraphConvNet, target_stats: tuple[float, float]
) -> None:
    """
    Make a model that predicts in the original units predict standardized targets, by
    rewriting its output layer. `train` needs this for a model loaded from a checkpoint.

    Args:
        model (CrystalGraphConvNet): The model, changed in place.
        target_stats (tuple[float, float]): Mean and standard deviation from `target_statistics`.
    """
    mean, std = target_stats
    with torch.no_grad():
        model.fc_out.bias.sub_(mean).div_(std)
        model.fc_out.weight.div_(std)


def train(
    model: CrystalGraphConvNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epochs: int,
    checkpoint_path: str,
    checkpoint_args: dict,
    target_stats: tuple[float, float],
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    stop_patience: int | None = None,
    bias_temperature: float = -1.0,
) -> None:
    """
    Train a model and keep the weights with the lowest validation MAE.

    The model is trained on standardized targets, `(target - mean) / std`, so that one
    learning rate suits properties of any unit and scale. A new model can be passed as it
    is; a model that already predicts in the original units, such as one loaded from a
    checkpoint, needs `standardize_output` first. The loss is the mean squared error.

    Checkpoints have the statistics folded into the output layer, so they predict in the
    original units. When training ends the model holds the weights of the best checkpoint.

    Args:
        model (CrystalGraphConvNet): The model, already on `device`.
        train_loader (DataLoader): Training data, batched with `collate_pool`.
        valid_loader (DataLoader): Validation data.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device | str): Where the model lives.
        epochs (int): Number of epochs.
        checkpoint_path (str): Where the best checkpoint is written.
        checkpoint_args (dict): Stored under the checkpoint's `args` key. It must hold the
            model hyperparameters (`atom_fea_len`, `n_conv`, `h_fea_len`, `n_h`) so that
            `model_from_checkpoint` can rebuild the model.
        target_stats (tuple[float, float]): Mean and standard deviation of the training
            targets, from `target_statistics`.
        scheduler (ReduceLROnPlateau | None): Stepped with the validation MAE every epoch.
        stop_patience (int | None): Epochs without improvement before training stops.
        bias_temperature (float): If positive, every error is weighted by
            `exp(-target / bias_temperature)` in the loss and in the validation MAE, so
            that structures with low targets (low energies) weigh more.
    """
    mean, std = target_stats
    best_valid_mae = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch, targets, _ in train_loader:
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(*batch_to_device(batch, device))
            loss = (outputs - (targets - mean) / std) ** 2
            if bias_temperature > 0:
                loss = loss * torch.exp(-targets / bias_temperature)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        predictions, _, targets, _ = predict(model, valid_loader, device)
        errors = np.abs(predictions * std + mean - targets)
        if bias_temperature > 0:
            errors = errors * np.exp(-targets / bias_temperature)
        valid_mae = float(errors.mean())
        if scheduler is not None:
            scheduler.step(valid_mae)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        logging.info(
            f"Epoch [{epoch:03d}/{epochs}] - Train Loss: {train_loss:.5f}, "
            f"Valid MAE: {valid_mae:.5f}, LR: {learning_rates}"
        )
        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            epochs_without_improvement = 0
            # fold the target statistics into the output layer: the checkpoint predicts
            # in the original units
            state_dict = dict(model.state_dict())
            state_dict["fc_out.weight"] = state_dict["fc_out.weight"] * std
            state_dict["fc_out.bias"] = state_dict["fc_out.bias"] * std + mean
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": state_dict,
                    "best_mae_error": valid_mae,
                    "args": checkpoint_args,
                },
                checkpoint_path,
            )
            logging.info(f"  [SAVE] Best model at epoch {epoch}.")
        elif stop_patience is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= stop_patience:
                logging.info(
                    f"Early stopping after {stop_patience} epochs without improvement."
                )
                break
    logging.info("Training complete.")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True)["state_dict"])


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------


def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device | str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Run a model over a loader.

    Args:
        model (torch.nn.Module): The model.
        loader (DataLoader): The data, batched with `collate_pool`.
        device (torch.device | str): Where the model lives.

    Returns:
        predictions (np.ndarray): shape (n,), the predicted property of each crystal.
        descriptors (np.ndarray): shape (n, atom_fea_len), the mean atom features of each crystal after the convolutions.
        targets (np.ndarray): shape (n,), NaN where the dataset has no targets.
        cif_ids (list[str]): The ID of each crystal.
    """
    model.eval()
    predictions, descriptors, targets, cif_ids = [], [], [], []
    with torch.inference_mode():
        for batch, target, cif_id in loader:
            output, crys_fea = model(*batch_to_device(batch, device))
            predictions.append(output.cpu().numpy().ravel())
            descriptors.append(crys_fea.cpu().numpy())
            targets.append(target.numpy().ravel())
            cif_ids.extend(cif_id)
    return (
        np.concatenate(predictions),
        np.concatenate(descriptors),
        np.concatenate(targets),
        cif_ids,
    )


def cgcnn_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    results_file: str = "results.csv",
    plot_file: str = "parity_plot.png",
    axis_limits: Sequence[float] | None = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
) -> None:
    """
    Predict over a loader and write the results to a CSV file. When the dataset has
    targets, also log the MAE, MSE and R² and save a parity plot.

    Args:
        model (torch.nn.Module): The model.
        loader (DataLoader): The data, batched with `collate_pool`.
        device (torch.device | str): Where the model lives.
        results_file (str): CSV file receiving `cif_id,true,pred` rows sorted by ID.
        plot_file (str): PNG file receiving the parity plot.
        axis_limits (Sequence[float] | None): `(min, max)` of true values for an
            additional parity plot clipped to that range.
        xlabel (str): x-axis label of the parity plot.
        ylabel (str): y-axis label of the parity plot.
    """
    predictions, _, targets, cif_ids = predict(model, loader, device)
    df = pd.DataFrame(
        {"cif_id": cif_ids, "true": targets, "pred": predictions}
    ).sort_values("cif_id")
    df.to_csv(results_file, index=False)
    logging.info(f"Prediction results have been saved to {results_file}")

    if df["true"].isna().any():
        logging.info("The dataset has no targets; metrics and parity plot are skipped.")
        return
    errors = df["true"] - df["pred"]
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    r2 = 1 - np.sum(errors**2) / np.sum((df["true"] - df["true"].mean()) ** 2)
    logging.info(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    plot_hexbin(df, xlabel, ylabel, out_png=plot_file)
    logging.info(f"Hexbin plot has been saved to {plot_file}")
    if axis_limits is not None:
        low, high = axis_limits
        clipped_file = plot_file.replace(".png", f"_axis_limits_{low}_{high}.png")
        plot_hexbin(
            df[(df["true"] >= low) & (df["true"] <= high)],
            xlabel,
            ylabel,
            out_png=clipped_file,
        )
        logging.info(
            f"Hexbin plot clipped to {axis_limits} on true has been saved to {clipped_file}"
        )


def cgcnn_pred(
    model_path: str | os.PathLike[str],
    full_set: str | os.PathLike[str],
    device: torch.device | str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the property of every structure in a directory with a trained model.

    Args:
        model_path (str): Path to the checkpoint.
        full_set (str): Directory with the CIF files and `atom_init.json`; `id_prop.csv`
            is optional.
        device (torch.device | str): Where to run the model.
        batch_size (int): Crystals per batch.
        num_workers (int): DataLoader worker processes.

    Returns:
        predictions (np.ndarray): shape (n,), the predicted property of each crystal, in
            the order of `id_prop.csv` or sorted by ID without it.
        descriptors (np.ndarray): shape (n, atom_fea_len), the mean atom features of each
            crystal after the convolutions, for downstream analysis.
    """
    dataset = CIFData(full_set)
    model = model_from_checkpoint(load_checkpoint(model_path), dataset).to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pool,
    )
    predictions, descriptors, _, _ = predict(model, loader, device)
    return predictions, descriptors


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def metrics_text(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("mae", "r2"),
    metrics_precision: str = "3f",
    unit: str | None = None,
    unit_scale: float = 1.0,
) -> str:
    """
    Format metrics of the `true` and `pred` columns of a DataFrame as LaTeX lines.

    Args:
        df (pd.DataFrame): DataFrame with `true` and `pred` columns.
        metrics (Sequence[str]): Any of `mae`, `mse`, `rmse` and `r2`.
        metrics_precision (str): Format spec of the metric values.
        unit (str | None): Unit of the property.
        unit_scale (float): Factor applied to the error metrics.

    Returns:
        str: One line per metric.
    """
    errors = df["true"] - df["pred"]
    values: dict[str, float] = {}
    for m in metrics:
        m_lower = m.lower()
        if m_lower == "mae":
            values["MAE"] = np.mean(np.abs(errors)) * unit_scale
        elif m_lower == "mse":
            values["MSE"] = np.mean(errors**2) * unit_scale**2
        elif m_lower == "rmse":
            values["RMSE"] = np.sqrt(np.mean(errors**2)) * unit_scale
        elif m_lower == "r2":
            values["R^2"] = 1 - np.sum(errors**2) / np.sum(
                (df["true"] - df["true"].mean()) ** 2
            )
        else:
            raise ValueError(f"Unsupported metric: {m}")

    text_lines: list[str] = []
    for name, val in values.items():
        if name == "R^2":
            text_lines.append(rf"$R^2: {val:.3f}$")
            continue
        unit_str = ""
        if unit:
            unit_str = (
                rf"\,\mathrm{{{unit}}}^2" if name == "MSE" else rf"\,\mathrm{{{unit}}}"
            )
        text_lines.append(rf"$\mathrm{{{name}}}: {val:.{metrics_precision}}{unit_str}$")
    return "\n".join(text_lines)


def plot_hexbin(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes | None = None,
    metrics: Sequence[str] = ("mae", "r2"),
    metrics_precision: str = "3f",
    unit: str | None = None,
    unit_scale: float = 1.0,
    subfigure_label: str | None = None,
    out_png: str | None = None,
) -> None:
    """
    Draw a hexbin parity plot of the `true` and `pred` columns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with `true` and `pred` columns.
        xlabel (str): Label of the x-axis.
        ylabel (str): Label of the y-axis.
        ax (plt.Axes | None): Axes to draw on; a new figure when None.
        metrics (Sequence[str]): Metrics written on the plot, see `metrics_text`.
        metrics_precision (str): Format spec of the metric values.
        unit (str | None): Unit of the property.
        unit_scale (float): Factor applied to the error metrics.
        subfigure_label (str | None): Label written above the metrics.
        out_png (str | None): PNG file to save the figure to.
    """
    with plt.rc_context(PLOT_RC_PARAMS):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        else:
            fig = ax.get_figure()
            if not isinstance(fig, plt.Figure):
                raise TypeError("`ax` must be attached to a top-level Figure")

        hb = ax.hexbin(df["true"], df["pred"], gridsize=40, cmap="viridis", bins="log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)

        # y = x reference line over the current limits
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        min_val, max_val = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        ax.plot([min_val, max_val], [min_val, max_val], "--", color="grey", linewidth=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # density colorbar inside the plot
        cax = inset_axes(
            ax, width="3.5%", height="70%", loc="lower right", borderpad=0.5
        )
        fig.colorbar(hb, cax=cax)
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")

        text = metrics_text(df, metrics, metrics_precision, unit, unit_scale)
        if subfigure_label is not None:
            text = f"{subfigure_label}\n{text}"
        ax.text(0.025, 0.975, text, transform=ax.transAxes, ha="left", va="top")

        if out_png is not None:
            fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
