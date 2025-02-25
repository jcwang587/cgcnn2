import os
import csv
import sys
import glob
import torch
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from pymatviz import density_hexbin

from torch.utils.data import DataLoader
from .cgcnn_data import CIFData_pred, collate_pool
from .cgcnn_model import CrystalGraphConvNet, Normalizer


def output_id_gen():
    """
    This function obtains the current date and time, formats it as 'mmdd_HHMM',
    and prepends 'output_' to form a unique identifier. This can be useful
    for creating distinct output folder names or filenames at runtime.

    Returns:
        - str: A string that represents the current date and time in the format of 'output_mmdd_HHMM'.
    """

    now = datetime.now()
    # Format time to match desired format (mmdd_HHMM)
    timestamp = now.strftime("%m%d_%H%M")
    # Prepend 'output_' to timestamp to form folder name
    folder_name = f"output_{timestamp}"

    return folder_name


def id_prop_gen(cif_dir):
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


def get_lr(optimizer):
    """
    This function iterates over the parameter groups of a given PyTorch optimizer,
    extracting the learning rate from each group. The learning rates are then returned in a list.

    Parameters:
        - optimizer (torch.optim.Optimizer): The PyTorch optimizer to extract learning rates from.

    Returns:
        - list: A list of learning rates, one for each parameter group in the optimizer.
    """

    return [param_group["lr"] for param_group in optimizer.param_groups]


def extract_fea(model, loader, device):
    """
    Applies a trained model to a dataset to extract learned feature
    representations, targets, and CIF IDs, returning these as tensors.

    Parameters:
        - model (torch.nn.Module): The trained model.
        - loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        - device (str): The device ('cuda' or 'cpu') to send tensors to.

    Returns:
        - tuple (torch.Tensor, torch.Tensor, list): A tuple where the first element is
        the tensor of extracted features, the second element is the tensor of targets,
        and the third is a list of CIF IDs.
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
    model,
    loader,
    device,
    plot_file="parity_plot.svg",
    results_file="results.csv",
    axis_limits=None,
    **kwargs,
):
    """
    This function tests a trained machine learning model on a provided dataset, calculates the Mean Squared Error
    (MSE) and R2 score, and prints these results. It also saves the prediction results as a CSV file and
    generates a parity plot as an SVG file. The plot displays the model's predictions versus the actual values,
    color-coded by the point density.

    Parameters:
        - model (torch.nn.Module): The trained model.
        - loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        - device (str): The device ('cuda' or 'cpu') where the model will be run.
        - plot_file (str, optional): The file path where the parity plot will be saved. Defaults to 'parity_plot.svg'.
        - results_file (str, optional): The file path where the results will be saved as a CSV file. Defaults to 'results.csv'.
        - axis_limits (list, optional): The limits for the x and y axes of the parity plot. Defaults to None.
        - **kwargs:
            - xlabel (str): x-axis label for the parity plot. Defaults to "Actual".
            - ylabel (str): y-axis label for the parity plot. Defaults to "Predicted".
            - Any other extra keyword arguments you want to pass in.
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


def pred_calculator(
    model,
    loader,
    device,
    verbose,
):
    """
    This function applies a trained model to a dataset, returning the model's predictions and the last layer's output.

    Parameters:
        - model (torch.nn.Module): The trained model.
        - loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        - device (str): The device ('cuda' or 'cpu') where the model will be run.
        - verbose (int): The verbosity level of the output.
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


def cgcnn_pred(model_path, all_set, verbose=3, cuda=False, num_workers=0):
    """
    This function loads a trained CGCNN model from a file, applies it to a dataset, and returns the model's predictions
    and the last layer's output.

    Parameters:
        - model_path (str): The path to the file containing the trained model parameters.
        - all_set (str): The path to the directory containing all CIF files for the dataset.
        - verbose (int): The verbosity level of the output.
        - cuda (bool): Set to True to use CUDA, False to use CPU.
        - num_workers (int): The number of subprocesses to use for data loading.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"=> No model params found at '{model_path}'")

    total_dataset = CIFData_pred(all_set)

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

    pred, last_layer = pred_calculator(model, full_loader, device, verbose)

    return pred, last_layer


def parse_arguments():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for the Crystal Graph Convolutional Neural Network (CGCNN) model."
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        help="Path to the file containing the trained model parameters.",
    )
    parser.add_argument(
        "-as",
        "--total-set",
        type=str,
        help="Path to the directory containing all CIF files for the dataset.",
    )
    parser.add_argument(
        "-trs",
        "--train-set",
        type=str,
        help="Path to the directory containing CIF files for the train dataset.",
    )
    parser.add_argument(
        "-vs",
        "--valid-set",
        type=str,
        help="Path to the directory containing CIF files for the validation dataset.",
    )
    parser.add_argument(
        "-ts",
        "--test-set",
        type=str,
        help="Path to the directory containing CIF files for the test dataset.",
    )
    parser.add_argument(
        "-trr",
        "--train-ratio",
        default=0.6,
        type=float,
        help="The ratio of the dataset to be used for training. Default: 0.6",
    )
    parser.add_argument(
        "-vr",
        "--valid-ratio",
        default=0.2,
        type=float,
        help="The ratio of the dataset to be used for validation. Default: 0.2",
    )
    parser.add_argument(
        "-tr",
        "--test-ratio",
        default=0.2,
        type=float,
        help="The ratio of the dataset to be used for testing. Default: 0.2",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        default=10000,
        type=float,
        help="Total epochs for training the model.",
    )
    parser.add_argument(
        "-sp",
        "--stop-patience",
        default=100,
        type=float,
        help="Epochs for early stopping.",
    )
    parser.add_argument(
        "-lrp",
        "--lr-patience",
        default=0,
        type=float,
        help="Epochs for reducing learning rate.",
    )
    parser.add_argument(
        "-lrf",
        "--lr-factor",
        default=0.0,
        type=float,
        help="Factor for reducing learning rate.",
    )
    parser.add_argument(
        "-tlfc",
        "--train-last-fc",
        default=0,
        type=int,
        help="Train on the last fully connected layer or all the fully connected layers",
    )
    parser.add_argument(
        "-lrfc",
        "--lr-fc",
        default=0.01,
        type=float,
        help="Learning rate for training the last fully connected layer.",
    )
    parser.add_argument(
        "-lrnfc",
        "--lr-non-fc",
        default=0.001,
        type=float,
        help="Learning rate for training the non-last fully connected layers.",
    )
    parser.add_argument(
        "-rs", "--random-seed", default=123, type=int, help="Random seed."
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="The size of each batch during training or testing. Default: 256",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="The number of subprocesses to use for data loading. Default: 0",
    )
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Set this flag to disable CUDA, even if it is available.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default=1,
        type=int,
        help="Set to 1 to train the model, or 0 to test the model. Default: 1",
    )
    parser.add_argument(
        "-ji", "--job-id", default=None, type=str, help="The id of the current job."
    )
    parser.add_argument(
        "-r",
        "--replace",
        default=1,
        type=int,
        help="Replace the training layer to restart.",
    )
    parser.add_argument(
        "-bt",
        "--bias-temperature",
        default=0.0,
        type=float,
        help=(
            "If set > 0, bias the loss function using a Boltzmann-like factor.\n"
            "Smaller 'bias_temperature' strongly favors low-energy structures.\n"
            "Larger 'bias_temperature' reduces the low-energy bias.\n"
            "If not specified or non-positive, no bias is applied."
        ),
    )
    parser.add_argument(
        "-al",
        "--axis-limits",
        nargs=2,
        default=None,
        type=float,
        help="The limits for the x and y axes of the parity plot.",
    )

    args = parser.parse_args(sys.argv[1:])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    # Warning if train ratio and test ratio don't sum to 1
    if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1) > 1e-6:
        print("Warning: Train ratio, Valid ratio and Test ratio do not sum up to 1")

    return args
