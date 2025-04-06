import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch
from cgcnn2 import CIFData, CrystalGraphConvNet, cgcnn_test, collate_pool
from torch.utils.data import DataLoader


def parse_arguments(args=None):
    """
    Parses command-line arguments for the script.

    Parameters
    ----------
    args : list, optional
        List of command line arguments to parse. If None, sys.argv[1:] is used.
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
        "-trrfs",
        "--train-ratio-force-set",
        type=str,
        help="Under the setting of input training dataset using train_ratio, this option allows you to force a specific set of cif files to be used for training.",
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
    # Learning rate scheduler
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
    # Advanced fine-tuning options
    parser.add_argument(
        "-r",
        "--replace",
        default=1,
        type=int,
        help="Replace the training layer to restart.",
    )
    parser.add_argument(
        "-tlfc",
        "--train-last-fc",
        action="store_true",
        help="Train on the last fully connected layer or all the fully connected layers. Default: False",
    )
    parser.add_argument(
        "-lrfc",
        "--lr-fc",
        default=0.01,
        type=float,
        help="Learning rate for training the last fully connected layer. Default: 0.01",
    )
    parser.add_argument(
        "-lrnfc",
        "--lr-non-fc",
        default=0.001,
        type=float,
        help="Learning rate for training the non-last fully connected layers. Default: 0.001",
    )
    parser.add_argument(
        "-rs",
        "--random-seed",
        default=42,
        type=int,
        help="Random seed for reproducibility. Default: 42",
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
        help="Force disable CUDA, even if a compatible GPU is available. Default: False",
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
    parser.add_argument(
        "-ji",
        "--job-id",
        default=None,
        type=str,
        help="The id of the current job. stdout and stderr files will be saved to the output folder.",
    )

    parsed_args = parser.parse_args(args if args is not None else sys.argv[1:])
    parsed_args.cuda = not parsed_args.disable_cuda and torch.cuda.is_available()

    # Warning if train ratio and test ratio don't sum to 1
    if (
        abs(
            parsed_args.train_ratio
            + parsed_args.valid_ratio
            + parsed_args.test_ratio
            - 1
        )
        > 1e-6
    ):
        warnings.warn(
            "Train ratio, Valid ratio and Test ratio do not sum up to 1",
            UserWarning,
            stacklevel=2,
        )

    return parsed_args


def main():
    # Parse command-line arguments
    args = parse_arguments()
    print(args)

    # Set the seed for reproducibility
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create the output folder
    output_folder = "output_" + args.job_id

    # Validate the existence of the model file
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"=> No model params found at '{args.model_path}'")

    if args.total_set:
        total_dataset = CIFData(args.total_set)
    else:
        raise ValueError("Total dataset must be provided in prediction mode.")

    # Instantiate the CrystalGraphConvNet model using parameters from the checkpoint
    checkpoint = torch.load(
        args.model_path,
        map_location=lambda storage, loc: storage if not args.cuda else None,
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
    if args.cuda:
        model.cuda()

    device = "cuda" if args.cuda else "cpu"
    model.to(device).eval()

    full_loader = DataLoader(
        dataset=total_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=args.cuda,
    )

    # In predict mode, make predictions
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    # Test the model
    cgcnn_test(
        model,
        full_loader,
        device,
        plot_file=os.path.join(output_folder, "parity_plot_test_mode.svg"),
        results_file=os.path.join(output_folder, "test_results_test_mode.csv"),
        xlabel="Actual (eV)",
        ylabel="Predicted (eV)",
        axis_limits=args.axis_limits,
    )
