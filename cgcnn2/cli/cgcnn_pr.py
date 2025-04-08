import argparse
import os
import random
import sys

import numpy as np
import torch
from cgcnn2.data import CIFData, collate_pool
from cgcnn2.model import CrystalGraphConvNet
from cgcnn2.util import cgcnn_test
from torch.utils.data import DataLoader


def parse_arguments(args=None):
    """
    Parses command-line arguments for the prediction script.

    Parameters
    ----------
    args : list, optional
        List of command line arguments to parse. If None, sys.argv[1:] is used.
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for the CGCNN prediction script."
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        help="Path to the file containing the pre-trained model parameters.",
    )
    parser.add_argument(
        "-as",
        "--full-set",
        type=str,
        help="Path to the directory containing all CIF files for the prediction dataset.",
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

    if args.full_set:
        full_dataset = CIFData(args.full_set)
    else:
        raise ValueError("Full dataset must be provided in prediction mode.")

    # Instantiate the CrystalGraphConvNet model using parameters from the checkpoint
    checkpoint = torch.load(
        args.model_path,
        map_location=lambda storage, loc: storage if not args.cuda else None,
        weights_only=False,
    )
    structures, _, _ = full_dataset[0]
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
        dataset=full_dataset,
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
