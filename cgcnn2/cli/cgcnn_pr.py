import argparse
import logging
import os
import sys

import torch
from cgcnn2.data import CIFData, collate_pool
from cgcnn2.model import CrystalGraphConvNet
from cgcnn2.util import (
    cgcnn_test,
    print_checkpoint_info,
    seed_everything,
    setup_logging,
)
from torch.utils.data import DataLoader


def parse_arguments(args=None):
    """
    Parses command-line arguments for the CGCNN prediction script.

    Args:
        args (list, optional): List of command-line arguments to parse. If None, sys.argv[1:] is used.
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for the CGCNN prediction script."
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file",
    )
    parser.add_argument(
        "-as",
        "--full-set",
        type=str,
        required=True,
        help="Path to the directory containing CIF files for prediction",
    )
    parser.add_argument(
        "-rs",
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for DataLoader (default: 256)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0)",
    )
    parser.add_argument(
        "--disable-cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument(
        "-ji",
        "--job-id",
        type=str,
        default=f"{os.getpid()}",
        help="Job ID for naming output folder (default: <PID>)",
    )
    # Parity plot options
    parser.add_argument(
        "-al",
        "--axis-limits",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Axis limits for the parity plot",
    )
    parser.add_argument(
        "-x",
        "--xlabel",
        type=str,
        default="Actual",
        help="X-axis label for the parity plot",
    )
    parser.add_argument(
        "-y",
        "--ylabel",
        type=str,
        default="Predicted",
        help="Y-axis label for the parity plot",
    )

    parsed = parser.parse_args(args if args is not None else sys.argv[1:])
    parsed.device = torch.device(
        "cuda" if not parsed.disable_cuda and torch.cuda.is_available() else "cpu"
    )
    return parsed


def main():
    setup_logging()
    # Parse command-line arguments
    args = parse_arguments()

    # Set seeds for reproducibility
    seed_everything(args.random_seed)

    # Validate paths
    if not os.path.isfile(args.model_path):
        logging.error(f"No model checkpoint found at '{args.model_path}'")
        sys.exit(1)
    if not os.path.isdir(args.full_set):
        logging.error(f"Dataset directory '{args.full_set}' does not exist")
        sys.exit(1)

    # Prepare output folder
    output_folder = f"output_{args.job_id}"
    os.makedirs(output_folder, exist_ok=True)

    # Load checkpoint onto device
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_args = argparse.Namespace(**checkpoint["args"])

    # Prepare dataset and infer feature dimensions
    dataset = CIFData(args.full_set)
    atom_graph, _, _ = dataset[0]
    orig_atom_fea_len = atom_graph[0].shape[-1]
    nbr_fea_len = atom_graph[1].shape[-1]

    # Build DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=(args.device.type == "cuda"),
    )

    # Initialize and load model
    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=model_args.atom_fea_len,
        n_conv=model_args.n_conv,
        h_fea_len=model_args.h_fea_len,
        n_h=model_args.n_h,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    model.eval()

    print_checkpoint_info(checkpoint, args.model_path)

    # Run inference and save results
    cgcnn_test(
        model=model,
        loader=loader,
        device=args.device,
        plot_file=os.path.join(output_folder, "parity_plot.png"),
        results_file=os.path.join(output_folder, "results.csv"),
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        axis_limits=args.axis_limits,
    )


if __name__ == "__main__":
    main()
