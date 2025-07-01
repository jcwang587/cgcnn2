import argparse
import logging
import os
import sys
from pprint import pformat
from random import sample

import torch
import torch.nn as nn
from cgcnn2.data import CIFData, collate_pool, full_set_split
from cgcnn2.model import CrystalGraphConvNet
from cgcnn2.util import Normalizer, cgcnn_test, get_lr, seed_everything, setup_logging
from torch.utils.data import DataLoader


def parse_arguments(args=None):
    """
    Parses command-line arguments for the CGCNN training script.

    Args:
        args (list, optional): List of command-line arguments to parse. If None, sys.argv[1:] is used.
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for the CGCNN training script."
    )
    # Dataset arguments
    parser.add_argument(
        "-as",
        "--full-set",
        type=str,
        help="Path to the directory containing all CIF files for the entire dataset.\n"
        "Training, validation, and test ratios are mandatory when using this option.",
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
        help="Ratio of the dataset for training. Default: 0.6",
    )
    parser.add_argument(
        "-trfs",
        "--train-force-set",
        default=None,
        type=str,
        help="When using the combined full-set and ratios option,\n"
        "this allows you to force a specific set to be used for training.\n"
        "The train : valid : test ratio will be not be kept as is. Default: None",
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
    # Early stopping scheduler
    parser.add_argument(
        "-e",
        "--epoch",
        default=1000,
        type=float,
        help="Number of epochs for training. Default: 1000",
    )
    parser.add_argument(
        "-sp",
        "--stop-patience",
        default=None,
        type=float,
        help="Number of epochs for early stopping patience. Default: None (no early stopping)",
    )

    # Learning rate & scheduler
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-2,
        type=float,
        help="Learning rate for all parameters. Default: 1e-2",
    )
    parser.add_argument(
        "-lrp",
        "--lr-patience",
        default=0,
        type=float,
        help="Patience (in epochs) for reducing LR if validation loss stalls. Default: 0 (disabled).",
    )
    parser.add_argument(
        "-lrf",
        "--lr-factor",
        default=0.5,
        type=float,
        help="Factor by which LR is reduced when LR scheduler is triggered. Default: 0.5.\n"
        "Ignored if lr-patience=0.",
    )
    # Advanced training options
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Disable CUDA even if available",
    )
    parser.add_argument(
        "-rs",
        "--random-seed",
        default=42,
        type=int,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=256,
        type=int,
        help="Batch size for DataLoader (default: 256)",
    )
    parser.add_argument(
        "-cs",
        "--cache-size",
        default=None,
        type=int,
        help="Cache size for training DataLoader (default: None), which is unlimited",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        help="Number of DataLoader workers (default: 0)",
    )
    parser.add_argument(
        "-bt",
        "--bias-temperature",
        default=-1.0,
        type=float,
        help="If set > 0, apply a Boltzmann-like factor weighting in the loss.\n"
        "Smaller values favor low-energy structures more strongly.",
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
    # Model hyperparameters
    parser.add_argument(
        "--task",
        default="regression",
        choices=["regression", "classification"],
        help="Task type: 'regression' or 'classification'. Default: regression",
    )
    parser.add_argument(
        "--atom-fea-len",
        default=64,
        type=int,
        help="Number of hidden features for each atom. Default: 64",
    )
    parser.add_argument(
        "--n-conv",
        default=3,
        type=int,
        help="Number of convolutional layers in the CrystalGraphConvNet. Default: 3",
    )
    parser.add_argument(
        "--h-fea-len",
        default=128,
        type=int,
        help="Number of hidden features after pooling crystal-wise. Default: 128",
    )
    parser.add_argument(
        "--n-h",
        default=1,
        type=int,
        help="Number of hidden layers after pooling crystal-wise. Default: 1",
    )

    parsed = parser.parse_args(args if args is not None else sys.argv[1:])
    parsed.device = torch.device(
        "cuda" if not parsed.disable_cuda and torch.cuda.is_available() else "cpu"
    )

    # Warn if dataset ratios don't sum to 1
    total_ratio = parsed.train_ratio + parsed.valid_ratio + parsed.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logging.warning("Train ratio + Valid ratio + Test ratio != 1.0")

    return parsed


def main():
    setup_logging()

    # Parse command-line arguments
    args = parse_arguments()
    logging.info(f"Parsed arguments:\n{pformat(vars(args))}")

    # Set the seed for reproducibility
    seed_everything(args.random_seed)

    # Create the output folder
    output_folder = f"output_{args.job_id}"
    os.makedirs(output_folder, exist_ok=True)

    # Load separate datasets or split from a full set
    if args.cache_size:
        logging.info(f"Using cache size: {args.cache_size} for DataLoader")
    if args.train_set and args.valid_set and args.test_set:
        if args.full_set:
            logging.error("Cannot specify both full-set and train, valid, test sets.")
            sys.exit(1)
        train_dataset = CIFData(args.train_set, cache_size=args.cache_size)
        valid_dataset = CIFData(args.valid_set)
        test_dataset = CIFData(args.test_set)
    elif args.full_set:
        if args.train_set or args.valid_set or args.test_set:
            logging.error("Cannot specify both full-set and train, valid, test sets.")
            sys.exit(1)
        train_set_dir, valid_set_dir, test_set_dir = full_set_split(
            args.full_set, args.train_ratio, args.valid_ratio, args.train_force_set
        )
        train_dataset = CIFData(train_set_dir, cache_size=args.cache_size)
        valid_dataset = CIFData(valid_set_dir)
        test_dataset = CIFData(test_set_dir)
    else:
        logging.error(
            "Either train, valid, and test datasets or a full data directory must be provided."
        )
        sys.exit(1)

    full_dataset = [*train_dataset, *valid_dataset, *test_dataset]

    # Normalizer setup
    # For classification we use a dummy normalizer, otherwise compute mean/std from data
    if args.task == "classification":
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
    else:
        if len(full_dataset) < 500:
            logging.warning(
                "Dataset has fewer than 500 data points; results may have higher variance."
            )
            sample_data_list = [full_dataset[i] for i in range(len(full_dataset))]
        else:
            sample_indices = sample(range(len(full_dataset)), 500)
            sample_data_list = [full_dataset[i] for i in sample_indices]

        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # Build model
    # 1) gather input dimensions from first sample
    structures, _, _ = train_dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    # 2) instantiate CGCNN
    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=(args.task == "classification"),
    )

    # Move to device
    model.to(args.device)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=args.device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=args.device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=args.device.type == "cuda",
    )

    # Single LR optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Optional LR scheduler
    scheduler = None
    if args.lr_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor, patience=int(args.lr_patience)
        )
        logging.info(
            f"[Scheduler] factor={args.lr_factor}, patience={int(args.lr_patience)}"
        )

    # Training epochs
    num_epochs = int(float(args.epoch))
    best_valid_loss = float("inf")
    criterion = nn.MSELoss(reduction="none")

    # Set the patience for early stopping
    stop_patience = int(float(args.stop_patience)) if args.stop_patience else None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # --------------------
        # TRAIN
        # --------------------
        model.train()
        train_loss_sum = 0.0
        for input_data, targets, _ in train_loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
            atom_fea = atom_fea.to(args.device)
            nbr_fea = nbr_fea.to(args.device)
            nbr_fea_idx = nbr_fea_idx.to(args.device)
            crystal_atom_idx = [idx_map.to(args.device) for idx_map in crystal_atom_idx]
            targets = targets.to(args.device)

            optimizer.zero_grad()

            outputs, _ = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss = criterion(outputs, targets)

            if args.bias_temperature > 0.0:
                # Boltzmann factor weighting
                bias = torch.exp(-targets / args.bias_temperature).to(args.device)
                loss = (loss * bias).mean()
            else:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # --------------------
        # VALIDATION
        # --------------------
        model.eval()
        valid_loss_sum = 0.0
        with torch.inference_mode():
            for input_data, targets, _ in valid_loader:
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input_data
                atom_fea = atom_fea.to(args.device)
                nbr_fea = nbr_fea.to(args.device)
                nbr_fea_idx = nbr_fea_idx.to(args.device)
                crystal_atom_idx = [
                    idx_map.to(args.device) for idx_map in crystal_atom_idx
                ]
                targets = targets.to(args.device)

                outputs, _ = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                loss = criterion(outputs, targets)

                if args.bias_temperature > 0.0:
                    # Boltzmann factor weighting
                    bias = torch.exp(-targets / args.bias_temperature).to(args.device)
                    loss = (loss * bias).mean()
                else:
                    loss = loss.mean()

                valid_loss_sum += loss.item()

        avg_valid_loss = valid_loss_sum / len(valid_loader)

        # Scheduler step
        if scheduler is not None:
            scheduler.step(avg_valid_loss)

        current_lr = get_lr(optimizer)
        logging.info(
            f"Epoch [{epoch + 1:03d}/{num_epochs}] - "
            f"Train Loss: {avg_train_loss:.5f}, "
            f"Valid Loss: {avg_valid_loss:.5f}, "
            f"LR: {current_lr}"
        )

        # --------------------
        # CHECKPOINTING
        # --------------------
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            epochs_without_improvement = 0

            savepoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "normalizer": normalizer.state_dict(),
                "best_mse_error": avg_valid_loss,
                "args": vars(args),
            }
            ckpt_path = os.path.join(output_folder, "best_model.ckpt")
            torch.save(savepoint, ckpt_path)
            logging.info(f"  [SAVE] Best model at epoch {epoch + 1}.")
        else:
            if stop_patience:
                epochs_without_improvement += 1

        # Early stopping
        if stop_patience and epochs_without_improvement >= stop_patience:
            logging.info(
                f"Early stopping after {stop_patience} epochs without improvement."
            )
            break

    logging.info("Training complete.")

    # --------------------
    # TEST WITH BEST MODEL
    # --------------------
    best_checkpoint = torch.load(os.path.join(output_folder, "best_model.ckpt"))
    model.load_state_dict(best_checkpoint["state_dict"])
    normalizer.load_state_dict(best_checkpoint["normalizer"])

    cgcnn_test(
        model,
        test_loader,
        args.device,
        plot_file=os.path.join(output_folder, "parity_plot.png"),
        results_file=os.path.join(output_folder, "test_results.csv"),
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        axis_limits=args.axis_limits,
    )


if __name__ == "__main__":
    main()
