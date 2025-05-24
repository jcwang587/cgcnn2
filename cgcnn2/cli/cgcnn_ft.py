import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from cgcnn2.data import CIFData, collate_pool, train_force_ratio, train_force_set
from cgcnn2.model import CrystalGraphConvNet
from cgcnn2.util import Normalizer, cgcnn_test, get_lr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def parse_arguments(args=None):
    """
    Parses command-line arguments for the fine-tuning script.

    Args:
        args (list, optional): List of command line arguments to parse. If None, sys.argv[1:] is used.
    """
    parser = argparse.ArgumentParser(
        description="Command-line interface for the CGCNN fine-tuning script."
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        required=True,
        help="Path to the file containing the pre-trained model parameters.",
    )
    parser.add_argument(
        "-as",
        "--full-set",
        type=str,
        help="Path to the directory containing all CIF files for the whole dataset.\n"
        "Training, validation and test ratios are mandatory when using this option.",
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
        "-trfr",
        "--train-force-ratio",
        type=str,
        help="When using the full-set / ratios option, this allows you to force a specific set of cif files to be used for training.\n"
        "The train : valid : test ratio will be kept as is.",
    )
    parser.add_argument(
        "-trfs",
        "--train-force-set",
        type=str,
        help="When using the full-set / ratios option, this allows you to force a specific set of cif files to be used for training.\n"
        "The train : valid : test ratio will not be kept as is.",
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
        help="Epochs for training the model. Default: 1000",
    )
    parser.add_argument(
        "-sp",
        "--stop-patience",
        default=100,
        type=float,
        help="Epochs for early stopping. Default: 100",
    )
    # Learning rate scheduler
    parser.add_argument(
        "-lrp",
        "--lr-patience",
        default=0,
        type=float,
        help="Epochs for reducing learning rate. Default: 0\n"
        "If set to 0, the learning rate scheduler will not be used.",
    )
    parser.add_argument(
        "-lrf",
        "--lr-factor",
        default=0.5,
        type=float,
        help="Factor for reducing learning rate. Default: 0.5\n"
        "If lr-patience is set to 0, this parameter will be ignored.",
    )
    # Advanced fine-tuning options
    parser.add_argument(
        "-r",
        "--reset",
        action="store_true",
        help="Whether to reset (reinitialize) the last fully connected layer or all the fully connected layers.\n"
        "Not specified: Keep the last fully connected layer as is;\n"
        "Specified: Reset the last fully connected layer.",
    )
    parser.add_argument(
        "-tlfc",
        "--train-last-fc",
        action="store_true",
        help="Whether to train on the last fully connected layer or all the fully connected layers.\n"
        "Not specified: Train on all the fully connected layers;\n"
        "Specified: Train on the last fully connected layer.",
    )
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Disable CUDA even if available",
    )
    parser.add_argument(
        "-lrfc",
        "--lr-fc",
        default=0.01,
        type=float,
        help="Learning rate for the layers to be fine-tuned (fully connected layers). Default: 0.01",
    )
    parser.add_argument(
        "-lrnfc",
        "--lr-non-fc",
        default=0.001,
        type=float,
        help="Learning rate for the layers to be frozen (non-fully connected layers). Default: 0.001",
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
        metavar="N",
        help="Batch size for DataLoader (default: 256)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="Number of DataLoader workers (default: 0)",
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
    parser.add_argument(
        "-ji",
        "--job-id",
        type=str,
        default=f"{os.getpid()}",
        help="Job ID for naming output folder (default: <PID>)",
    )

    parsed = parser.parse_args(args if args is not None else sys.argv[1:])
    parsed.device = torch.device(
        "cuda" if not parsed.disable_cuda and torch.cuda.is_available() else "cpu"
    )

    # Warn if dataset ratios don't sum to 1
    total_ratio = parsed.train_ratio + parsed.valid_ratio + parsed.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        warnings.warn(
            "Train ratio + Valid ratio + Test ratio != 1.0", UserWarning, stacklevel=2
        )

    return parsed


def main():
    # Parse command-line arguments
    args = parse_arguments()
    print(f"Using device: {args.device}")

    # Set reproducibility
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare output folder
    output_folder = f"output_{args.job_id}"
    os.makedirs(output_folder, exist_ok=True)

    # Check model_path if specified
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"=> No model params found at '{args.model_path}'")

    # Either load separate sets or split from a full set
    if args.train_set and args.valid_set and args.test_set:
        train_dataset = CIFData(args.train_set)
        valid_dataset = CIFData(args.valid_set)
        test_dataset = CIFData(args.test_set)
    elif args.full_set:
        if args.train_force_set:
            train_dataset, valid_test_dataset = train_force_set(
                args.full_set, args.train_force_set, args.train_ratio, args.random_seed
            )
        elif args.train_force_ratio:
            train_dataset, valid_test_dataset = train_force_ratio(
                args.full_set, args.train_force_ratio, args.train_ratio
            )
        else:
            full_dataset = CIFData(args.full_set)
            train_dataset, valid_test_dataset = train_test_split(
                full_dataset, test_size=(1 - args.train_ratio)
            )

        valid_ratio_adjusted = args.valid_ratio / (1 - args.train_ratio)
        valid_dataset, test_dataset = train_test_split(
            valid_test_dataset, test_size=(1 - valid_ratio_adjusted)
        )

    else:
        raise ValueError(
            "Either train, valid, and test datasets or a full data directory must be provided."
        )

    # Load checkpoint onto device
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_args = argparse.Namespace(**checkpoint["args"])

    # Prepare dataset and infer feature dimensions
    if args.full_set:
        sample_set = args.full_set
    else:
        sample_set = args.train_set

    dataset = CIFData(sample_set)
    atom_graph, _, _ = dataset[0]
    orig_atom_fea_len = atom_graph[0].shape[-1]
    nbr_fea_len = atom_graph[1].shape[-1]

    # Initialize model
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

    normalizer = Normalizer(torch.zeros(3))
    normalizer.load_state_dict(checkpoint["normalizer"])

    print(
        f"=> Loaded model from '{args.model_path}' (epoch {checkpoint['epoch']}, validation error {checkpoint.get('best_mse_error', checkpoint.get('best_mae_error', 'N/A'))})"
    )

    # Initialize DataLoader
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
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=args.device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=args.device.type == "cuda",
    )

    if args.train_last_fc:
        print(
            "* Only the last fully connected layer will be trained (or with higher lr)."
        )

        if args.reset:
            print("* The last fully connected layer will be reset.")
            # Reset the fully connected layers after graph features were obtained
            model.fc_out = nn.Linear(model.fc_out.in_features, 1)
            if args.device.type == "cuda":
                model.fc_out = model.fc_out.cuda()

        # Define parameters to be fine-tuned
        fc_parameters = [param for param in model.fc_out.parameters()]

    else:
        print("* All the fully connected layers will be trained (or with higher lr).")

        if args.reset:
            # Reset all the fully connected layers
            print("* All the fully connected layers will be reset.")
            model.conv_to_fc = nn.Linear(
                model.conv_to_fc.in_features, model.conv_to_fc.out_features
            )
            if args.device.type == "cuda":
                model.conv_to_fc = model.conv_to_fc.cuda()

            if hasattr(model, "fcs"):
                model.fcs = nn.ModuleList(
                    [
                        nn.Linear(layer.in_features, layer.out_features)
                        for layer in model.fcs
                    ]
                )
                if args.device.type == "cuda":
                    model.fcs = nn.ModuleList([layer.cuda() for layer in model.fcs])

            model.fc_out = nn.Linear(model.fc_out.in_features, 1)
            if args.device.type == "cuda":
                model.fc_out = model.fc_out.cuda()

        # Define parameters to be trained
        fc_parameters = [param for param in model.conv_to_fc.parameters()]
        if hasattr(model, "fcs"):
            for fc in model.fcs:
                fc_parameters += [param for param in fc.parameters()]
        fc_parameters += [param for param in model.fc_out.parameters()]

    # Get ids of the fc_parameters
    fc_parameters_ids = set(map(id, fc_parameters))

    other_parameters = [
        param for param in model.parameters() if id(param) not in fc_parameters_ids
    ]

    # Define optimizer with differential learning rates
    optimizer = torch.optim.Adam(
        [
            {"params": fc_parameters, "lr": args.lr_fc},
            {"params": other_parameters, "lr": args.lr_non_fc},
        ]
    )

    # Initialize the scheduler
    scheduler = None

    # Define the loss function
    criterion = nn.MSELoss(reduction="none")

    # Define a learning rate scheduler
    if args.lr_patience:
        lr_patience = int(float(args.lr_patience))
        factor = float(args.lr_factor)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=int(lr_patience)
        )
        print(
            "* Learning rate adjustment is configured with a factor of {} and patience of {} epochs.".format(
                factor, lr_patience
            )
        )
    else:
        lr_patience = None
        print("* The training will proceed with a fixed learning rate.")

    # Training epochs
    num_epochs = int(float(args.epoch))
    best_valid_loss = float("inf")

    # Set the patience for early stopping
    stop_patience = int(float(args.stop_patience))
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # --------------------
        # TRAIN
        # --------------------
        model.train()
        train_loss = 0.0
        for i, (input, targets, _) in enumerate(train_loader):
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
            atom_fea = atom_fea.to(args.device)
            nbr_fea = nbr_fea.to(args.device)
            nbr_fea_idx = nbr_fea_idx.to(args.device)
            crystal_atom_idx = [idx_map.to(args.device) for idx_map in crystal_atom_idx]
            targets = targets.to(args.device)

            # Forward pass
            outputs, _ = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss = criterion(outputs, targets)
            if args.bias_temperature > 0.0:
                # Boltzmann factor weighting
                bias = torch.exp(-targets / args.bias_temperature).to(args.device)
                loss = (loss * bias).mean()
            else:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add the loss to the training loss
            train_loss += loss.item()

        # --------------------
        # VALIDATION
        # --------------------
        model.eval()
        valid_loss = 0.0
        with torch.inference_mode():
            for i, (input, targets, _) in enumerate(valid_loader):
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
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

                valid_loss += loss.item()

        # Print average training / validation loss per epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        if args.lr_patience:
            scheduler.step(avg_valid_loss)

        lr = get_lr(optimizer)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.5f}, Valid Loss: {avg_valid_loss:.5f}, LR: {str(lr)}"
        )

        # --------------------
        # CHECKPOINTING
        # --------------------
        if avg_valid_loss < best_valid_loss:

            savepoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "normalizer": normalizer.state_dict(),
                "best_mse_error": avg_valid_loss,
                "args": vars(model_args),
            }
            torch.save(savepoint, os.path.join(output_folder, "best_model.ckpt"))
            best_valid_loss = avg_valid_loss
            epochs_without_improvement = 0
            print(f" [SAVE] Best model at epoch {epoch + 1}")
        else:
            epochs_without_improvement += 1

        # If the validation loss hasn't improved in 'stop_patience' epochs, stop training
        if epochs_without_improvement == stop_patience:
            print(f"Early stopping after {stop_patience} epochs without improvement.")
            break

    print("Training completed.")

    # --------------------
    # TEST WITH BEST MODEL
    # --------------------
    checkpoint = torch.load(
        os.path.join(output_folder, "best_model.ckpt"), weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])

    # Test the model
    cgcnn_test(
        model,
        test_loader,
        args.device,
        plot_file=os.path.join(output_folder, "parity_plot.svg"),
        results_file=os.path.join(output_folder, "test_results.csv"),
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        axis_limits=args.axis_limits,
    )


if __name__ == "__main__":
    main()
