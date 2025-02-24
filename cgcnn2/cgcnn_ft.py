# Python Standard Library
import os
import random
import argparse
import warnings

# Third-party Libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# Local Application / Specific Library Imports
from cgcnn2 import (
    CrystalGraphConvNet,
    Normalizer,
    collate_pool,
    CIFData,
    get_lr,
    parse_arguments,
    cgcnn_test,
)


# Suppress specific warnings from pymatgen
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")


def main():
    # Parse command-line arguments
    args = parse_arguments()
    print(args)

    # Set the seed for reproducibility
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Validate the existence of the model file
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"=> No model params found at '{args.model_path}'")

    # Depending on the arguments, either load the separate datasets or split the total data
    if args.train_set and args.valid_set and args.test_set:
        # Load data from CIF files
        train_dataset = CIFData(args.train_set)
        valid_dataset = CIFData(args.valid_set)
        test_dataset = CIFData(args.test_set)
    elif args.total_set:
        # Load data from CIF files
        total_dataset = CIFData(args.total_set)
        # Split data into train and (temporary) test
        train_dataset, temp_dataset = train_test_split(
            total_dataset, test_size=(1 - args.train_ratio)
        )
        # Split the temporary test dataset into validation and test
        valid_ratio_adjusted = args.valid_ratio / (1 - args.train_ratio)
        valid_dataset, test_dataset = train_test_split(
            temp_dataset, test_size=(1 - valid_ratio_adjusted)
        )

    else:
        raise ValueError(
            "Either train, valid, and test datasets or a total data directory must be provided."
        )

    # Instantiate the CrystalGraphConvNet model using parameters from the checkpoint
    checkpoint = torch.load(
        args.model_path,
        map_location=lambda storage, loc: storage if not args.cuda else None,
        weights_only=False,
    )
    structures, _, _ = train_dataset[0]
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

    # Load the normalizer and model weights from the checkpoint
    normalizer = Normalizer(torch.zeros(3))
    normalizer.load_state_dict(checkpoint["normalizer"])
    model.load_state_dict(checkpoint["state_dict"])

    print(
        f"=> Loaded model from '{args.model_path}' (epoch {checkpoint['epoch']}, validation error {checkpoint['best_mae_error']})"
    )

    # Depending on the mode (train or test), either train the model or make predictions
    device = "cuda" if args.cuda else "cpu"
    model.to(device).eval()

    output_folder = "output_" + args.job_id

    if args.mode:
        # Initialize DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_pool,
            pin_memory=args.cuda,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_pool,
            pin_memory=args.cuda,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_pool,
            pin_memory=args.cuda,
        )

        if args.train_last_fc:
            print(
                "* Only the last fully connected layer will be trained (or with higher lr)."
            )

            if args.replace:
                print(
                    "* The last fully connected layer will be replaced to be cleaned."
                )
                # Replace the fully connected layers after graph features were obtained
                model.fc_out = nn.Linear(model.fc_out.in_features, 1)
                if args.cuda:
                    model.fc_out = model.fc_out.cuda()

            # Define parameters to be trained
            fc_parameters = [param for param in model.fc_out.parameters()]

        else:
            print(
                "* All the fully connected layers will be trained (or with higher lr)."
            )

            if args.replace:
                # Replace conv_to_fc layer
                print(
                    "* All the fully connected layers will be replaced to be cleaned."
                )
                model.conv_to_fc = nn.Linear(
                    model.conv_to_fc.in_features, model.conv_to_fc.out_features
                )
                if args.cuda:
                    model.conv_to_fc = model.conv_to_fc.cuda()

                # If present, replace the fcs layers
                if hasattr(model, "fcs"):
                    model.fcs = nn.ModuleList(
                        [
                            nn.Linear(layer.in_features, layer.out_features)
                            for layer in model.fcs
                        ]
                    )
                    if args.cuda:
                        model.fcs = nn.ModuleList([layer.cuda() for layer in model.fcs])

                # Replace fc_out layer
                model.fc_out = nn.Linear(model.fc_out.in_features, 1)
                if args.cuda:
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
        criterion = nn.MSELoss(reduction="none")  # Returns a per-sample loss vector

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
            model.train()
            train_loss = 0.0
            for i, (input, targets, _) in enumerate(train_loader):
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
                atom_fea = atom_fea.to(device)
                nbr_fea = nbr_fea.to(device)
                nbr_fea_idx = nbr_fea_idx.to(device)
                crystal_atom_idx = [idx_map.to(device) for idx_map in crystal_atom_idx]
                targets = targets.to(device)

                # Forward pass
                outputs, _ = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                loss = criterion(outputs, targets)
                if args.bias_temperature > 0.0:
                    # shape [batch_size], each sample has its own bias weight
                    bias = torch.exp(-targets / args.bias_temperature).to(device)
                    # Weighted average across the batch
                    loss = (loss * bias).mean()
                else:
                    # Unweighted average across the batch
                    loss = loss.mean()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Add the loss to the training loss
                train_loss += loss.item()

            # Start of the validation loop
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for i, (input, targets, _) in enumerate(valid_loader):
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
                    atom_fea = atom_fea.to(device)
                    nbr_fea = nbr_fea.to(device)
                    nbr_fea_idx = nbr_fea_idx.to(device)
                    crystal_atom_idx = [
                        idx_map.to(device) for idx_map in crystal_atom_idx
                    ]
                    targets = targets.to(device)

                    # Forward pass
                    outputs, _ = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                    loss = criterion(outputs, targets)

                    if args.bias_temperature > 0.0:
                        # Per-sample Boltzmann factor weighting
                        bias = torch.exp(-targets / args.bias_temperature).to(device)
                        loss = (loss * bias).mean()
                    else:
                        # Just average the per-sample losses
                        loss = loss.mean()

                    valid_loss += loss.item()

            # Print average training / validation loss per epoch
            avg_train_loss = train_loss / len(train_loader)
            avg_valid_loss = valid_loss / len(valid_loader)

            if args.lr_patience:
                scheduler.step(avg_valid_loss)

            lr = get_lr(optimizer)
            print(
                f"| Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.5f}, Validation Loss: {avg_valid_loss:.5f}, Learning Rates: {str(lr)}"
            )

            # Check if the validation loss improved
            if avg_valid_loss < best_valid_loss:
                # Create a dictionary to save all necessary information
                savepoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "best_mae_error": avg_valid_loss,
                    "args": vars(model_args),
                }
                torch.save(savepoint, os.path.join(output_folder, "best_model.ckpt"))
                best_valid_loss = avg_valid_loss
                epochs_without_improvement = 0
                print(f"✔️ New best model saved for epoch {epoch + 1}")
            else:
                epochs_without_improvement += 1

            # If the validation loss hasn't improved in 'stop_patience' epochs, stop training
            if epochs_without_improvement == stop_patience:
                print(
                    f"Early stopping after {stop_patience} epochs without improvement."
                )
                break

        print("✔️ Training completed.")

        # Load the best model
        checkpoint = torch.load(
            os.path.join(output_folder, "best_model.ckpt"), weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])

        # Test the model
        cgcnn_test(
            model,
            test_loader,
            device,
            plot_file=os.path.join(output_folder, "parity_plot.svg"),
            results_file=os.path.join(output_folder, "test_results.csv"),
            xlabel="Actual (eV)",
            ylabel="Predicted (eV)",
            axis_limits=args.axis_limits,
        )

    else:
        # Prediction mode
        if args.total_set:
            total_dataset = CIFData(args.total_set)
        else:
            raise ValueError("Total dataset must be provided in prediction mode.")

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


if __name__ == "__main__":
    main()
