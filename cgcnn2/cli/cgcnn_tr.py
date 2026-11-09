"""Train a CGCNN model from scratch."""

import argparse
import logging
import os
from pprint import pformat

import torch

from cgcnn2.cli.common import (
    add_dataset_arguments,
    add_runtime_arguments,
    add_training_arguments,
    load_datasets,
    make_loader,
    make_scheduler,
    parse_arguments,
)
from cgcnn2.model import CrystalGraphConvNet
from cgcnn2.utils import (
    cgcnn_test,
    input_feature_lengths,
    seed_everything,
    setup_logging,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CGCNN model from scratch.")
    add_dataset_arguments(parser)
    model = parser.add_argument_group("model")
    model.add_argument(
        "--atom-fea-len",
        default=64,
        type=int,
        help="Hidden features per atom. Default: 64",
    )
    model.add_argument(
        "--n-conv", default=3, type=int, help="Convolutional layers. Default: 3"
    )
    model.add_argument(
        "--h-fea-len",
        default=128,
        type=int,
        help="Hidden features after pooling. Default: 128",
    )
    model.add_argument(
        "--n-h", default=1, type=int, help="Hidden layers after pooling. Default: 1"
    )
    training = add_training_arguments(parser)
    training.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-2,
        type=float,
        help="Learning rate. Default: 1e-2",
    )
    add_runtime_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_arguments(build_parser(), argv)
    logging.info(f"Parsed arguments:\n{pformat(vars(args))}")
    seed_everything(args.random_seed)
    output_folder = f"output_{args.job_id}"
    os.makedirs(output_folder, exist_ok=True)

    train_dataset, valid_dataset, test_dataset = load_datasets(args)
    orig_atom_fea_len, nbr_fea_len = input_feature_lengths(train_dataset)
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_path = os.path.join(output_folder, "best_model.ckpt")
    train(
        model,
        make_loader(train_dataset, args, shuffle=True),
        make_loader(valid_dataset, args),
        optimizer,
        args.device,
        epochs=int(args.epoch),
        checkpoint_path=checkpoint_path,
        checkpoint_args=vars(args),
        scheduler=make_scheduler(optimizer, args),
        stop_patience=int(args.stop_patience) if args.stop_patience else None,
        bias_temperature=args.bias_temperature,
    )

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True)["state_dict"])
    cgcnn_test(
        model,
        make_loader(test_dataset, args),
        args.device,
        results_file=os.path.join(output_folder, "test_results.csv"),
        plot_file=os.path.join(output_folder, "parity_plot.png"),
        axis_limits=args.axis_limits,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )


if __name__ == "__main__":
    main()
