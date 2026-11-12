"""Fine-tune a trained CGCNN model on a new dataset."""

import argparse
import logging
import os
from pprint import pformat
from typing import cast

import torch
import torch.nn as nn

from cgcnn2.cli.common import (
    add_dataset_arguments,
    add_runtime_arguments,
    add_training_arguments,
    load_datasets,
    make_loader,
    make_scheduler,
    parse_arguments,
)
from cgcnn2.utils import (
    cgcnn_test,
    load_checkpoint,
    model_from_checkpoint,
    seed_everything,
    setup_logging,
    standardize_output,
    target_statistics,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a trained CGCNN model.")
    parser.add_argument(
        "-mp",
        "--model-path",
        required=True,
        help="Checkpoint of the model to fine-tune.",
    )
    add_dataset_arguments(parser)
    training = add_training_arguments(parser)
    training.add_argument(
        "-r",
        "--reset",
        action="store_true",
        help="Reinitialize the fully connected layers that are fine-tuned.",
    )
    training.add_argument(
        "-tlfc",
        "--train-last-fc",
        action="store_true",
        help="Fine-tune only the output layer instead of all fully connected layers.",
    )
    training.add_argument(
        "-lrfc",
        "--lr-fc",
        default=0.01,
        type=float,
        help="Learning rate of the fine-tuned fully connected layers. Default: 0.01",
    )
    training.add_argument(
        "-lrnfc",
        "--lr-non-fc",
        default=0.001,
        type=float,
        help="Learning rate of all other layers. Default: 0.001",
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
    checkpoint = load_checkpoint(args.model_path)
    model = model_from_checkpoint(checkpoint, train_dataset).to(args.device)
    target_stats = target_statistics(train_dataset)

    if args.train_last_fc:
        fc_layers = [model.fc_out]
        logging.info(f"Fine-tuning the output layer with lr {args.lr_fc}.")
    else:
        fc_layers = cast(list[nn.Linear], [model.conv_to_fc, *model.fcs, model.fc_out])
        logging.info(f"Fine-tuning all fully connected layers with lr {args.lr_fc}.")
    if args.reset:
        logging.info("Reinitializing the fine-tuned layers.")
        for layer in fc_layers:
            layer.reset_parameters()
    else:
        # the loaded output layer predicts in the original units; training runs on
        # standardized targets
        standardize_output(model, target_stats)
    fc_params = [param for layer in fc_layers for param in layer.parameters()]
    fc_param_ids = set(map(id, fc_params))
    other_params = [p for p in model.parameters() if id(p) not in fc_param_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": fc_params, "lr": args.lr_fc},
            {"params": other_params, "lr": args.lr_non_fc},
        ]
    )

    checkpoint_path = os.path.join(output_folder, "best_model.ckpt")
    train(
        model,
        make_loader(train_dataset, args, shuffle=True),
        make_loader(valid_dataset, args),
        optimizer,
        args.device,
        epochs=int(args.epoch),
        checkpoint_path=checkpoint_path,
        checkpoint_args=checkpoint["args"],
        target_stats=target_stats,
        scheduler=make_scheduler(optimizer, args),
        stop_patience=int(args.stop_patience) if args.stop_patience else None,
        bias_temperature=args.bias_temperature,
    )

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
