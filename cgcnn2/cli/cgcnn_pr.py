"""Predict with a trained CGCNN model and, when targets are available, evaluate it."""

import argparse
import logging
import os
from pprint import pformat

from cgcnn2.cli.common import add_runtime_arguments, make_loader, parse_arguments
from cgcnn2.data import CIFData
from cgcnn2.utils import (
    cgcnn_test,
    load_checkpoint,
    model_from_checkpoint,
    seed_everything,
    setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict with a trained CGCNN model and evaluate it when the dataset has targets."
    )
    parser.add_argument(
        "-mp", "--model-path", required=True, help="Checkpoint of the trained model."
    )
    parser.add_argument(
        "-as",
        "--full-set",
        required=True,
        help="Directory with the CIF files to predict. With an id_prop.csv the metrics and parity plot are produced as well.",
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

    dataset = CIFData(args.full_set)
    model = model_from_checkpoint(load_checkpoint(args.model_path), dataset).to(
        args.device
    )
    cgcnn_test(
        model,
        make_loader(dataset, args),
        args.device,
        results_file=os.path.join(output_folder, "results.csv"),
        plot_file=os.path.join(output_folder, "parity_plot.png"),
        axis_limits=args.axis_limits,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )


if __name__ == "__main__":
    main()
