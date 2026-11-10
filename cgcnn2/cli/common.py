"""Argument groups, dataset loading and DataLoader construction shared by the CLIs."""

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from cgcnn2.data import CIFData, collate_pool, full_set_split


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "-as",
        "--full-set",
        help="Directory with the CIF files of the whole dataset, split by the ratios below.",
    )
    group.add_argument(
        "-trs", "--train-set", help="Directory with the CIF files of the training set."
    )
    group.add_argument(
        "-vs", "--valid-set", help="Directory with the CIF files of the validation set."
    )
    group.add_argument(
        "-ts", "--test-set", help="Directory with the CIF files of the test set."
    )
    group.add_argument(
        "-trr",
        "--train-ratio",
        default=0.6,
        type=float,
        help="Fraction of the full set used for training. Default: 0.6",
    )
    group.add_argument(
        "-vr",
        "--valid-ratio",
        default=0.2,
        type=float,
        help="Fraction of the full set used for validation. Default: 0.2",
    )
    group.add_argument(
        "-tr",
        "--test-ratio",
        default=0.2,
        type=float,
        help="Fraction of the full set used for testing. Default: 0.2",
    )
    group.add_argument(
        "-trfs",
        "--train-force-set",
        help="Directory with CIF files that are all added to the training set when a full set is split.",
    )
    group.add_argument(
        "-cs",
        "--cache-size",
        type=int,
        help="Number of parsed training crystals kept in memory. Default: all; 0 disables the cache.",
    )


def add_training_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    group = parser.add_argument_group("training")
    group.add_argument(
        "-e",
        "--epoch",
        default=1000,
        type=float,
        help="Number of epochs. Default: 1000",
    )
    group.add_argument(
        "-sp",
        "--stop-patience",
        type=float,
        help="Epochs without validation improvement before training stops. Default: no early stopping",
    )
    group.add_argument(
        "-lrp",
        "--lr-patience",
        default=0,
        type=float,
        help="Epochs without validation improvement before the learning rate is reduced. Default: 0 (no scheduler)",
    )
    group.add_argument(
        "-lrf",
        "--lr-factor",
        default=0.5,
        type=float,
        help="Factor applied to the learning rate when it is reduced. Default: 0.5",
    )
    group.add_argument(
        "-bt",
        "--bias-temperature",
        default=-1.0,
        type=float,
        help="If > 0, weight each squared error by exp(-target / bias_temperature), favouring low-energy structures. Default: off",
    )
    group.add_argument(
        "-rs",
        "--random-seed",
        default=42,
        type=int,
        help="Seed of the weight initialization and the batch order; the split of --full-set does not depend on it. Default: 42",
    )
    return group


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("runtime and output")
    group.add_argument(
        "--device",
        help="Torch device, e.g. cpu, cuda, cuda:1 or mps. Default: cuda if available, else cpu",
    )
    group.add_argument(
        "-bs",
        "--batch-size",
        default=256,
        type=int,
        help="Crystals per batch. Default: 256",
    )
    group.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        help="DataLoader worker processes. Default: 0",
    )
    group.add_argument(
        "-ji",
        "--job-id",
        default=str(os.getpid()),
        help="Suffix of the output folder output_<job-id>. Default: the process ID",
    )
    group.add_argument(
        "-al",
        "--axis-limits",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Also save a parity plot clipped to this range of true values",
    )
    group.add_argument(
        "-x",
        "--xlabel",
        default="Actual",
        help="x-axis label of the parity plot. Default: Actual",
    )
    group.add_argument(
        "-y",
        "--ylabel",
        default="Predicted",
        help="y-axis label of the parity plot. Default: Predicted",
    )


def parse_arguments(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    """Parse `argv` (default `sys.argv[1:]`) and resolve the torch device."""
    args = parser.parse_args(argv)
    args.device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if (
        hasattr(args, "train_ratio")
        and abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-6
    ):
        logging.warning("Train ratio + Valid ratio + Test ratio != 1.0")
    return args


def load_datasets(args: argparse.Namespace) -> tuple[CIFData, CIFData, CIFData]:
    """Training, validation and test datasets from three directories or a split full set."""
    if args.full_set and not (args.train_set or args.valid_set or args.test_set):
        train_dir, valid_dir, test_dir = full_set_split(
            args.full_set, args.train_ratio, args.valid_ratio, args.train_force_set
        )
    elif args.train_set and args.valid_set and args.test_set and not args.full_set:
        train_dir, valid_dir, test_dir = args.train_set, args.valid_set, args.test_set
    else:
        parser_error = "Provide either --full-set or all of --train-set, --valid-set and --test-set."
        raise SystemExit(parser_error)
    return (
        CIFData(train_dir, cache_size=args.cache_size),
        CIFData(valid_dir),
        CIFData(test_dir),
    )


def make_loader(
    dataset: CIFData, args: argparse.Namespace, shuffle: bool = False
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        collate_fn=collate_pool,
        pin_memory=args.device.type == "cuda",
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """ReduceLROnPlateau when --lr-patience is positive, else None."""
    if args.lr_patience <= 0:
        return None
    patience = int(args.lr_patience)
    logging.info(
        f"[Scheduler] ReduceLROnPlateau: factor={args.lr_factor}, patience={patience}"
    )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=patience
    )
