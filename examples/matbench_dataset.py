"""Export a Matbench task as cgcnn2 train/valid/test directories.

    uv run --with matminer examples/matbench_dataset.py matbench_log_kvrh data/matbench_log_kvrh

Writes <out>/train, <out>/valid and <out>/test in the layout `cgcnn2.data.CIFData`
expects: one `<id>.cif` per structure, `id_prop.csv` and `atom_init.json`. The test
set is the Matbench v0.1 fold-0 test split (`KFold(5, shuffle=True,
random_state=18012019)`, stratified for classification tasks), so test errors are
comparable with the Matbench leaderboard. The validation set is a random tenth of
the fold-0 training split. IDs follow the Matbench convention, e.g.
`mb-log-kvrh-00006`.
"""

import argparse
from importlib import resources
from pathlib import Path

from matminer.datasets import load_dataset
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def main(task: str, out: str) -> None:
    df = load_dataset(task, pbar=True)
    (target,) = df.columns.drop("structure")
    y = df[target]
    folds = StratifiedKFold if y.dtype == bool else KFold
    train_idx, test_idx = next(
        folds(n_splits=5, shuffle=True, random_state=18012019).split(df, y)
    )
    train_idx, valid_idx = train_test_split(train_idx, test_size=0.1, random_state=0)

    prefix = task.replace("matbench", "mb").replace("_", "-")
    digits = len(str(len(df)))
    atom_init = resources.files("cgcnn2") / "asset" / "atom_init.json"
    splits = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    for split, idx in splits.items():
        split_dir = Path(out, split)
        split_dir.mkdir(parents=True)
        (split_dir / "atom_init.json").write_bytes(atom_init.read_bytes())
        print(f"Writing {len(idx)} structures to {split_dir}")
        with open(split_dir / "id_prop.csv", "w") as id_prop:
            for i in sorted(idx):
                cif_id = f"{prefix}-{i + 1:0{digits}d}"
                df["structure"].iloc[i].to(filename=split_dir / f"{cif_id}.cif")
                id_prop.write(f"{cif_id},{float(y.iloc[i])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", help="Matbench task name, e.g. matbench_log_kvrh")
    parser.add_argument("out", help="Output directory for train/, valid/ and test/")
    args = parser.parse_args()
    main(args.task, args.out)
