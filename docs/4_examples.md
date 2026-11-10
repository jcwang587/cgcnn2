# Examples

## Programmatic Examples

A series of notebooks demonstrates how to load and use the pretrained models for property prediction and structural descriptor extraction.

1. [Property Prediction](https://github.com/jcwang587/cgcnn2/blob/HEAD/examples/p01_property_prediction.ipynb)
1. [Structural Descriptor](https://github.com/jcwang587/cgcnn2/blob/HEAD/examples/p02_structural_descriptor.ipynb)

## Command Line Examples

The following Slurm scripts are examples of how to run the training, prediction, and fine-tuning of the models.

1. [Train a CGCNN from scratch](https://github.com/jcwang587/cgcnn2/blob/HEAD/examples/s01_slurm_train.sh)
1. [Make predictions using a pretrained model](https://github.com/jcwang587/cgcnn2/blob/HEAD/examples/s02_slurm_predict.sh)
1. [Fine-tune a pretrained model](https://github.com/jcwang587/cgcnn2/blob/HEAD/examples/s03_slurm_fine_tuning.sh)

## Retraining on Matbench

The bundled models were trained on a 2017 Materials Project snapshot that can no longer be downloaded. [Matbench](https://matbench.materialsproject.org/) provides fixed, versioned datasets for the same properties, fetched through [matminer](https://hackingmaterials.lbl.gov/matminer/), which is not a dependency of `cgcnn2`. The four tasks below correspond to the bundled `formation-energy-per-atom`, `band-gap`, `bulk-moduli` and `shear-moduli` models; the last column is the 5-fold mean of the CGCNN entry on the [Matbench leaderboard](https://matbench.materialsproject.org/).

| Task                 | Target               | Unit       | Structures | Download | Leaderboard CGCNN |
| -------------------- | -------------------- | ---------- | ---------- | -------- | ----------------- |
| `matbench_mp_e_form` | formation energy     | eV/atom    | 132,752    | 167 MB   | MAE 0.0337        |
| `matbench_mp_gap`    | PBE band gap         | eV         | 106,113    | 137 MB   | MAE 0.2972        |
| `matbench_log_kvrh`  | bulk modulus, log10  | log10(GPa) | 10,987     | 4 MB     | MAE 0.0712        |
| `matbench_log_gvrh`  | shear modulus, log10 | log10(GPa) | 10,987     | 4 MB     | MAE 0.0895        |

[`examples/matbench_dataset.py`](https://github.com/jcwang587/cgcnn2/blob/HEAD/examples/matbench_dataset.py) writes a task as `train/`, `valid/` and `test/` directories in the `CIFData` layout (`<id>.cif` files, `id_prop.csv` and `atom_init.json`). The test set is the Matbench fold-0 test split, `KFold(n_splits=5, shuffle=True, random_state=18012019)`, so a test error corresponds to the fold-0 score of a leaderboard entry. The table lists 5-fold means, and a single fold scatters around its mean. The validation set is a random 10% of the fold-0 training split. IDs follow the Matbench convention, e.g. `mb-log-kvrh-00006`.

```bash
uv run --with matminer examples/matbench_dataset.py matbench_log_kvrh data/matbench_log_kvrh
```

(or `pip install matminer` and run the script with `python`). matminer caches the downloaded file in its package directory unless `MATMINER_DATA` points elsewhere; writing the 132,752 CIF files of `matbench_mp_e_form` takes about two minutes. Training and evaluation then use the regular commands: `cgcnn-tr` evaluates the best checkpoint on the test set when training ends, and `cgcnn-pr` re-evaluates any checkpoint on it. Both write the per-structure predictions as CSV and a parity plot annotated with the MAE, the leaderboard metric, to `output_<job-id>/`. A 30-epoch smoke test:

```bash
cgcnn-tr --train-set data/matbench_log_kvrh/train --valid-set data/matbench_log_kvrh/valid --test-set data/matbench_log_kvrh/test --epoch 30 --job-id kvrh
cgcnn-pr --model-path output_kvrh/best_model.ckpt --full-set data/matbench_log_kvrh/test --job-id kvrh_test
```

### Full runs on a single GPU

Blackwell GPUs such as the RTX 5080 need a PyTorch build with CUDA 12.8 or newer. The default Linux wheels on PyPI qualify; on Windows the PyPI wheel is CPU-only, so install PyTorch from the CUDA index first (`pip install torch --index-url https://download.pytorch.org/whl/cu128`). The model has about 100k parameters, so 16 GB of GPU memory is far more than `--batch-size 256` needs; the run is bound by CIF parsing and Python data loading instead, and `--workers` decides where that cost goes:

- `--workers 0` (default): the dataset's LRU cache keeps every parsed crystal graph in RAM, so each CIF is parsed once, during the first epoch. The cache costs about 3 KB per atom: 0.3 GB for `matbench_log_kvrh` and about 11 GB for `matbench_mp_e_form` (3.9 million atoms).
- `--workers N`: the `DataLoader` re-creates its worker processes every epoch with empty caches, so the whole dataset is re-parsed every epoch (in parallel). Only worth it when RAM is short.

```bash
for task in matbench_mp_e_form matbench_mp_gap matbench_log_kvrh matbench_log_gvrh; do
    uv run --with matminer examples/matbench_dataset.py $task data/$task
    cgcnn-tr --train-set data/$task/train --valid-set data/$task/valid --test-set data/$task/test \
        --epoch 1e3 --stop-patience 1e2 --learning-rate 1e-2 --lr-patience 20 \
        --batch-size 256 --workers 0 --job-id $task --random-seed 42
done
```

The test MAE of each run is printed on `output_<task>/parity_plot.png`, with the predictions in `output_<task>/test_results.csv`.
