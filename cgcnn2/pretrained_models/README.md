# Pre-trained CGCNN models

The bundled models are trained with cgcnn2 on [Matbench](https://matbench.materialsproject.org/) v0.1 datasets, which come from the Materials Project. Each model was trained on the fold-0 training split of its task and evaluated once on the fold-0 test split.

| File                                | Property             | Unit       | Matbench task        | Structures | Test MAE | Leaderboard CGCNN |
| ----------------------------------- | -------------------- | ---------- | -------------------- | ---------- | -------- | ----------------- |
| `formation-energy-per-atom.pth.tar` | Formation energy     | eV/atom    | `matbench_mp_e_form` | 132,752    | 0.0323   | 0.0340            |
| `band-gap.pth.tar`                  | PBE band gap         | eV         | `matbench_mp_gap`    | 106,113    | 0.2520   | 0.2978            |
| `bulk-moduli.pth.tar`               | Bulk modulus, log10  | log10(GPa) | `matbench_log_kvrh`  | 10,987     | 0.0648   | 0.0702            |
| `shear-moduli.pth.tar`              | Shear modulus, log10 | log10(GPa) | `matbench_log_gvrh`  | 10,987     | 0.0821   | 0.0870            |

The last column is the test MAE of the [CGCNN entry on the Matbench leaderboard](https://matbench.materialsproject.org/Full%20Benchmark%20Data/matbench_v0.1_cgcnnv2019/) on the same fold. That entry trained on 60 % of each dataset and these models on 72 %.

## Training

Of each dataset, 72 % is used for training, 8 % for validation and 20 % for testing. The models use the default architecture and were trained at commit [`ea5d804`](https://github.com/jcwang587/cgcnn2/commit/ea5d8041107ca10fc316b7e0608c4bc8a5b36ef0) with the commands of [Retraining on Matbench](https://jcwang.dev/cgcnn2/4_examples/#retraining-on-matbench), for example:

```bash
uv run --with matminer examples/matbench_dataset.py matbench_log_kvrh data/matbench_log_kvrh
cgcnn-tr --train-set data/matbench_log_kvrh/train --valid-set data/matbench_log_kvrh/valid --test-set data/matbench_log_kvrh/test --stop-patience 100 --lr-patience 20
```

## Before using the models

- A model only generalizes to crystals from the distribution it was trained on. The Materials Project mostly holds relaxed structures of known compounds, so expect larger errors for hypothetical or unstable crystals.
- The bulk and shear modulus models predict the base-10 logarithm of the modulus in GPa.
- Predictions are in the units listed above, and the checkpoints load with `torch.load(weights_only=True)`.

## How to cite

Please cite the model and the data:

- Tian Xie and Jeffrey C. Grossman, [Phys. Rev. Lett. 120, 145301 (2018)](https://doi.org/10.1103/PhysRevLett.120.145301), for CGCNN.
- Alexander Dunn *et al.*, [npj Comput. Mater. 6, 138 (2020)](https://doi.org/10.1038/s41524-020-00406-3), for Matbench.

## Sharing a model

To share a model of your own, open a pull request with the checkpoint, the property and its unit, and references for the data and the model. Only peer-reviewed work is accepted.
