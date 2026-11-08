"""One-time conversion of the bundled pre-trained CGCNN checkpoints.

The original checkpoints from https://github.com/txie-93/cgcnn were trained on
standardized targets and store the target mean/std under ``normalizer``. cgcnn2
never applied it, so predictions came out standardized. This script folds the
mean/std into the output layer (exact, since the layer is affine), drops the
optimizer state and the normalizer, and stores plain Python scalars so the files
load with ``torch.load(weights_only=True)``.

Run from the repository root: ``python dev/fold_pretrained.py``
"""

from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "cgcnn2" / "pretrained_models"

for path in sorted(MODEL_DIR.glob("*.pth.tar")):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "normalizer" not in checkpoint:
        print(f"{path.name}: already converted, skipped")
        continue
    mean = float(checkpoint["normalizer"]["mean"])
    std = float(checkpoint["normalizer"]["std"])
    state_dict = checkpoint["state_dict"]
    if checkpoint["args"]["task"] == "regression":
        state_dict["fc_out.weight"].mul_(std)
        state_dict["fc_out.bias"].mul_(std).add_(mean)
    else:
        assert (mean, std) == (0.0, 1.0), f"{path.name}: unexpected normalizer"
    torch.save(
        {
            "epoch": int(checkpoint["epoch"]),
            "best_mae_error": float(checkpoint["best_mae_error"]),
            "args": checkpoint["args"],
            "state_dict": state_dict,
        },
        path,
    )
    torch.load(path, map_location="cpu", weights_only=True)
    print(
        f"{path.name}: mean={mean:.4f} std={std:.4f} -> {path.stat().st_size / 1e6:.2f} MB"
    )
