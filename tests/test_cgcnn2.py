"""Fast end-to-end checks on the sample data (CPU only)."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from cgcnn2.cli import cgcnn_ft, cgcnn_pr, cgcnn_tr
from cgcnn2.data import CIFData, collate_pool
from cgcnn2.model import ConvLayer, CrystalGraphConvNet
from cgcnn2.pretrained_models import (
    BAND_GAP_MODEL,
    BULK_MODULI_MODEL,
    FORMATION_ENERGY_MODEL,
    SHEAR_MODULI_MODEL,
)
from cgcnn2.utils import (
    cgcnn_pred,
    load_checkpoint,
    metrics_text,
    model_from_checkpoint,
    predict,
    standardize_output,
    target_statistics,
    train,
)

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "data" / "sample-regression"


def test_dataset_items_and_cache():
    dataset = CIFData(SAMPLE)
    assert len(dataset) == 10
    (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id = dataset[0]
    n_atoms = atom_fea.shape[0]
    assert atom_fea.shape == (n_atoms, 92)
    assert nbr_fea.shape == (n_atoms, 12, 41)
    assert nbr_fea_idx.shape == (n_atoms, 12)
    assert nbr_fea_idx.max() < n_atoms
    assert target.shape == (1,) and cif_id == "1000041"
    assert dataset[0] is dataset[0]  # served from the LRU cache
    assert dataset.id_prop_data == CIFData(SAMPLE, cache_size=0).id_prop_data


def test_dataset_without_targets(tmp_path):
    for cif in sorted(SAMPLE.glob("*.cif"))[:3]:
        shutil.copy(cif, tmp_path)
    shutil.copy(SAMPLE / "atom_init.json", tmp_path)
    dataset = CIFData(tmp_path)
    assert [cif_id for cif_id, _ in dataset.id_prop_data] == [
        "1000041",
        "1000050",
        "1101051",
    ]
    assert all(np.isnan(target) for _, target in dataset.id_prop_data)


def test_collate_and_pooling():
    dataset = CIFData(SAMPLE)
    (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, cif_ids = collate_pool(
        [dataset[i] for i in range(3)]
    )
    sizes = [dataset[i][0][0].shape[0] for i in range(3)]
    assert atom_fea.shape[0] == nbr_fea.shape[0] == nbr_fea_idx.shape[0] == sum(sizes)
    assert crystal_atom_idx.tolist() == [
        i for i, n in enumerate(sizes) for _ in range(n)
    ]
    assert target.shape == (3, 1) and len(cif_ids) == 3

    model = CrystalGraphConvNet(92, 41)
    embedded = torch.randn(sum(sizes), 64)
    pooled = model.pooling(embedded, crystal_atom_idx)
    expected = torch.stack([embedded[crystal_atom_idx == i].mean(0) for i in range(3)])
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_conv_layer_matches_concatenated_linear():
    torch.manual_seed(0)
    n_atoms, n_nbr, atom_fea_len, nbr_fea_len = 50, 12, 16, 7
    layer = ConvLayer(atom_fea_len, nbr_fea_len).eval()
    atom_fea = torch.randn(n_atoms, atom_fea_len)
    nbr_fea = torch.randn(n_atoms, n_nbr, nbr_fea_len)
    nbr_fea_idx = torch.randint(0, n_atoms, (n_atoms, n_nbr))

    # The original formulation: one linear layer over [atom_i, atom_j, bond_ij].
    total = torch.cat(
        [atom_fea.unsqueeze(1).expand(-1, n_nbr, -1), atom_fea[nbr_fea_idx], nbr_fea],
        dim=2,
    )
    gated = layer.fc_full(total)
    gated = layer.bn1(gated.view(-1, 2 * atom_fea_len)).view(n_atoms, n_nbr, -1)
    nbr_filter, nbr_core = gated.chunk(2, dim=2)
    summed = layer.bn2(
        (torch.sigmoid(nbr_filter) * torch.nn.functional.softplus(nbr_core)).sum(1)
    )
    expected = torch.nn.functional.softplus(atom_fea + summed)
    with torch.no_grad():
        assert torch.allclose(
            layer(atom_fea, nbr_fea, nbr_fea_idx), expected, atol=1e-5
        )


@pytest.mark.parametrize(
    "model_path",
    [BAND_GAP_MODEL, BULK_MODULI_MODEL, FORMATION_ENERGY_MODEL, SHEAR_MODULI_MODEL],
)
def test_bundled_models_load_and_predict(model_path):
    checkpoint = load_checkpoint(model_path)
    predictions, descriptors = cgcnn_pred(model_path, SAMPLE)
    assert np.isfinite(predictions).all()
    assert descriptors.shape == (10, checkpoint["args"]["h_fea_len"])


def test_pretrained_prediction():
    predictions, descriptors = cgcnn_pred(FORMATION_ENERGY_MODEL, SAMPLE)
    assert predictions.shape == (10,) and descriptors.shape == (10, 128)
    assert predictions[0] == pytest.approx(-2.060, abs=1e-3)  # eV/atom
    batched, _ = cgcnn_pred(FORMATION_ENERGY_MODEL, SAMPLE, batch_size=3)
    assert np.allclose(predictions, batched, atol=1e-5)


def test_train_standardizes_targets_and_saves_original_units(tmp_path):
    torch.manual_seed(0)
    dataset = CIFData(SAMPLE)
    loader = DataLoader(dataset, batch_size=5, collate_fn=collate_pool)
    mean, std = target_statistics(dataset)
    hyperparameters = {"atom_fea_len": 8, "n_conv": 3, "h_fea_len": 8, "n_h": 1}
    model = CrystalGraphConvNet(92, 41, **hyperparameters)
    train(
        model,
        loader,
        loader,
        torch.optim.Adam(model.parameters(), lr=1e-2),
        "cpu",
        epochs=3,
        checkpoint_path=str(tmp_path / "best.ckpt"),
        checkpoint_args=hyperparameters,
        target_stats=(mean, std),
    )
    predictions, _, targets, _ = predict(model, loader, "cpu")
    # predictions are in the units of the targets, not standardized
    assert abs(predictions.mean() - mean) < std
    # the model ends with the weights of the checkpoint, selected by validation MAE
    checkpoint = load_checkpoint(tmp_path / "best.ckpt")
    reloaded = model_from_checkpoint(checkpoint, dataset)
    assert np.allclose(predictions, predict(reloaded, loader, "cpu")[0])
    assert checkpoint["best_mae_error"] == pytest.approx(
        np.abs(predictions - targets).mean(), rel=1e-4
    )


def test_standardize_output():
    torch.manual_seed(0)
    dataset = CIFData(SAMPLE)
    loader = DataLoader(dataset, batch_size=10, collate_fn=collate_pool)
    model = CrystalGraphConvNet(92, 41, atom_fea_len=8, h_fea_len=8)
    before = predict(model, loader, "cpu")[0]
    standardize_output(model, (1.5, 0.25))
    after = predict(model, loader, "cpu")[0]
    assert np.allclose(after, (before - 1.5) / 0.25, atol=1e-5)


def test_target_statistics_needs_varying_targets(tmp_path):
    for cif in sorted(SAMPLE.glob("*.cif"))[:3]:
        shutil.copy(cif, tmp_path)
    shutil.copy(SAMPLE / "atom_init.json", tmp_path)
    with pytest.raises(ValueError):  # no id_prop.csv, so the targets are missing
        target_statistics(CIFData(tmp_path))
    (tmp_path / "id_prop.csv").write_text("1000041,1.5\n1000050,1.5\n1101051,1.5\n")
    with pytest.raises(ValueError):  # all targets are equal
        target_statistics(CIFData(tmp_path))


def test_cli_train_finetune_predict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    common = ["--full-set", str(SAMPLE), "--device", "cpu", "--epoch", "2"]
    cgcnn_tr.main([*common, "--job-id", "tr"])
    checkpoint = torch.load("output_tr/best_model.ckpt", weights_only=True)
    assert set(checkpoint) == {"epoch", "state_dict", "best_mae_error", "args"}
    assert (tmp_path / "output_tr" / "test_results.csv").exists()
    assert (tmp_path / "output_tr" / "parity_plot.png").exists()

    cgcnn_ft.main(
        [
            *common,
            "--model-path",
            "output_tr/best_model.ckpt",
            "--reset",
            "--job-id",
            "ft",
        ]
    )
    assert (tmp_path / "output_ft" / "best_model.ckpt").exists()

    cgcnn_pr.main(
        [
            "--model-path",
            "output_ft/best_model.ckpt",
            "--full-set",
            str(SAMPLE),
            "--device",
            "cpu",
            "--job-id",
            "pr",
        ]
    )
    results = (tmp_path / "output_pr" / "results.csv").read_text().splitlines()
    assert results[0] == "cif_id,true,pred" and len(results) == 11


def test_metrics_text_scales_squared_units():
    df = pd.DataFrame({"true": [0.0, 0.0], "pred": [0.003, -0.003]})  # eV
    text = metrics_text(df, metrics=("mae", "mse", "rmse"), unit="meV", unit_scale=1000)
    assert "MAE}: 3.000" in text and "RMSE}: 3.000" in text and "MSE}: 9.000" in text
