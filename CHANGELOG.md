<a id="v1.0.0"></a>
## [v1.0.0](https://github.com/jcwang587/cgcnn2/releases/tag/v1.0.0) - 2026-11-12

cgcnn2 1.0.0 reduces the package to one uniform, current implementation of CGCNN and ships retrained models with stated accuracy. The `cgcnn2` package goes from 2,978 to 1,644 lines while graph construction, pooling and the convolution all get faster, and training reaches the accuracy of the original implementation.

## Highlights
- **New bundled models with stated accuracy.** The four bundled models are retrained with cgcnn2 on Matbench, and each ships with its test MAE on fold 0: formation energy 0.0323 eV/atom, band gap 0.2520 eV, bulk modulus 0.0648 and shear modulus 0.0821 log10(GPa). The leaderboard's CGCNN has 0.0340, 0.2978, 0.0702 and 0.0870 on the same fold. They predict in real units; the 0.6.0 models returned standardized values, for example −0.39 instead of about −2.1 eV/atom for the formation energy of NaCl ([#165](https://github.com/jcwang587/cgcnn2/issues/165), [#178](https://github.com/jcwang587/cgcnn2/issues/178)).
- **Training matches the original CGCNN.** Targets are standardized during training and the checkpoint with the lowest validation MAE is kept; checkpoints still predict in the original units. On Matbench fold 0, over five seeds, the test MAE goes from 0.0912 ± 0.0042 to 0.0850 ± 0.0007 on `log_gvrh` and from 0.0815 ± 0.0070 to 0.0665 ± 0.0010 on `log_kvrh` ([#177](https://github.com/jcwang587/cgcnn2/issues/177)).
- **Descriptors that carry information.** `cgcnn_pred` returns the crystal feature vector of the CGCNN paper: the atom features after the convolutions, averaged over each crystal (64 values by default). The hidden features that 0.6.0 returned are mostly inactive in models trained on large datasets, 12 of 128 in the new formation-energy model. Used to predict a property the model was not trained on, the new descriptors do much better: R² goes from 0.07 to 0.53 for the formation-energy model's descriptors and the bulk modulus ([#179](https://github.com/jcwang587/cgcnn2/issues/179)).
- **Faster crystal graphs.** Graph construction runs on pymatgen's C-backed `Structure.get_neighbor_list` with vectorized NumPy instead of per-atom Python loops; the tensors are bit-identical to 0.6.0 ([#163](https://github.com/jcwang587/cgcnn2/issues/163)).
- **Faster model, same parameters.** Pooling is one `scatter_reduce_` over a flat crystal index ([#164](https://github.com/jcwang587/cgcnn2/issues/164)). The convolution applies the weight blocks of `fc_full` to the atom, neighbor and bond features separately instead of building the per-neighbor concatenation: about 30 % less time in its forward and backward pass on CPU, and CPU training is run-to-run deterministic ([#173](https://github.com/jcwang587/cgcnn2/issues/173)). Checkpoints are unchanged.
- **One code path for everything.** One dataset class (`CIFData`; `id_prop.csv` is optional), one training loop shared by `cgcnn-tr` and `cgcnn-ft`, one set of CLI arguments, and `load_checkpoint` / `model_from_checkpoint` for loading models ([#173](https://github.com/jcwang587/cgcnn2/issues/173)).
- **Safe checkpoints.** Checkpoints load with `torch.load(weights_only=True)` ([#165](https://github.com/jcwang587/cgcnn2/issues/165)).
- **Prediction.** `cgcnn_pred` batches inference and returns NumPy arrays; `cgcnn-pr` also works on a directory without `id_prop.csv` ([#166](https://github.com/jcwang587/cgcnn2/issues/166), [#173](https://github.com/jcwang587/cgcnn2/issues/173)).
- **Any torch device.** `--device cpu | cuda | cuda:1 | mps` on all three commands ([#173](https://github.com/jcwang587/cgcnn2/issues/173)).
- **Python 3.15.** Supported next to 3.11 to 3.14. pymatgen and PyYAML have no 3.15 wheels yet and are built from source there, which needs a C compiler ([#175](https://github.com/jcwang587/cgcnn2/issues/175), [#180](https://github.com/jcwang587/cgcnn2/issues/180)).
- **Retraining recipe.** `examples/matbench_dataset.py` exports a Matbench task in the `CIFData` layout with the published Matbench v0.1 fold, and the docs give the training commands that produced the bundled models ([#171](https://github.com/jcwang587/cgcnn2/issues/171), [#178](https://github.com/jcwang587/cgcnn2/issues/178)).
- **Tests.** A pytest suite covers the dataset, batching, the model, training, the bundled models and the three commands end to end ([#173](https://github.com/jcwang587/cgcnn2/issues/173), [#177](https://github.com/jcwang587/cgcnn2/issues/177), [#178](https://github.com/jcwang587/cgcnn2/issues/178), [#179](https://github.com/jcwang587/cgcnn2/issues/179), [#180](https://github.com/jcwang587/cgcnn2/issues/180)).

## Upgrading from 0.6.0
- The bundled `formation-energy-per-atom`, `band-gap`, `bulk-moduli` and `shear-moduli` models are new, so their predictions differ from the 2017 checkpoints', and they are in the documented units instead of standardized values. If you converted 0.6.0 outputs yourself with the checkpoint's `normalizer` mean and std, remove that step.
- `efermi`, `final-energy-per-atom` and `poisson-ratio` are no longer bundled: Matbench has no data to retrain them with a stated test error.
- Checkpoints you trained with 0.6.0 load unchanged in `cgcnn-pr`, `cgcnn-ft` and `cgcnn_pred`.
- Training picks the best checkpoint by validation MAE instead of MSE and logs `Valid MAE`; new checkpoints store `best_mae_error` instead of `best_mse_error`. Targets are standardized internally, so a learning rate tuned for raw targets may need another look, and training stops with an error when the training targets are missing or all equal ([#180](https://github.com/jcwang587/cgcnn2/issues/180)).
- `cgcnn_pred(model_path, full_set, device="cpu", batch_size=256, num_workers=0)` returns `(predictions, descriptors)` as arrays of shape `(n,)` and `(n, atom_fea_len)`; `verbose` and `cuda` are removed. The descriptors are the pooled atom features, no longer the hidden features before the output layer, and the second output of `CrystalGraphConvNet.forward` changes in the same way ([#179](https://github.com/jcwang587/cgcnn2/issues/179)).
- CLI: `--disable-cuda` becomes `--device cpu`; `--axis-limits` takes `MIN MAX`; `--task` is removed; `cgcnn-pr` no longer takes `--random-seed`, which only ever affected training.
- `CIFData_NoTarget` becomes `CIFData` and `AtomCustomJSONInitializer` becomes `AtomInitializer`; the `random_seed` argument of the dataset is removed.
- `collate_pool` returns `crystal_atom_idx` as one `(N,)` LongTensor and `CrystalGraphConvNet.forward` takes it in that form. Code that uses `collate_pool` with a `DataLoader` is unaffected.
- `cgcnn_test` takes `xlabel` and `ylabel` as keyword arguments.
- Removed: `Normalizer`; the classification path (`CrystalGraphConvNet(classification=...)`, the semi-metal checkpoint and the classification sample data), which could not be trained or used for prediction ([#170](https://github.com/jcwang587/cgcnn2/issues/170), [#173](https://github.com/jcwang587/cgcnn2/issues/173)); `cgcnn_descriptor`, `print_checkpoint_info`, `get_lr`, `plot_scatter`, `plot_convergence`, `unique_structures_clean` and `output_id_gen`.
- `torch>=2.10`: the first release of 2026, following the same rule as pymatgen, and the release that fixes CVE-2026-24747 in the `weights_only` unpickler.

## Fixes
- `metrics_text` scales the MSE by `unit_scale**2`; with a unit conversion it was off by a factor of `unit_scale` ([#174](https://github.com/jcwang587/cgcnn2/issues/174)).

## Housekeeping
- `cgcnn2.pretrained_models` exports a path constant for each of the four bundled models; the files shrink from 0.79 MB to 0.34 MB each ([#178](https://github.com/jcwang587/cgcnn2/issues/178)).
- An MIT `LICENSE` file with the copyright notice of the original CGCNN ships in the sdist and the wheel, and the metadata uses the SPDX license expression.
- The README is cut down to what the package does, how to install it and where the docs are.
- The docs draw the NaCl structure with [bravais](https://github.com/jcwang587/bravais) instead of a 1.6 MB GLB model ([#176](https://github.com/jcwang587/cgcnn2/issues/176)).
- The `dev/` profiling scripts are removed, and the refreshed `uv.lock` resolves the open Dependabot alerts (certifi, filelock, virtualenv).
- uv-build `<0.14` ([#169](https://github.com/jcwang587/cgcnn2/issues/169)), pre-commit hooks updated ([#168](https://github.com/jcwang587/cgcnn2/issues/168)); the ty hook follows the version in `uv.lock`, so every hook passes locally ([#174](https://github.com/jcwang587/cgcnn2/issues/174)).

## Verification
On `dev` at dbac3f9, macOS arm64, torch 2.15.0, CPU:
- `pytest`: 15 passed on Python 3.14.7 and on 3.15.0rc2. `pre-commit run --all-files` passes with no hook skipped.
- Each bundled model reproduces the predictions of its training run on the full Matbench test split (2,198 to 26,551 structures) to 4e-7, with the test MAEs listed above.
- The descriptor change ([#179](https://github.com/jcwang587/cgcnn2/issues/179)) leaves the predictions of the bundled models and seeded training logs identical, and each of the 64 descriptor values varies across structures in all four models.
- Two identical `cgcnn-tr` runs give identical logs.
- `cgcnn-tr` on Matbench `log_gvrh` and `log_kvrh`, fold 0, seeds 0 to 4: test MAE 0.0850 ± 0.0007 and 0.0665 ± 0.0010, measured for [#177](https://github.com/jcwang587/cgcnn2/issues/177).
- A checkpoint trained with the `v0.6.0` tag loads in 1.0.0 `cgcnn-pr` and `cgcnn-ft`; its predictions agree between the two versions to 4e-6.
- `uv build` produces the 1.0.0 sdist and wheel with the four models and the license. The wheel installed into a clean Python 3.13 environment predicts −2.060 eV/atom for the first sample structure with the bundled formation-energy model and returns descriptors of shape (10, 64).
- `mkdocs build` succeeds.

**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.6.0...v1.0.0


[Changes][v1.0.0]


<a id="v0.6.0"></a>
## [v0.6.0](https://github.com/jcwang587/cgcnn2/releases/tag/v0.6.0) - 2026-07-07

## What's Changed
* Improved checkpoint loading compatibility by [@jcwang587](https://github.com/jcwang587) in [#154](https://github.com/jcwang587/cgcnn2/pull/154)
* Update .lycheeignore and README.md for clarity and accuracy by [@jcwang587](https://github.com/jcwang587) in [#158](https://github.com/jcwang587/cgcnn2/pull/158)
* Update GitHub Actions workflow to use uv version 8.3.0 by [@jcwang587](https://github.com/jcwang587) in [#159](https://github.com/jcwang587/cgcnn2/pull/159)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.5.3...v0.6.0

[Changes][v0.6.0]


<a id="v0.5.3"></a>
## [v0.5.3](https://github.com/jcwang587/cgcnn2/releases/tag/v0.5.3) - 2025-12-15

## What's Changed
* simplify examples by [@jcwang587](https://github.com/jcwang587) in [#125](https://github.com/jcwang587/cgcnn2/pull/125)
* Add a local `ty` pre-commit hook to run static checks by [@jcwang587](https://github.com/jcwang587) in [#126](https://github.com/jcwang587/cgcnn2/pull/126)
* Add `ruff`’s `isort` rules by [@jcwang587](https://github.com/jcwang587) in [#127](https://github.com/jcwang587/cgcnn2/pull/127)
* Update example links in the Examples documentation by [@jcwang587](https://github.com/jcwang587) in [#128](https://github.com/jcwang587/cgcnn2/pull/128)
* Replace manual cache with pip cache by [@jcwang587](https://github.com/jcwang587) in [#130](https://github.com/jcwang587/cgcnn2/pull/130)
* Update DOI links in README and documentation by [@jcwang587](https://github.com/jcwang587) in [#142](https://github.com/jcwang587/cgcnn2/pull/142)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.5.2...v0.5.3

[Changes][v0.5.3]


<a id="v0.5.2"></a>
## [v0.5.2](https://github.com/jcwang587/cgcnn2/releases/tag/v0.5.2) - 2025-08-13

* Reset verbosity level by [@jcwang587](https://github.com/jcwang587) in [#110](https://github.com/jcwang587/cgcnn2/pull/110)
* Add a pre-commit hook for `pyproject` formatting by [@jcwang587](https://github.com/jcwang587) in [#111](https://github.com/jcwang587/cgcnn2/pull/111)
* Add a pre-commit hook for `yaml` formatting by [@jcwang587](https://github.com/jcwang587) in [#112](https://github.com/jcwang587/cgcnn2/pull/112)
* Correct documentation math expressions formatting by [@jcwang587](https://github.com/jcwang587) in [#113](https://github.com/jcwang587/cgcnn2/pull/113)
* Add docstrings by [@coderabbitai](https://github.com/coderabbitai)[bot] in [#115](https://github.com/jcwang587/cgcnn2/pull/115)
* Simplify `super()` calls by [@jcwang587](https://github.com/jcwang587) in [#117](https://github.com/jcwang587/cgcnn2/pull/117)
* Add support returning plot axes by [@jcwang587](https://github.com/jcwang587) in [#119](https://github.com/jcwang587/cgcnn2/pull/119)
* Correct import statements in example notebooks and documentation by [@tlp-tau](https://github.com/tlp-tau) in [#121](https://github.com/jcwang587/cgcnn2/pull/121)

### New Contributors
* [@coderabbitai](https://github.com/coderabbitai)[bot] made their first contribution in [#115](https://github.com/jcwang587/cgcnn2/pull/115)
* [@tlp-tau](https://github.com/tlp-tau) made their first contribution in [#121](https://github.com/jcwang587/cgcnn2/pull/121)

**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.5.1...v0.5.2

[Changes][v0.5.2]


<a id="v0.5.1"></a>
## [v0.5.1](https://github.com/jcwang587/cgcnn2/releases/tag/v0.5.1) - 2025-07-25

* Add function for generating convergence plots by [@jcwang587](https://github.com/jcwang587) in [#104](https://github.com/jcwang587/cgcnn2/pull/104)
* Apply `np.mean()` for MAE, MSE, and RMSE computations by [@jcwang587](https://github.com/jcwang587) in [#106](https://github.com/jcwang587/cgcnn2/pull/106)
* Add an optional parameter to scale metric values by [@jcwang587](https://github.com/jcwang587) in [#107](https://github.com/jcwang587/cgcnn2/pull/107)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.5.0...v0.5.1

[Changes][v0.5.1]


<a id="v0.5.0"></a>
## [v0.5.0](https://github.com/jcwang587/cgcnn2/releases/tag/v0.5.0) - 2025-07-17

* pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#92](https://github.com/jcwang587/cgcnn2/pull/92)
* Add new scatter plot visualization by [@jcwang587](https://github.com/jcwang587) in [#93](https://github.com/jcwang587/cgcnn2/pull/93)
* Plotting functions return the plot objects by [@jcwang587](https://github.com/jcwang587) in [#98](https://github.com/jcwang587/cgcnn2/pull/98)
* Improve variable naming for axis limits by [@jcwang587](https://github.com/jcwang587) in [#99](https://github.com/jcwang587/cgcnn2/pull/99)

### New Contributors
* [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] made their first contribution in [#92](https://github.com/jcwang587/cgcnn2/pull/92)

**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.8...v0.5.0

[Changes][v0.5.0]


<a id="v0.4.8"></a>
## [v0.4.8](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.8) - 2025-07-07

* Update GitHub Actions workflow to install the latest version of `uv` by [@jcwang587](https://github.com/jcwang587) in [#86](https://github.com/jcwang587/cgcnn2/pull/86)
* Customizable units and improved formatting for hexbin plot by [@jcwang587](https://github.com/jcwang587) in [#87](https://github.com/jcwang587/cgcnn2/pull/87)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.7...v0.4.8

[Changes][v0.4.8]


<a id="v0.4.7"></a>
## [v0.4.7](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.7) - 2025-06-29

* Enhance the pooling operation to use stacking instead of concatenation by [@jcwang587](https://github.com/jcwang587) in [#78](https://github.com/jcwang587/cgcnn2/pull/78)
* Add `seed_everything` function by [@jcwang587](https://github.com/jcwang587) in [#79](https://github.com/jcwang587/cgcnn2/pull/79)
* Generate a changelog by [@jcwang587](https://github.com/jcwang587) in [#80](https://github.com/jcwang587/cgcnn2/pull/80)
* Update the `model-viewer` library by [@jcwang587](https://github.com/jcwang587) in [#81](https://github.com/jcwang587/cgcnn2/pull/81)
* Add hexbin plot function and remove `pymatviz` from the required dependencies by [@jcwang587](https://github.com/jcwang587) in [#85](https://github.com/jcwang587/cgcnn2/pull/85)

**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.6...v0.4.7

[Changes][v0.4.7]


<a id="v0.4.6"></a>
## [v0.4.6](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.6) - 2025-06-12

* Fix an error when generating the parity plot for a specific range by [@jcwang587](https://github.com/jcwang587) in [#65](https://github.com/jcwang587/cgcnn2/pull/65)
* Improve the `CIFData` caching mechanism by [@jcwang587](https://github.com/jcwang587) in [#76](https://github.com/jcwang587/cgcnn2/pull/76)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.5...v0.4.6

[Changes][v0.4.6]


<a id="v0.4.5"></a>
## [v0.4.5](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.5) - 2025-06-08

* Add a new section for training options in the documentation by [@jcwang587](https://github.com/jcwang587) in [#46](https://github.com/jcwang587/cgcnn2/pull/46)
* Info displayed with `logging` by [@jcwang587](https://github.com/jcwang587) in [#47](https://github.com/jcwang587/cgcnn2/pull/47)
* Add flexible size of caching for DataLoader by [@jcwang587](https://github.com/jcwang587) in [#52](https://github.com/jcwang587/cgcnn2/pull/52)
* Simplify dataset splitting process for training, validation, and testing by [@jcwang587](https://github.com/jcwang587) in [#54](https://github.com/jcwang587/cgcnn2/pull/54)
* Enhance neighbor list loading with Cythonized `get_neighbor_list` by [@jcwang587](https://github.com/jcwang587) in [#60](https://github.com/jcwang587/cgcnn2/pull/60)
* Change the output format of parity plots from SVG to PNG by [@jcwang587](https://github.com/jcwang587) in [#63](https://github.com/jcwang587/cgcnn2/pull/63)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.4...v0.4.5

[Changes][v0.4.5]


<a id="v0.4.4"></a>
## [v0.4.4](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.4) - 2025-05-29

* Enhance model inference efficiency by adopting `torch.inference_mode` by [@jcwang587](https://github.com/jcwang587) in [#38](https://github.com/jcwang587/cgcnn2/pull/38)
* Add option to force training set inclusion while not preserving the train:valid:test ratio by [@jcwang587](https://github.com/jcwang587) in [#39](https://github.com/jcwang587/cgcnn2/pull/39)
* Add `print_checkpoint_info` by [@jcwang587](https://github.com/jcwang587) in [#40](https://github.com/jcwang587/cgcnn2/pull/40)
* Set `shuffle=False` for validation and test set by [@jcwang587](https://github.com/jcwang587) in [#42](https://github.com/jcwang587/cgcnn2/pull/42)
* Remove `scikit-learn` dependency by [@jcwang587](https://github.com/jcwang587) in [#44](https://github.com/jcwang587/cgcnn2/pull/44)
* Switch the build system from `poetry` to `uv` for package building and publishing by [@jcwang587](https://github.com/jcwang587) in [#45](https://github.com/jcwang587/cgcnn2/pull/45)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.3...v0.4.4

[Changes][v0.4.4]


<a id="v0.4.3"></a>
## [v0.4.3](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.3) - 2025-05-22

* Fix an error when inputting dataset without a split ratio by [@jcwang587](https://github.com/jcwang587) in [#36](https://github.com/jcwang587/cgcnn2/pull/36)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.2...v0.4.3

[Changes][v0.4.3]


<a id="v0.4.2"></a>
## [v0.4.2](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.2) - 2025-05-16

* Allow custom x‑ and y‑axis labels in parity plot by [@jcwang587](https://github.com/jcwang587) in [#33](https://github.com/jcwang587/cgcnn2/pull/33)
* Sort output indices by structure ID when running predictions by [@jcwang587](https://github.com/jcwang587) in [#34](https://github.com/jcwang587/cgcnn2/pull/34)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.1...v0.4.2

[Changes][v0.4.2]


<a id="v0.4.1"></a>
## [v0.4.1](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.1) - 2025-04-29

* Add CLI scripts `id_gen` and `atom_gen` by [@jcwang587](https://github.com/jcwang587) in [#23](https://github.com/jcwang587/cgcnn2/pull/23)
* Add documentation by [@jcwang587](https://github.com/jcwang587) in [#24](https://github.com/jcwang587/cgcnn2/pull/24)
* Remove the requirement for `id_prop.csv` when running predictions on unknown datasets by [@jcwang587](https://github.com/jcwang587) in [#29](https://github.com/jcwang587/cgcnn2/pull/29)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.0...v0.4.1

[Changes][v0.4.1]


<a id="v0.4.0"></a>
## [v0.4.0](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.0) - 2025-04-09

* Add a function to deduplicate CIF files for cleaning the dataset by [@jcwang587](https://github.com/jcwang587) in [#8](https://github.com/jcwang587/cgcnn2/pull/8)
* Set up pre-trained models in subpackage by [@jcwang587](https://github.com/jcwang587) in [#10](https://github.com/jcwang587/cgcnn2/pull/10)
* Add some missing type hints to `cgcnn_model.py` by [@Andrew-S-Rosen](https://github.com/Andrew-S-Rosen) in [#14](https://github.com/jcwang587/cgcnn2/pull/14)
* Fix return value in `p01_prediction.ipynb` by [@Andrew-S-Rosen](https://github.com/Andrew-S-Rosen) in [#11](https://github.com/jcwang587/cgcnn2/pull/11)
* Sort imports and remove unused imports by [@Andrew-S-Rosen](https://github.com/Andrew-S-Rosen) in [#12](https://github.com/jcwang587/cgcnn2/pull/12)
* Use `dict` instead of `Dict` for type hinting by [@Andrew-S-Rosen](https://github.com/Andrew-S-Rosen) in [#13](https://github.com/jcwang587/cgcnn2/pull/13)
* Fix docstring for `ConvLayer` by [@Andrew-S-Rosen](https://github.com/Andrew-S-Rosen) in [#17](https://github.com/jcwang587/cgcnn2/pull/17)
* Add a `cgcnn-pr` cli script for prediction by [@jcwang587](https://github.com/jcwang587) in [#20](https://github.com/jcwang587/cgcnn2/pull/20)
* Add a `cgcnn-tr` cli script for training from scratch by [@jcwang587](https://github.com/jcwang587) in [#22](https://github.com/jcwang587/cgcnn2/pull/22)

### New Contributors
* [@Andrew-S-Rosen](https://github.com/Andrew-S-Rosen) made their first contribution in [#14](https://github.com/jcwang587/cgcnn2/pull/14)

**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.3.4...v0.4.0

[Changes][v0.4.0]


<a id="v0.3.4"></a>
## [v0.3.4](https://github.com/jcwang587/cgcnn2/releases/tag/v0.3.4) - 2025-03-26

* Add option to force training set inclusion while preserving the train:valid:test ratio by [@jcwang587](https://github.com/jcwang587) in [#7](https://github.com/jcwang587/cgcnn2/pull/7)

**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.3.2...v0.3.4

[Changes][v0.3.4]


<a id="v0.3.2"></a>
## [v0.3.2](https://github.com/jcwang587/cgcnn2/releases/tag/v0.3.2) - 2025-03-19

* Add an option to the fine-tuning script to generate a parity plot within a user-specified range

### New Contributors
* [@jcwang587](https://github.com/jcwang587) made their first contribution

**Full Changelog**: https://github.com/jcwang587/cgcnn2/commits/v0.3.2

[Changes][v0.3.2]


[v1.0.0]: https://github.com/jcwang587/cgcnn2/compare/v0.6.0...v1.0.0
[v0.6.0]: https://github.com/jcwang587/cgcnn2/compare/v0.5.3...v0.6.0
[v0.5.3]: https://github.com/jcwang587/cgcnn2/compare/v0.5.2...v0.5.3
[v0.5.2]: https://github.com/jcwang587/cgcnn2/compare/v0.5.1...v0.5.2
[v0.5.1]: https://github.com/jcwang587/cgcnn2/compare/v0.5.0...v0.5.1
[v0.5.0]: https://github.com/jcwang587/cgcnn2/compare/v0.4.8...v0.5.0
[v0.4.8]: https://github.com/jcwang587/cgcnn2/compare/v0.4.7...v0.4.8
[v0.4.7]: https://github.com/jcwang587/cgcnn2/compare/v0.4.6...v0.4.7
[v0.4.6]: https://github.com/jcwang587/cgcnn2/compare/v0.4.5...v0.4.6
[v0.4.5]: https://github.com/jcwang587/cgcnn2/compare/v0.4.4...v0.4.5
[v0.4.4]: https://github.com/jcwang587/cgcnn2/compare/v0.4.3...v0.4.4
[v0.4.3]: https://github.com/jcwang587/cgcnn2/compare/v0.4.2...v0.4.3
[v0.4.2]: https://github.com/jcwang587/cgcnn2/compare/v0.4.1...v0.4.2
[v0.4.1]: https://github.com/jcwang587/cgcnn2/compare/v0.4.0...v0.4.1
[v0.4.0]: https://github.com/jcwang587/cgcnn2/compare/v0.3.4...v0.4.0
[v0.3.4]: https://github.com/jcwang587/cgcnn2/compare/v0.3.2...v0.3.4
[v0.3.2]: https://github.com/jcwang587/cgcnn2/tree/v0.3.2

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.1 -->
