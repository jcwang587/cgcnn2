<a id="v0.4.7"></a>
## [v0.4.7](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.7) - 2025-06-29

### What's Changed
* Enhanced the pooling operation to use stacking instead of concatenation by [@jcwang587](https://github.com/jcwang587) in [#78](https://github.com/jcwang587/cgcnn2/pull/78)
* Added `seed_everything` function by [@jcwang587](https://github.com/jcwang587) in [#79](https://github.com/jcwang587/cgcnn2/pull/79)
* Generated a changelog by [@jcwang587](https://github.com/jcwang587) in [#80](https://github.com/jcwang587/cgcnn2/pull/80)
* Updated the `model-viewer` JavaScript library by [@jcwang587](https://github.com/jcwang587) in [#81](https://github.com/jcwang587/cgcnn2/pull/81)
* Added scatter plot function and removed `pymatviz` from the required dependencies by [@jcwang587](https://github.com/jcwang587) in [#85](https://github.com/jcwang587/cgcnn2/pull/85)

**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.6...v0.4.7

[Changes][v0.4.7]


<a id="v0.4.6"></a>
## [v0.4.6](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.6) - 2025-06-12

### What's Changed
* Fixed an error when generating the parity plot for a specific range by [@jcwang587](https://github.com/jcwang587) in [#65](https://github.com/jcwang587/cgcnn2/pull/65)
* Improved the `CIFData` caching mechanism by [@jcwang587](https://github.com/jcwang587) in [#76](https://github.com/jcwang587/cgcnn2/pull/76)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.5...v0.4.6

[Changes][v0.4.6]


<a id="v0.4.5"></a>
## [v0.4.5](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.5) - 2025-06-08

### What's Changed
* Added a new section for training options in the documentation by [@jcwang587](https://github.com/jcwang587) in [#46](https://github.com/jcwang587/cgcnn2/pull/46)
* Info displayed with `logging` by [@jcwang587](https://github.com/jcwang587) in [#47](https://github.com/jcwang587/cgcnn2/pull/47)
* Added flexible size of caching for DataLoader by [@jcwang587](https://github.com/jcwang587) in [#52](https://github.com/jcwang587/cgcnn2/pull/52)
* Simplified dataset splitting process for training, validation, and testing by [@jcwang587](https://github.com/jcwang587) in [#54](https://github.com/jcwang587/cgcnn2/pull/54)
* Enhanced neighbor list loading with Cythonized `get_neighbor_list` by [@jcwang587](https://github.com/jcwang587) in [#60](https://github.com/jcwang587/cgcnn2/pull/60)
* Changed the output format of parity plots from SVG to PNG by [@jcwang587](https://github.com/jcwang587) in [#63](https://github.com/jcwang587/cgcnn2/pull/63)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.4...v0.4.5

[Changes][v0.4.5]


<a id="v0.4.4"></a>
## [v0.4.4](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.4) - 2025-05-29

### What's Changed
* Enhanced model inference efficiency by adopting `torch.inference_mode` by [@jcwang587](https://github.com/jcwang587) in [#38](https://github.com/jcwang587/cgcnn2/pull/38)
* Add option to force training set inclusion while not preserving the train:valid:test ratio by [@jcwang587](https://github.com/jcwang587) in [#39](https://github.com/jcwang587/cgcnn2/pull/39)
* Add `print_checkpoint_info` by [@jcwang587](https://github.com/jcwang587) in [#40](https://github.com/jcwang587/cgcnn2/pull/40)
* Set `shuffle=False` for validation and test set by [@jcwang587](https://github.com/jcwang587) in [#42](https://github.com/jcwang587/cgcnn2/pull/42)
* Remove `scikit-learn` dependency by [@jcwang587](https://github.com/jcwang587) in [#44](https://github.com/jcwang587/cgcnn2/pull/44)
* Switch the build system from `poetry` to `uv` for package building and publishing by [@jcwang587](https://github.com/jcwang587) in [#45](https://github.com/jcwang587/cgcnn2/pull/45)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.3...v0.4.4

[Changes][v0.4.4]


<a id="v0.4.3"></a>
## [v0.4.3](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.3) - 2025-05-22

### What's Changed
* Fix the error when inputting dataset without a split ratio by [@jcwang587](https://github.com/jcwang587) in [#36](https://github.com/jcwang587/cgcnn2/pull/36)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.2...v0.4.3

[Changes][v0.4.3]


<a id="v0.4.2"></a>
## [v0.4.2](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.2) - 2025-05-16

### What's Changed
* Allow custom x‑ and y‑axis labels in parity plot by [@jcwang587](https://github.com/jcwang587) in [#33](https://github.com/jcwang587/cgcnn2/pull/33)
* Sort output indices by structure ID when running predictions by [@jcwang587](https://github.com/jcwang587) in [#34](https://github.com/jcwang587/cgcnn2/pull/34)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.1...v0.4.2

[Changes][v0.4.2]


<a id="v0.4.1"></a>
## [v0.4.1](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.1) - 2025-04-29

### What's Changed
* Add CLI scripts `id_gen` and `atom_gen` by [@jcwang587](https://github.com/jcwang587) in [#23](https://github.com/jcwang587/cgcnn2/pull/23)
* Add documentation by [@jcwang587](https://github.com/jcwang587) in [#24](https://github.com/jcwang587/cgcnn2/pull/24)
* Remove the requirement for `id_prop.csv` when running predictions on unknown datasets by [@jcwang587](https://github.com/jcwang587) in [#29](https://github.com/jcwang587/cgcnn2/pull/29)


**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.4.0...v0.4.1

[Changes][v0.4.1]


<a id="v0.4.0"></a>
## [v0.4.0](https://github.com/jcwang587/cgcnn2/releases/tag/v0.4.0) - 2025-04-09

### What's Changed
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

### What's Changed
* Add option to force training set inclusion while preserving the train:valid:test ratio by [@jcwang587](https://github.com/jcwang587) in [#7](https://github.com/jcwang587/cgcnn2/pull/7)

**Full Changelog**: https://github.com/jcwang587/cgcnn2/compare/v0.3.2...v0.3.4

[Changes][v0.3.4]


<a id="v0.3.2"></a>
## [v0.3.2](https://github.com/jcwang587/cgcnn2/releases/tag/v0.3.2) - 2025-03-19

### What's Changed
* Add an option to the fine-tuning script to generate a parity plot within a user-specified range

### New Contributors
* [@jcwang587](https://github.com/jcwang587) made their first contribution

**Full Changelog**: https://github.com/jcwang587/cgcnn2/commits/v0.3.2

[Changes][v0.3.2]


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

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.0 -->
