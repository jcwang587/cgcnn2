# CGCNN2

[![Release](https://img.shields.io/github/v/release/jcwang587/cgcnn2)](https://github.com/jcwang587/cgcnn2/releases/latest)
[![PyPI Downloads](https://static.pepy.tech/badge/cgcnn2)](https://pepy.tech/projects/cgcnn2)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

As the original Crystal Graph Convolutional Neural Networks (CGCNN) repository is no longer actively maintained, this repository is a reproduction of [CGCNN](https://github.com/txie-93/cgcnn) by Xie *et al*. It includes necessary updates for deprecated components and a few additional functions to ensure smooth operation. Despite its age, CGCNN remains a straightforward and fast deep learning framework that is easy to learn and use.

The package provides the following major functions:

- **Training** a CGCNN model using a custom dataset.
- **Predicting** material properties with a pre-trained CGCNN model.
- **Fine-tuning** a pre-trained CGCNN model on a new dataset.
- **Extracting** structural features as descriptors for downstream tasks.
<!---**Augmenting** training data by pertubing atomic positions (in development).-->

## Installation

Make sure you have a Python interpreter, preferably version 3.11 or higher. Then, you can simply install cgcnn2 from
PyPI using `pip`:

```bash
pip install cgcnn2
```

If you'd like to use the latest unreleased version on the main branch, you can install it directly from GitHub:

```bash
pip install git+https://github.com/jcwang587/cgcnn2@main
```

## Get Started

There are entry points for training, predicting, and fine-tuning CGCNN models. For example, to explore the usage of the provided training script `cgcnn-tr`, you can use the `--help` option of the command:

```bash
cgcnn-tr --help
```

Similarly, you can access the predicting and fine-tuning functionalities through `cgcnn-pr` and `cgcnn-ft` commands. A detailed user guide documentation is available at: [https://jcwang.dev/cgcnn2/](https://jcwang.dev/cgcnn2/)

## References

The original paper describes the CGCNN framework in detail:

```bibtex
@article{PhysRevLett2018,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301}
}
```
