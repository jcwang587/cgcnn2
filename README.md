# CGCNN2

[![PyPI](https://img.shields.io/pypi/v/cgcnn2)](https://pypi.org/project/cgcnn2/)
[![PyPI Downloads](https://static.pepy.tech/badge/cgcnn2)](https://pepy.tech/projects/cgcnn2)

A maintained reproduction of the Crystal Graph Convolutional Neural Network ([CGCNN](https://github.com/txie-93/cgcnn)) by Xie and Grossman, whose original repository is no longer active. CGCNN remains a small, fast model for predicting material properties from crystal structures that is easy to learn and use.

- **Train** a model on your own dataset with `cgcnn-tr`.
- **Predict** material properties with a trained model with `cgcnn-pr`.
- **Fine-tune** a trained model on a new dataset with `cgcnn-ft`.
- **Extract** crystal descriptors for downstream tasks with `cgcnn_pred`.

## Installation

```bash
pip install cgcnn2
```

## Documentation

The user guide, examples and API reference are at [jcwang.dev/cgcnn2](https://jcwang.dev/cgcnn2/).

## Reference

Tian Xie and Jeffrey C. Grossman, "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties", [Phys. Rev. Lett. 120, 145301 (2018)](https://doi.org/10.1103/PhysRevLett.120.145301).
