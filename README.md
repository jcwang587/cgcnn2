# CGCNN2

[![Release](https://img.shields.io/github/v/release/jcwang587/cgcnn2)](https://github.com/jcwang587/cgcnn2/releases/latest)
[![PyPI Downloads](https://static.pepy.tech/badge/cgcnn2)](https://pepy.tech/projects/cgcnn2)
  
As the original Crystal Graph Convolutional Neural Networks (CGCNN) repository is no longer actively maintained, this repository is a reproduction of [CGCNN](https://github.com/txie-93/cgcnn) by Xie et al. It includes necessary updates for deprecated components and a few additional functions to ensure smooth operation. Despite its age, CGCNN remains a straightforward and fast deep learning framework that is easy to learn and use.

The package provides following major functions:

- **Training** a CGCNN model with a customized dataset.
- **Predicting** material properties with a pre-trained CGCNN model.
- **Fine-tuning** a pre-trained CGCNN model on a new dataset.
- **Extracting** atomic features as descriptors for the downstream task.

## Installation

Make sure you have a Python interpreter, preferably version 3.10 or higher. Then, you can simply install xdatbus from
PyPI using `pip`:

```bash
pip install cgcnn2
```

If you'd like to use the latest unreleased version on the main branch, you can install it directly from GitHub:

```bash
pip install git+https://github.com/jcwang587/cgcnn2
```

## Get Started

```bash
cgcnn-ft --help
```


## References

The original paper describes the details of the CGCNN framework:

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
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
```
