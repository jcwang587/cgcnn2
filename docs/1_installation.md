# Installation

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

## Installation Steps

It is recommended to first check your available CUDA version (or CPU only) and install PyTorch following the instructions on the PyTorch official [website](https://pytorch.org/get-started/locally/). Then, you can simply install `cgcnn2` from PyPI using pip:

```bash
pip install cgcnn2
```

If you'd like to use the latest unreleased version on the main branch, you can install it directly from GitHub:

```bash
pip install git+https://github.com/jcwang587/cgcnn2@main
```

## Dependencies

The package requires the following dependencies:

* [matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pymatgen](https://pymatgen.org/)
* [pymatviz](https://pymatviz.janosh.dev/)
* [PyTorch](https://pytorch.org/)

These will be automatically installed when you install the package.

## Verifying Installation

To verify your installation, run:
```python
import cgcnn2
print(cgcnn2.__version__)
``` 
