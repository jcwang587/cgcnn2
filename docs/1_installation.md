# Installation

## Prerequisite

- Python 3.11 or higher

## Installation Steps

It is recommended to first check your available CUDA version (or CPU only) and install PyTorch following the instructions on the PyTorch official [website](https://pytorch.org/get-started/locally/). Then, you can simply install `cgcnn2` from [PyPI](https://pypi.org/project/cgcnn2/) using pip:

```bash
pip install cgcnn2
```

If you'd like to use the latest unreleased version on the main branch, you can install it directly from GitHub:

```bash
pip install git+https://github.com/jcwang-dev/cgcnn2@main
```

## Dependencies

The package requires the following dependencies:

* [matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pymatgen](https://pymatgen.org/)
* [PyTorch](https://pytorch.org/)

These will be automatically installed when you install the package.

## Verifying Installation

To verify your installation, launch the Python interpreter and run:
```python
import cgcnn2
print(cgcnn2.__version__)
``` 
