# Installation

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation Steps

Make sure you have a Python interpreter, preferably version 3.10 or higher. Then, you can simply install `cgcnn2` from
PyPI using `pip`:

```bash
pip install cgcnn2
```

If you'd like to use the latest unreleased version on the main branch, you can install it directly from GitHub:

```bash
pip install git+https://github.com/jcwang587/cgcnn2
```

## Dependencies

The package requires the following dependencies:

* [Ase](https://wiki.fysik.dtu.dk/ase/)
* [Matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Pymatgen](https://pymatgen.org/)
* [Pymatviz](https://pymatviz.janosh.dev/)
* [PyTorch](https://pytorch.org/)
* [Scikit-learn](https://scikit-learn.org/)

These will be automatically installed when you install the package.

## Verifying Installation

To verify your installation, run:
```python
import cgcnn2
print(cgcnn2.__version__)
``` 