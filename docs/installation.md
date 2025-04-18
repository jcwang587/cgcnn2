# Installation

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/jcwang587/cgcnn2.git
cd cgcnn2
```

2. Install the package:
```bash
pip install -e .
```

## Dependencies

The package requires the following dependencies:
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Pymatgen

These will be automatically installed when you install the package.

## Verifying Installation

To verify your installation, run:
```python
import cgcnn2
print(cgcnn2.__version__)
``` 