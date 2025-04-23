# Usage

## Basic Usage

Here's a basic example of how to use CGCNN2:

Let's first try to import a NaCl crystal structure from a CIF file.

```python
import cgcnn2
from cgcnn2.data import CIFData
from cgcnn2.model import CrystalGraphConvNet

# Load your data
dataset = CIFData('path/to/your/data')

# Initialize the model
model = CrystalGraphConvNet(
    orig_atom_fea_len=92,
    nbr_fea_len=41,
    atom_fea_len=64,
    n_conv=3,
    h_fea_len=128,
    n_h=1
)

# Train the model
# ... (training code here)
```

## Data Preparation

CGCNN2 expects data in CIF format. Make sure your data is properly formatted before training.

## Model Configuration

The model can be configured with various parameters:
- `orig_atom_fea_len`: Original atom feature length
- `nbr_fea_len`: Neighbor feature length
- `atom_fea_len`: Atom feature length
- `n_conv`: Number of convolutional layers
- `h_fea_len`: Hidden feature length
- `n_h`: Number of hidden layers

## Training

Detailed training instructions will be added here. 