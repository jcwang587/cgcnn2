# Examples

## Basic Example

Here's a basic example of training a CGCNN model:

```python
import torch
from cgcnn2.data import CIFData
from cgcnn2.model import CrystalGraphConvNet

# Load data
dataset = CIFData('path/to/data')
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True,
    collate_fn=collate_pool, num_workers=0)

# Initialize model
model = CrystalGraphConvNet(
    orig_atom_fea_len=92,
    nbr_fea_len=41,
    atom_fea_len=64,
    n_conv=3,
    h_fea_len=128,
    n_h=1
)

# Training loop
for epoch in range(100):
    for batch in train_loader:
        # Training code here
        pass
```

## Advanced Example

For more advanced usage, including hyperparameter tuning and custom data processing, see the examples in the `examples` directory of the repository.

## Available Examples

1. Basic training
2. Hyperparameter optimization
3. Custom data processing
4. Model evaluation

Each example includes detailed comments and explanations of the code. 