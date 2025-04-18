# API Reference

## Data Module

::: cgcnn2.data
    options:
      heading_level: 2

### `CIFData`
```python
class CIFData(root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2)
```
Loads and processes CIF files for training.

Parameters:
- `root_dir`: Root directory containing the CIF files
- `max_num_nbr`: Maximum number of neighbors to consider
- `radius`: Cutoff radius for finding neighbors
- `dmin`: Minimum distance for neighbor search
- `step`: Step size for distance grid

## Model Module

::: cgcnn2.model
    options:
      heading_level: 2

### `CrystalGraphConvNet`
```python
class CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, atom_fea_len=64,
                         n_conv=3, h_fea_len=128, n_h=1)
```
Main model class for crystal graph convolutional neural network.

Parameters:
- `orig_atom_fea_len`: Original atom feature length
- `nbr_fea_len`: Neighbor feature length
- `atom_fea_len`: Atom feature length
- `n_conv`: Number of convolutional layers
- `h_fea_len`: Hidden feature length
- `n_h`: Number of hidden layers

## Utility Functions

::: cgcnn2.utils
    options:
      heading_level: 2

### `collate_pool`
```python
def collate_pool(dataset_list)
```
Collate function for creating mini-batches.

Parameters:
- `dataset_list`: List of data samples

Returns:
- Batched data for training 