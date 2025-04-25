# Usage

## Basic Usage

Here's a basic tutorial on going through the prediction script using the functions provided by the package. 

### 1. Importing the package

There are three main modules available in the package:

- `cgcnn2.data`: For loading and preprocessing the data.
- `cgcnn2.model`: Building blocks for the CGCNN model.
- `cgcnn2.util`: Some utility functions.

```python
from cgcnn2.data import CIFData, collate_pool
from cgcnn2.model import CrystalGraphConvNet
from cgcnn2.util import cgcnn_test
```

### 2. Data Preparation

To input material structures into CGCNN, you need to define a custom dataset. Before doing so, make sure you have the following files:

- `CIF` files recording the structures of the materials you wish to study.
- Target properties for each material (not needed for prediction jobs).

Organize these files in a directory (`root_dir`) with the following structure:

1. `id_prop.csv` (optional for prediction)
    A CSV with two columns, the first column is a unique material ID, and the second column is the corresponding target property value.
2. `atom_init.json`
    A JSON file that provides the initialization vector for each element. You can use the example at `/cgcnn2/asset/atom_init.json` from the original CGCNN repository; it should work for most applications.
3. `.cif` files
    One `.cif` file per material, named `ID.cif`, where `ID` matches the entries in `id_prop.csv`.

Once your `root_dir` (for example, `/examples/data/sample_regression`) contains these files, you can load the dataset using the `CIFData` class:

```python
dataset = CIFData("/examples/data/sample_regression")
```

This will prepare your crystal structures (and, if provided, their target properties) for use with CGCNN. Then, we can build a `torch.utils.data.DataLoader` object that can be used to load the dataset in a batch.

```python
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    collate_fn=collate_pool,
)
```

### 3. Model Initialization

We need some information about the model architecture to initialize the model, which can be done by:

```python
atom_graph, _, _ = dataset[0]
orig_atom_fea_len = atom_graph[0].shape[-1]
nbr_fea_len = atom_graph[1].shape[-1]
```

`dataset[0]` is a tuple of `(atom_graph, target, cif_id)`, where `atom_graph` is a tuple of `(atom_fea, nbr_fea, nbr_fea_idx)`, `target` is the target property value, and `cif_id` is the unique ID of the material. The `atom_graph` tuple contains the atom features, neighbor features, and neighbor indices, and the dimensions of these features given by `orig_atom_fea_len` and `nbr_fea_len` can be used to initialize the model. Now, we can load the model checkpoint and run the prediction.

```python
model = CrystalGraphConvNet(
    orig_atom_fea_len=orig_atom_fea_len,
    nbr_fea_len=nbr_fea_len,
    atom_fea_len=model_args.atom_fea_len,
    n_conv=model_args.n_conv,
    h_fea_len=model_args.h_fea_len,
    n_h=model_args.n_h,
)
checkpoint = torch.load(args.model_path, map_location=args.device)
model_args = argparse.Namespace(**checkpoint["args"])

model.load_state_dict(checkpoint["state_dict"])
model.to(args.device)
model.eval()

cgcnn_test(
    model=model,
    loader=loader,
    device=args.device,
    plot_file=os.path.join(output_folder, "parity_plot.svg"),
)
```
