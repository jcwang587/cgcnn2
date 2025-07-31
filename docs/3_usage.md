# Usage

## Basic Usage

Here's a basic tutorial on going through the prediction script using the functions provided by the package.

### 1. Importing the package

There are three main modules available in the package:

- `cgcnn2.data`: For loading and preprocessing the data.
- `cgcnn2.model`: Building blocks for the CGCNN model.
- `cgcnn2.utils`: Some utility functions.

```python
from cgcnn2.data import CIFData, collate_pool
from cgcnn2.model import CrystalGraphConvNet
from cgcnn2.utils import cgcnn_test
```

### 2. Data Preparation

To input material structures into CGCNN, you need to define a custom dataset. Before doing so, make sure you have the following files:

- `CIF` files recording the structures of the materials you wish to study.
- Target properties for each material (not needed for prediction jobs).

Organize these files in a directory (`root_dir`) with the following structure:

1. `id_prop.csv` (optional for prediction):
   A CSV with two columns, the first column is a unique material ID, and the second column is the corresponding target property value.
1. `atom_init.json`:
   A `JSON` file that provides the initialization vector for each element. You can use the example at `/cgcnn2/asset/atom_init.json` from the original CGCNN repository; it should work for most applications.
1. `CIF` files:
   One `.cif` file per material, named `ID.cif`, where `ID` matches the entries in `id_prop.csv`.

Once your `root_dir` (for example, `/examples/data/sample_regression`) contains these files, you can load the dataset using the `CIFData` class:

```python
dataset = CIFData("/examples/data/sample_regression")
```

This will prepare your crystal structures (and, if provided, their target properties) for use with CGCNN. Then, we can build a `torch.utils.data.DataLoader` object that can be used to load the dataset in a batch.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    collate_fn=collate_pool,
)
```

### 3. Model Initialization

We need some information from the dataset to initialize the model, which can be done by:

```python
atom_graph, _, _ = dataset[0]
orig_atom_fea_len = atom_graph[0].shape[-1]
nbr_fea_len = atom_graph[1].shape[-1]
```

where `dataset[0]` is a tuple of `(atom_graph, target, cif_id)`, where `atom_graph` is a tuple of `(atom_fea, nbr_fea, nbr_fea_idx)`, `target` is the target property value, and `cif_id` is the unique ID of the material. The `atom_graph` tuple contains the atom features, neighbor features, and neighbor indices, and the dimensions of these features given by `orig_atom_fea_len` and `nbr_fea_len` are needed to initialize the model.

Besides, we need some information about the pre-trained model architecture, which can be done by:

```python
import torch
import argparse

checkpoint = torch.load(args.model_path, map_location=args.device)
model_args = argparse.Namespace(**checkpoint["args"])
atom_fea_len = model_args.atom_fea_len
n_conv = model_args.n_conv
h_fea_len = model_args.h_fea_len
n_h = model_args.n_h
```

where `atom_fea_len`, `n_conv`, `h_fea_len`, and `n_h` are the dimensions of the atom features, the number of convolutional layers, the dimension of the hidden features, and the number of hidden layers, respectively. Now, we can initialize the model by:

```python
model = CrystalGraphConvNet(
    orig_atom_fea_len=orig_atom_fea_len,
    nbr_fea_len=nbr_fea_len,
    atom_fea_len=atom_fea_len,
    n_conv=n_conv,
    h_fea_len=h_fea_len,
    n_h=n_h,
)
```

### 4. Model Loading and Prediction

The `checkpoint` from the pre-trained model includes the model's state dictionary and training arguments. The state dictionary contains all the learned parameters of the model, while the training arguments store the hyperparameters used during training. The model can be loaded onto either CPU or GPU device by specifying `device` as `cpu` or `cuda`. Using GPU is recommended for faster inference if available.

```python
model.load_state_dict(checkpoint["state_dict"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

Now, we can run the prediction with the `cgcnn_test` utility function:

```python
cgcnn_test(
    model=model,
    loader=loader,
    device=device,
    plot_file=os.path.join(output_folder, "parity_plot.svg"),
)
```

## Training Options

You can view the training hyperparameters by using the `--help` flag with both the `cgcnn-tr` and `cgcnn-ft` commands.
Most of the hyperparameters are shared between these two scripts, including:

- `batch-size`: Specifies the number of samples that are grouped together and processed in one forward/backward pass during both training and fine-tuning. At each step, the DataLoader will pull `batch-size` examples from the dataset, feed them through the model, compute the loss, and then perform a gradient update. Choosing the right `batch-size` involves a trade-off between memory usage, computational efficiency, and optimization dynamics.
  By default, we set `batch-size = 256` as a balance between speed and memory requirements on modern GPUs. If you run into OOM errors, try reducing this value; if you have excess memory and want to speed up training, experiment with increasing it.

- `learning-rate`: Determines the step size used by the optimizer to update model weights at each gradient step. A higher learning rate makes larger jumps in parameter space, potentially speeding up initial convergence but risking instability or divergence. A lower rate yields more precise, stable updates at the cost of slower training. By default, we set learning-rate = 1e-2 as a good starting point for most graph-based models. If loss oscillates or diverges, try reducing it; if training stalls, consider increasing it or adding a scheduler.

- `epoch`: Specifies how many epochs the training loop will make over the entire dataset during both training and fine-tuning. More epochs give the model more opportunities to learn but increase total runtime and can lead to overfitting if too many are used. The default is `epoch = 1000`. Monitor your validation loss and use early stopping if you observe that performance plateaus.

- `device`: Indicates which hardware backend to use for both data loading and model computation. Usually `cuda` for NVIDIA GPUs or `cpu`. When set to `cuda`, tensors and the model are moved onto the GPU. By default, we auto-detect the device by checking if `cuda` is available. Using the GPU generally yields significant speed-ups.

```bash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Early Stopping

Both the training and fine-tuning scripts implement an early stopping strategy that halts training if the validation loss fails to improve for a specified number of consecutive epochs.

- `stop-patience`: Defines how many consecutive epochs without any decrease in validation loss are allowed before training is terminated. Default is `None`, which means no early stopping.

### Learning Rate Scheduler

There is a learning rate scheduler for the training and finetuning scripts, which applies the `ReduceLROnPlateau` strategy. The default parameters are:

- `factor`: The factor by which the learning rate will be reduced. `new_lr = lr * factor`
- `patience`: How many epochs to wait before reducing the learning rate.

You can check more details in the [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
