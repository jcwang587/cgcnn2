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

Once your `root_dir` (for example, `examples/data/sample-regression`) contains these files, you can load the dataset using the `CIFData` class:

```python
dataset = CIFData("examples/data/sample-regression")
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

### 3. Model Loading

A checkpoint holds the model weights and the hyperparameters of the architecture, so the model is rebuilt from the checkpoint and the dataset, which provides the input feature lengths:

```python
import torch
from cgcnn2.utils import load_checkpoint, model_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = load_checkpoint("formation-energy-per-atom.pth.tar")
model = model_from_checkpoint(checkpoint, dataset).to(device)
```

Checkpoints contain only tensors and plain Python values, so they load with the default `weights_only=True`; only load checkpoint files from sources you trust.

### 4. Prediction

`cgcnn_test` writes the predictions to a CSV file and, when the dataset has an `id_prop.csv`, logs the MAE, MSE and R² and saves a parity plot:

```python
cgcnn_test(
    model, loader, device, results_file="results.csv", plot_file="parity_plot.png"
)
```

For downstream analysis in Python, `cgcnn_pred` loads the dataset and the model in one call and returns the predictions and the crystal descriptors (the features before the output layer) as NumPy arrays; it writes no files and computes no metrics:

```python
from cgcnn2.pretrained_models import FORMATION_ENERGY_MODEL
from cgcnn2.utils import cgcnn_pred

predictions, descriptors = cgcnn_pred(
    FORMATION_ENERGY_MODEL, "examples/data/sample-regression"
)
```

## Training Options

You can view the training hyperparameters by using the `--help` flag with both the `cgcnn-tr` and `cgcnn-ft` commands.
Most of the hyperparameters are shared between these two scripts, including:

- `batch-size`: Specifies the number of samples that are grouped together and processed in one forward/backward pass during both training and fine-tuning. At each step, the DataLoader will pull `batch-size` examples from the dataset, feed them through the model, compute the loss, and then perform a gradient update. Choosing the right `batch-size` involves a trade-off between memory usage, computational efficiency, and optimization dynamics.
  By default, we set `batch-size = 256` as a balance between speed and memory requirements on modern GPUs. If you run into OOM errors, try reducing this value; if you have excess memory and want to speed up training, experiment with increasing it.

- `learning-rate`: Determines the step size used by the optimizer to update model weights at each gradient step. A higher learning rate makes larger jumps in parameter space, potentially speeding up initial convergence but risking instability or divergence. A lower rate yields more precise, stable updates at the cost of slower training. By default, we set learning-rate = 1e-2 as a good starting point for most graph-based models. If loss oscillates or diverges, try reducing it; if training stalls, consider increasing it or adding a scheduler.

- `epoch`: Specifies how many epochs the training loop will make over the entire dataset during both training and fine-tuning. More epochs give the model more opportunities to learn but increase total runtime and can lead to overfitting if too many are used. The default is `epoch = 1000`. Monitor your validation loss and use early stopping if you observe that performance plateaus.

- `device`: The torch device used for the model, e.g. `cpu`, `cuda`, `cuda:1` or `mps`. By default `cuda` is used when it is available, otherwise `cpu`.

### Targets and Checkpoints

Targets are standardized internally with the mean and standard deviation of the training set, so one learning rate suits properties of any unit and scale. The loss is the mean squared error on the standardized targets. After every epoch the model is evaluated on the validation set, and the checkpoint with the lowest validation MAE is kept. The statistics are folded into the output layer of the saved checkpoint, so a checkpoint predicts directly in the units of your data and needs nothing else at prediction time.

### Early Stopping

Both the training and fine-tuning scripts implement an early stopping strategy that halts training if the validation MAE fails to improve for a specified number of consecutive epochs.

- `stop-patience`: Defines how many consecutive epochs without any decrease in validation MAE are allowed before training is terminated. Default is `None`, which means no early stopping.

### Learning Rate Scheduler

There is a learning rate scheduler for the training and finetuning scripts, which applies the `ReduceLROnPlateau` strategy. The default parameters are:

- `factor`: The factor by which the learning rate will be reduced. `new_lr = lr * factor`
- `patience`: How many epochs to wait before reducing the learning rate.

You can check more details in the [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
