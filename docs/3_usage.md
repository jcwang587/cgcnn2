# Usage

## Basic Usage

Here's a basic tutorial on going through the prediction pipeline using the package. 

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

### 2. Loading the data


To input crystal structures to CGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting.

Before defining a customized dataset, you will need:

- `CIF` files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in id_prop.csv)

You can create a customized dataset by creating a directory root_dir with the following files:

id_prop.csv: a CSV file with two columns. The first column recodes a unique ID for each crystal, and the second column recodes the value of target property. If you want to predict material properties with predict.py, you can put any number in the second column. (The second column is still needed.)

atom_init.json: a JSON file that stores the initialization vector for each element. An example of atom_init.json is data/sample-regression/atom_init.json, which should be good for most applications.

ID.cif: a CIF file that recodes the crystal structure, where ID is the unique ID for the crystal.

We are going to use the `CIFData` class to load the data.

```python
data = CIFData("../examples/data/sample_regression")
```




