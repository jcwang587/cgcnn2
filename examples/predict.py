from cgcnn2 import cgcnn_pred

data_path = "./data/sample-regression"
model_path = "../models/formation-energy-per-atom.pth.tar"

pred, last_layer = cgcnn_pred(
    model_path=model_path,
    all_set=data_path
)

