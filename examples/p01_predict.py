from cgcnn2.cgcnn_utils import cgcnn_pred
from cgcnn2.models import FORMATION_ENERGY_MODEL

data_path = "./data/sample-regression"

pred, last_layer = cgcnn_pred(model_path=FORMATION_ENERGY_MODEL, all_set=data_path)
