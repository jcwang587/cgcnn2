import importlib.metadata
import warnings

__version__ = importlib.metadata.version("cgcnn2")

warnings.filterwarnings(
    "ignore", message="Issues encountered while parsing CIF", category=UserWarning
)
