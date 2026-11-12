import os

MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

BAND_GAP_MODEL = os.path.join(MODEL_DIR, "band-gap.pth.tar")
BULK_MODULI_MODEL = os.path.join(MODEL_DIR, "bulk-moduli.pth.tar")
FORMATION_ENERGY_MODEL = os.path.join(MODEL_DIR, "formation-energy-per-atom.pth.tar")
SHEAR_MODULI_MODEL = os.path.join(MODEL_DIR, "shear-moduli.pth.tar")
