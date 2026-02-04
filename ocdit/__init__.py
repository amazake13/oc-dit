"""OC-DiT: Object-Centric Diffusion Transformer for Zero-Shot Instance Segmentation."""

from ocdit.models.ocdit import OCDiT
from ocdit.models.diffuser import Diffuser
from ocdit.models.vae import BinaryMaskVAE
from ocdit.models.feature_extractor import SpatialDinov2

__version__ = "0.1.0"

__all__ = [
    "OCDiT",
    "Diffuser",
    "BinaryMaskVAE",
    "SpatialDinov2",
]
