"""Model architectures for OceanEmbed."""

from .encoder import SatelliteEmbeddingEncoder
from .decoder import DepthReconstructionDecoder
from .loss import PhysicsInformedLoss
from .ocean_embed_net import OceanEmbedNet

__all__ = [
    'SatelliteEmbeddingEncoder',
    'DepthReconstructionDecoder',
    'PhysicsInformedLoss',
    'OceanEmbedNet'
]
