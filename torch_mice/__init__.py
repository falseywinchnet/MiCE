__version__ = "0.1.0"

from .positive_linear import PositiveLinearHK, PositiveLinear3DHK
from .batched_icnn    import BatchedICNN
from .affine_norm     import BatchAffineNorm
from .atlas_projector import AtlasProjector, SingleStiefelProjector, SmoothStiefelProjector
from .convex_attention import ConvexExpansionAttention, ConvexContractionAttention
from .convex_embed import PositiveEmbeddingHK, GeometricConvexEmbedding

from .convex_gate     import ConvexGate
from .vector_hull     import VectorHull
