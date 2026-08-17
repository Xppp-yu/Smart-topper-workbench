"""Contact-mask and geometry primitives for pressure maps."""

from .popu import GEOMETRY_COLUMNS, build_contact_mask, describe_geometry
from .mask_strategies import MASK_STRATEGIES, build_strategy_mask

__all__ = ("GEOMETRY_COLUMNS", "MASK_STRATEGIES", "build_contact_mask", "build_strategy_mask", "describe_geometry")
