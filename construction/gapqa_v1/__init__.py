"""GapQA v1 (protocol v1.1): model-agnostic items stored as text with character offsets."""
from .types import ItemV1
from .render import RenderedV1, render_v1

__all__ = ["ItemV1", "RenderedV1", "render_v1"]
