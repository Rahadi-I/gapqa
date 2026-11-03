"""Track W reference implementation (the code that produced the paper's runs; imports adapted to the package).

    from gapqa.tracks.track_w import GLAConfig, GLAModel, ChatFormat, score_items

`score_items` writes one JSON line per item with the decision score s and, for recoverable items, the greedy
answer after a forced "ANSWER:" prefix. Every forward pass carries an explicit all-ones attention mask: recent
transformers versions otherwise read jumps in position_ids as packed-sequence boundaries (paper, Section 4.5).
Requires `pip install gapqa[trackw]`.
"""
from .chat import SYSTEM, ChatFormat
from .gap_module import GapEmbedding, fourier_features
from .model import GLAConfig, GLAModel, to_batch
from .runner import score_items
from .sequence import Sequence, build_sequence_v1, decision_token_ids_v1

__all__ = ["SYSTEM", "ChatFormat", "GapEmbedding", "fourier_features", "GLAConfig", "GLAModel", "to_batch",
           "score_items", "Sequence", "build_sequence_v1", "decision_token_ids_v1"]
