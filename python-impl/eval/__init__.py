"""评测包：量化 RAG 检索质量。"""

from eval.metrics import EvalReport, evaluate_retrieval
from eval.embedding_backends import EMBEDDING_BACKENDS, get_backend

__all__ = [
    "EvalReport",
    "evaluate_retrieval",
    "EMBEDDING_BACKENDS",
    "get_backend",
]
