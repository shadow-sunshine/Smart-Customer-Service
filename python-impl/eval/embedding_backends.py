"""
Embedding 后端实现。

保留 `baseline` —— 精确复现项目自带的 sha256 伪随机实现，
用于跑出"改造前"的基线数字，作为优化对照。
其余为真实语义 embedding，用于修复检索层。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

EVAL_FILE = Path(__file__).resolve().parent / "eval_set.json"


def _load_documents() -> list[dict]:
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    return data["documents"]


class BaseEmbedding(ABC):
    """检索后端抽象。子类实现 _embed 与 similarity。"""

    name = "base"
    dim = 1536

    def __init__(self) -> None:
        self.documents = _load_documents()
        self.vectors: np.ndarray | None = None

    @abstractmethod
    def _embed(self, text: str) -> np.ndarray:
        ...

    @abstractmethod
    def similarity(self, query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
        ...

    def build(self) -> None:
        self.vectors = np.vstack([self._embed(d["content"]) for d in self.documents])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.vectors is None:
            self.build()
        qv = self._embed(query)
        scored = []
        # vectors 可能是 np.ndarray（稠密）或 list[set]（稀疏），
        # 两者都支持 len() 与逐元素 zip 遍历。
        for doc, vec in zip(self.documents, self.vectors):
            scored.append(
                {
                    "source": doc["source"],
                    "content": doc["content"],
                    "score": self.similarity(qv, vec),
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class BaselineEmbedding(BaseEmbedding):
    """
    精确复现项目原始实现（memory/long_term.py:66-75）：

        text_hash = sha256(text).hexdigest()
        np.random.seed(int(text_hash[:8], 16) % 2**32)
        vec = np.random.randn(dim); vec /= norm(vec)

    没有任何语义：同一段文本永远得到同一向量，不同文本之间相似度
    随机分布。用于量化"改造前"的检索质量基线。
    """

    name = "baseline"

    def _embed(self, text: str) -> np.ndarray:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16) % (2**32))
        vec = np.random.randn(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def similarity(self, query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
        # 原实现用 FAISS IndexFlatIP，即内积
        return float(np.dot(query_vec, doc_vec))


class KeywordEmbedding(BaseEmbedding):
    """
    无外部依赖的关键词/字符级匹配基线。

    用字符二元组 + Jaccard 相似度，对中文比单词切分更稳，
    能在没有 embedding API 的环境下提供一个真实可用的下界基线。

    注意：二元组集合长度随文本变化，不能像稠密向量那样 vstack，
    因此直接存 set 列表，相似度在search 时现算。
    """

    name = "keyword"
    dim = 0

    def __init__(self) -> None:
        super().__init__()
        # 变长集合，无法用 np.vstack，直接存 set
        self.vectors: list[set[str]] | None = None

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        cleaned = re.sub(r"\s+", "", text)
        return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)} or {cleaned}

    def _embed(self, text: str) -> set[str]:
        return self._bigrams(text)

    def build(self) -> None:
        self.vectors = [self._bigrams(d["content"]) for d in self.documents]

    def similarity(self, query_vec, doc_vec) -> float:
        a, b = query_vec, doc_vec
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


class OpenAIEmbedding(BaseEmbedding):
    """
    真实语义 embedding（OpenAI 兼容协议，含 DeepSeek）。

    与项目 ai_service 一致的做法：走 OpenAI 兼容端点，
    通过 OPENAI_BASE_URL 切换官方 / 中转 / DeepSeek。
    """

    name = "openai"
    dim = 1536

    def __init__(self) -> None:
        super().__init__()
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            base_url = os.getenv("OPENAI_BASE_URL") or None
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "缺少 OPENAI_API_KEY 环境变量，无法使用 openai embedding backend"
                )
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def _embed(self, text: str) -> np.ndarray:
        resp = self._get_client().embeddings.create(model=self.model, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def build(self) -> None:
        # 真实 API 有网络开销，批量调用减少往返
        texts = [d["content"] for d in self.documents]
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=texts)
        self.vectors = np.vstack(
            [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        )

    def similarity(self, query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
        return float(np.dot(query_vec, doc_vec))


EMBEDDING_BACKENDS = {
    "baseline": BaselineEmbedding,
    "keyword": KeywordEmbedding,
    "openai": OpenAIEmbedding,
}


def get_backend(name: str) -> BaseEmbedding:
    if name not in EMBEDDING_BACKENDS:
        raise ValueError(
            f"未知 backend: {name}，可选: {list(EMBEDDING_BACKENDS.keys())}"
        )
    return EMBEDDING_BACKENDS[name]()
