"""
embedding 后端选择：把「用哪个实现」抽象成可配置项。

原项目的问题（改造起点）：
`memory/long_term.py` 的 `_simple_embedding()` 用 sha256(text) 当随机种子
生成伪随机向量，没有任何语义。实测该实现在 17 条标注集上
Recall@1 仅 38.5%，且知识库外问题全部瞎答（拒答率 0%）。

本模块提供可切换的实现：
- `keyword`  : 字符二元组 + Jaccard，零外部依赖，离线可用
- `openai`   : OpenAI 兼容 embedding（含 DeepSeek），需 API Key
- `fake`     : 保留原项目的伪随机实现，仅用于回归对照

选型依据见 docs/eval_report.md：用评测集量化对比，而非凭感觉选。
"""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod

import numpy as np

EMBEDDING_DIM = 1536


class BaseEmbedding(ABC):
    name = "base"

    def _embed(self, text: str) -> np.ndarray:  # pragma: no cover - 抽象
        raise NotImplementedError

    def _doc_vector(self, text: str) -> np.ndarray:
        return self._embed(text)

    def similarity(self, query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
        """内积。因为所有向量都已 L2 归一化，内积即余弦相似度。"""
        return float(np.dot(query_vec, doc_vec))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class FakeRandomEmbedding(BaseEmbedding):
    """
    复刻原项目实现：sha256 当种子生成伪随机向量。

    仅用于对照评测，证明「无语义 embedding 的检索质量有多差」。
    保留它是为了让改造前后的数字可比，不是为了在生产中使用。
    """

    name = "fake"

    def _embed(self, text: str) -> np.ndarray:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        np.random.seed(int(text_hash[:8], 16) % (2**32))
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)


class KeywordBigramEmbedding(BaseEmbedding):
    """
    字符二元组 + Jaccard 相似度。

    为什么不用分词：中文场景下 jieba 分词会引入词典依赖，
    而客服 query 往往是口语化短句。字符二元组无需词典、
    对错别字和未登录词更鲁棒，且实测在本评测集上表现最好。

    相似度定义用 Jaccard（交集/并集）而非余弦：
    短 query 与长文档做余弦会被文档长度稀释，
    Jaccard 只看重叠比例，更适合「短问对长答」的检索场景。
    """

    name = "keyword"

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        cleaned = re.sub(r"\s+", "", text)
        if len(cleaned) < 2:
            return {cleaned}
        return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}

    def _embed(self, text: str) -> np.ndarray:
        return np.array(sorted(self._bigrams(text)), dtype=object)

    def similarity(self, query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
        a, b = set(query_vec.tolist()), set(doc_vec.tolist())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI 兼容 embedding，支持 DeepSeek 等国内服务。

    通过 OPENAI_BASE_URL 切换端点，OPENAI_API_KEY 注入凭据。
    文档向量做批量请求以减少网络往返。
    """

    name = "openai"

    def __init__(self) -> None:
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = None
        self._cache: dict[str, np.ndarray] = {}

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("缺少 OPENAI_API_KEY，无法使用 openai embedding")
            self._client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
        return self._client

    def _embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            resp = self._get_client().embeddings.create(
                model=self.model, input=[text]
            )
            self._cache[text] = np.array(
                resp.data[0].embedding, dtype=np.float32
            )
        return self._cache[text]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """批量向量化文档，减少 API 往返。"""
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=texts)
        vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        for text, vec in zip(texts, vectors):
            self._cache[text] = vec
        return np.vstack(vectors)


BACKENDS = {
    "fake": FakeRandomEmbedding,
    "keyword": KeywordBigramEmbedding,
    "openai": OpenAIEmbedding,
}


def get_embedding(name: str | None = None) -> BaseEmbedding:
    """
    获取 embedding 实现。

    优先级：显式参数 > EMBEDDING_TYPE 环境变量 > keyword（离线默认）。
    默认不用 fake，因为它是错的；默认 keyword 是因为它零依赖且可跑。
    """
    key = name or os.getenv("EMBEDDING_TYPE", "keyword")
    if key not in BACKENDS:
        raise ValueError(f"未知 embedding 类型: {key}，可选: {list(BACKENDS)}")
    return BACKENDS[key]()
