"""
嵌入与向量存储模块 - Embedding & VectorStore

核心概念:
- Embedding: 将文本转换为高维向量，语义相近的文本向量距离近
- VectorStore: 存储向量并支持高效相似度检索
- 混合检索: 向量检索(语义) + BM25(关键词) → RRF融合

学习要点:
1. 向量相似度用余弦相似度衡量
2. IVF/ANN索引加速大规模检索(这里用暴力搜索演示原理)
3. BM25弥补向量检索在精确关键词匹配上的不足
"""

import math
import json
import os
import re
from typing import Protocol
from collections import Counter
from AgentForge.rag.models import Chunk, RetrievalResult


class Embedder(Protocol):
    """
    嵌入器协议 - 定义文本向量化接口

    使用Protocol(结构化子类)而非ABC:
    - 不需要显式继承，任何实现了embed方法的类都自动兼容
    - 利于依赖注入和测试(mock)

    生产环境推荐:
    - OpenAI: text-embedding-3-small (性价比最优)
    - 开源: sentence-transformers/all-MiniLM-L6-v2 (本地方案)
    """

    def embed(self, text: str) -> list[float]:
        """将单条文本转为向量"""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化(通常有API批量优化)"""
        ...


class SimpleEmbedder:
    """
    简单嵌入器 - 纯Python实现的演示用Embedder

    注意: 这只是教学演示，不产生有意义的语义向量！
    实际使用请接入OpenAI/HuggingFace等API。

    原理: 基于字符n-gram的哈希，生成固定维度向量。
    相同词汇会有相同贡献，因此能粗糙反映词汇重叠度。
    """

    def __init__(self, dimensions: int = 128, ngram_range: tuple = (2, 3)):
        self.dimensions = dimensions
        self.ngram_range = ngram_range

    def _tokenize(self, text: str) -> list[str]:
        """提取字符级n-gram作为token"""
        text = text.lower().strip()
        tokens = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                tokens.append(text[i:i+n])
        return tokens

    def embed(self, text: str) -> list[float]:
        """
        将文本哈希到固定维度向量

        算法:
        1. 提取字符n-gram
        2. 每个n-gram哈希到[0, dimensions)的桶
        3. 累加计数并L2归一化
        """
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)

        for token in tokens:
            # 哈希到维度索引
            idx = hash(token) % self.dimensions
            vector[idx] += 1.0

        # L2归一化，使相似度计算与文本长度无关
        norm = math.sqrt(sum(v**2 for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化"""
        return [self.embed(t) for t in texts]


class VectorStore:
    """
    向量存储 - 管理文档向量并提供相似度检索

    核心操作:
    - add: 插入向量
    - search: 最相似检索

    存储结构:
    ┌─────────────┬──────────────────┬───────────────┐
    │ chunk_id    │ embedding         │ chunk_data    │
    ├─────────────┼──────────────────┼───────────────┤
    │ uuid-1      │ [0.12, 0.34, ...]│ {text, meta}  │
    │ uuid-2      │ [0.56, 0.78, ...]│ {text, meta}  │
    └─────────────┴──────────────────┴───────────────┘

    生产环境推荐: Chroma / Qdrant / Weaviate / Milvus
    """

    def __init__(self, persist_path: str | None = None):
        """
        Args:
            persist_path: 持久化目录路径，None表示纯内存存储
        """
        self.persist_path = persist_path
        # 内存存储: {chunk_id: {"embedding": [...], "data": Chunk}}
        self._storage: dict[str, dict] = {}
        # 可选: 持久化到磁盘
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)

    def add(self, chunks: list[Chunk]) -> None:
        """
        将带向量的Chunk存入索引

        Args:
            chunks: 已填充embedding字段的Chunk列表
        """
        for chunk in chunks:
            if not chunk.embedding:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding. Run embedder first.")

            self._storage[chunk.doc_id + "_" + str(chunk.chunk_id)] = {
                "embedding": chunk.embedding,
                "data": chunk
            }

        # 持久化
        if self.persist_path:
            self._persist()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        """
        暴力搜索 - 计算query与所有向量的余弦相似度

        时间复杂度: O(n*d)
        - n: 文档块数量
        - d: 向量维度

        大规模数据需使用ANN索引(FAISS/ScaNN等)，复杂度降为O(log n)

        Args:
            query_embedding: 查询文本的向量
            top_k: 返回最相似的前k个结果

        Returns:
            按相似度降序排列的检索结果
        """
        results = []

        for key, item in self._storage.items():
            # 余弦相似度 = 点积(已归一化向量)
            score = self._cosine_similarity(query_embedding, item["embedding"])
            results.append(RetrievalResult(
                chunk=item["data"],
                score=score,
                retrieval_method="vector"
            ))

        # 按分数降序排列，取top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """
        计算余弦相似度

        cos(θ) = (A·B) / (||A|| * ||B||)

        如果向量已L2归一化，则简化为点积: cos(θ) = A·B
        """
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai**2 for ai in a))
        norm_b = math.sqrt(sum(bi**2 for bi in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def count(self) -> int:
        """返回索引中的文档块总数"""
        return len(self._storage)

    def _persist(self):
        """将索引序列化到磁盘(简化版JSON存储)"""
        if not self.persist_path:
            return
        data = {
            k: {"embedding": v["embedding"], "data": {
                "text": v["data"].text,
                "doc_id": v["data"].doc_id,
                "chunk_id": v["data"].chunk_id,
                "metadata": v["data"].metadata
            }}
            for k, v in self._storage.items()
        }
        with open(os.path.join(self.persist_path, "index.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class BM25Retriever:
    """
    BM25关键词检索器

    BM25是信息检索的经典算法，基于词频(TF)和逆文档频率(IDF):
    score(q,d) = Σ IDF(qi) * [f(qi,d) * (k1+1)] / [f(qi,d) + k1*(1-b+b*|d|/avgdl)]

    为什么需要BM25:
    - 向量检索: 善于语义匹配("车"→"汽车")
    - BM25: 善于精确匹配("Python 3.12"不会变成"Python 3.11")
    - 混合使用互补短板
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 词频饱和系数，控制词频对分数的影响程度
            b: 长度归一化系数，0=不归一化，1=完全归一化
        """
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.avg_doc_len: float = 0.0
        self.idf: dict[str, float] = {}

    def index(self, chunks: list[Chunk]) -> None:
        """构建BM25索引"""
        self.chunks = chunks

        # 分词并统计
        tokenized = [self._tokenize(c.text) for c in chunks]
        doc_freqs = Counter()  # 包含某词的文档数
        total_len = 0

        for tokens in tokenized:
            total_len += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1

        n = len(chunks)
        self.avg_doc_len = total_len / n if n > 0 else 1

        # IDF计算: log((n - df + 0.5) / (df + 0.5) + 1)
        for token, df in doc_freqs.items():
            self.idf[token] = math.log((n - df + 0.5) / (df + 0.5) + 1)

        # 保存分词结果供后续检索使用
        self._tokenized = tokenized

    def _tokenize(self, text: str) -> list[str]:
        """简单分词: 小写+正则提取词"""
        return re.findall(r'\w+', text.lower())

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        BM25检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            按BM25分数降序排列的结果
        """
        query_tokens = self._tokenize(query)
        results = []

        for i, tokens in enumerate(self._tokenized):
            # 词频统计
            tf = Counter(tokens)
            doc_len = len(tokens)

            # BM25打分
            score = 0.0
            for qt in query_tokens:
                if qt in tf:
                    idf = self.idf.get(qt, 0)
                    f = tf[qt]
                    # BM25公式
                    numerator = f * (self.k1 + 1)
                    denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    score += idf * numerator / denominator

            results.append(RetrievalResult(
                chunk=self.chunks[i],
                score=score,
                retrieval_method="bm25"
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class HybridRetriever:
    """
    混合检索引擎 - 融合向量检索和BM25

    融合策略: Reciprocal Rank Fusion (RRF)
    RRF_score(d) = Σ 1 / (k + rank_i(d))
    - k: 平滑常数(通常60)，防止排名靠前文档主导
    - rank_i: 文档d在第i个检索器中的排名

    RRF优势:
    - 不需要分数归一化(不同检索器分数尺度不同)
    - 简单高效，经验表明效果好于线性加权
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """
        混合检索入口

        流程:
        1. 向量检索获取初始结果
        2. BM25检索获取初始结果
        3. RRF融合两个结果列表
        4. 返回融合后的top_k
        """
        # 两种检索各取更多结果，给融合留空间
        fetch_k = top_k * 3

        # 向量检索
        vector_results = self.vector_store.search(query_embedding, fetch_k)

        # BM25检索
        bm25_results = self.bm25.search(query, fetch_k)

        # RRF融合
        return self._reciprocal_rank_fusion(vector_results, bm25_results, top_k)

    def _reciprocal_rank_fusion(
        self,
        list_a: list[RetrievalResult],
        list_b: list[RetrievalResult],
        top_k: int
    ) -> list[RetrievalResult]:
        """
        RRF融合算法

        为每个文档计算融合分数，最终排序返回。
        出现在多个检索器中的文档会获得更高分数。
        """
        # 收集所有唯一chunk的RRF分数
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, result in enumerate(list_a):
            cid = result.chunk.doc_id + "_" + str(result.chunk.chunk_id)
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[cid] = result.chunk

        for rank, result in enumerate(list_b):
            cid = result.chunk.doc_id + "_" + str(result.chunk.chunk_id)
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.rrf_k + rank + 1)
            chunk_map[cid] = result.chunk

        # 按RRF分数排序
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=rrf_scores[cid],
                retrieval_method="hybrid"
            )
            for cid in sorted_ids
        ]
