"""
知识库统一接口 - Knowledge Base

整合RAG模块的所有组件，对外提供统一的高层API。
每个Agent持有一个KnowledgeBase实例，实现知识隔离。

架构:
                  KnowledgeBase (统一接口)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   IngestPipeline  HybridRetriever  Reranker
        │              │
        ▼              ▼
   Chunk Embedding  VectorStore + BM25

学习要点:
1. 外观模式(Facade): 隐藏子系统复杂性，提供简洁接口
2. 依赖注入: 所有依赖通过构造函数注入，便于测试和替换
3. 延迟初始化: Embedder和Store可在首次使用时创建
"""

import os
from typing import Callable
from AgentForge.config import settings
from AgentForge.rag.models import Chunk, RAGContext, SourceType, RetrievalResult
from AgentForge.rag.ingestion.loaders import IngestPipeline
from AgentForge.rag.retrieval.vector_store import (
    Embedder, SimpleEmbedder, VectorStore, BM25Retriever, HybridRetriever
)
from AgentForge.rag.retrieval.reranker import Reranker, Score加权Reranker


class KnowledgeBase:
    """
    知识库 - RAG Engine的统一对外接口

    职责:
    1. ingest(): 摄取文档到知识库
    2. retrieve(): 检索与查询相关的知识
    3. get_context(): 获取格式化的RAG上下文(直接用于prompt)

    每个Agent实例化自己的KnowledgeBase，确保:
    - 知识隔离(不同Agent可访问不同文档集)
    - 独立检索策略(不同Agent可配置不同参数)

    用法示例:
        kb = KnowledgeBase(name="research_kb")
        kb.ingest("docs/paper.pdf", SourceType.PDF)
        context = kb.get_context("什么是多智能体系统?")
        # 将context.context_text拼入Agent的prompt
    """

    def __init__(
        self,
        name: str,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        top_k: int = None,
        persist_dir: str = None
    ):
        """
        Args:
            name: 知识库名称(用于持久化目录和日志标识)
            embedder: 向量化器，None则使用SimpleEmbedder(演示用)
            reranker: 重排器，None则使用Score加权Reranker
            chunk_size: 分块大小，None则取全局配置
            chunk_overlap: 块重叠量，None则取全局配置
            top_k: 检索返回数，None则取全局配置
            persist_dir: 持久化根目录
        """
        self.name = name

        # 使用配置默认值或自定义值
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.top_k = top_k or settings.RETRIEVAL_TOP_K

        # 初始化子组件
        self.pipeline = IngestPipeline(self.chunk_size, self.chunk_overlap)
        self.embedder = embedder or SimpleEmbedder()

        # 向量存储 - 每个知识库独立目录
        store_path = os.path.join(
            persist_dir or settings.VECTOR_DB_PATH,
            name
        )
        self.vector_store = VectorStore(persist_path=store_path)
        self.bm25 = BM25Retriever()

        # 混合检索引擎
        self.retriever = HybridRetriever(self.vector_store, self.bm25)

        # 重排器
        self.reranker = reranker or Score加权Reranker()

        # 持有所有chunk(供BM25索引和运行时查询)
        self._all_chunks: list[Chunk] = []
        self._is_indexed = False

    def ingest(self, path: str, source_type: SourceType) -> int:
        """
        摄取单个文档

        流程:
        1. 加载文档 → 分块
        2. 对每个块生成embedding
        3. 存储到向量库
        4. 重建BM25索引

        Args:
            path: 文件路径或URL
            source_type: 来源类型

        Returns:
            摄入的chunk数量
        """
        # 分块
        chunks = self.pipeline.ingest(path, source_type)

        # 向量化(批量)
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 存入向量库
        self.vector_store.add(chunks)

        # 暂存chunk，统一重建BM25索引
        self._all_chunks.extend(chunks)
        self._rebuild_bm25()

        return len(chunks)

    def ingest_text(self, text: str, metadata: dict = None) -> int:
        """
        直接摄取文本(无需文件)

        用于运行时动态添加知识，如工具返回结果、用户提供的上下文等。

        Args:
            text: 原始文本
            metadata: 附加元信息

        Returns:
            摄入的chunk数量
        """
        chunk = Chunk(
            text=text,
            doc_id=f"runtime_{len(self._all_chunks)}",
            chunk_id=0,
            metadata=metadata or {}
        )

        # 向量化
        chunk.embedding = self.embedder.embed(chunk.text)

        # 存储
        self.vector_store.add([chunk])
        self._all_chunks.append(chunk)
        self._rebuild_bm25()

        return 1

    def _rebuild_bm25(self):
        """重建BM25索引(增量场景可优化为增量更新)"""
        self.bm25.index(self._all_chunks)
        self._is_indexed = True

    def retrieve(self, query: str, top_k: int = None) -> list[RetrievalResult]:
        """
        检索与查询相关的文档

        流程:
        1. Query向量化
        2. 混合检索(向量+BM25)
        3. 重排

        Args:
            query: 用户查询
            top_k: 返回数量(None则用知识库默认值)

        Returns:
            检索结果列表
        """
        k = top_k or self.top_k

        # Query向量化
        query_embedding = self.embedder.embed(query)

        # 混合检索
        results = self.retriever.search(query, query_embedding, top_k=k * 2)

        # 重排(从2k中精选k个)
        results = self.reranker.rerank(query, results, top_k=k)

        return results

    def get_context(self, query: str, top_k: int = None) -> RAGContext:
        """
        获取格式化的RAG上下文

        这是Agent调用时最常用的方法，返回的context_text可直接拼入prompt。

        格式:
        <context>
        <chunk_1>
        [来源: doc_name] 文本内容...
        </chunk_1>
        <chunk_2>
        [来源: doc_name] 文本内容...
        </chunk_2>
        </context>

        Args:
            query: 查询文本
            top_k: 返回文档数

        Returns:
            RAGContext对象，包含格式化的上下文
        """
        results = self.retrieve(query, top_k)

        # 格式化为LLM可读的文本
        parts = []
        for i, result in enumerate(results):
            source = result.chunk.metadata.get("path", result.chunk.doc_id[:8])
            part = f"<chunk_{i+1}>\n[来源: {source}]\n{result.chunk.text}\n</chunk_{i+1}>"
            parts.append(part)

        context_text = "\n\n".join(parts)

        return RAGContext(
            query=query,
            chunks=[r.chunk for r in results],
            context_text=context_text,
            total_tokens=len(context_text) // 4  # 粗略token估算
        )

    @property
    def size(self) -> int:
        """知识库中的文档块总数"""
        return len(self._all_chunks)

    def __repr__(self):
        return f"KnowledgeBase(name='{self.name}', chunks={self.size})"
