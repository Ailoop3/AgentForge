"""
文档摄取模块 - Ingestion Pipeline

负责将原始文档转换为可索引的Chunk。
核心流程: Source → Loader → Chunker → Chunk[]

学习要点:
1. 不同数据源需要不同的Loader
2. 分块策略直接影响后续检索质量
3. 合理的overlap保持语义连贯性
"""

import uuid
from typing import Callable
from AgentForge.rag.models import Document, Chunk, SourceType


class DocumentLoader:
    """
    文档加载器 - 从各种来源读取原始文本

    支持:
    - 文本文件直接读取
    - 可扩展: PDF解析、Web爬取、Markdown解析

    生产级方案可使用 LangChain DocumentLoaders 或 LlamaIndex Readers
    """

    def load(self, path: str, source_type: SourceType, **kwargs) -> Document:
        """
        加载单个文档

        Args:
            path: 文件路径或URL
            source_type: 来源类型
            **kwargs: 额外参数(如编码、认证信息等)

        Returns:
            Document对象
        """
        if source_type == SourceType.TEXT or source_type == SourceType.MARKDOWN:
            # 文本/Markdown文件直接读取
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # PDF/Web等需要专门的解析器
            # 这里展示扩展点，实际可接入PyPDF2/BeautifulSoup
            raise NotImplementedError(
                f"Source type {source_type} requires additional dependencies. "
                f"Install: pip install pypdf2 beautifulsoup4"
            )

        # 生成唯一ID并包装为Document
        return Document(
            content=content,
            source=source_type,
            metadata={"path": path, "size": len(content)},
            doc_id=str(uuid.uuid4())
        )

    def load_batch(self, paths: list[str], source_type: SourceType) -> list[Document]:
        """批量加载多个文档"""
        return [self.load(p, source_type) for p in paths]


class TextChunker:
    """
    文本分块器 - 将长文档切分为可管理的小块

    分块策略对比:
    ┌─────────────────┬──────────────┬────────────────────────┐
    │ 策略            │ 优点          │ 缺点                   │
    ├─────────────────┼──────────────┼────────────────────────┤
    │ 固定长度        │ 简单快速      │ 可能切断语义           │
    │ 递归分割        │ 保持段落完整  │ 块大小不均匀           │
    │ 语义分块        │ 语义完整      │ 计算成本高             │
    │ 文档结构感知    │ 保留文档结构  │ 需要特定解析器          │
    └─────────────────┴──────────────┴────────────────────────┘

    这里实现递归分割策略(最常用)，按段落→句子→词的优先级切分。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None
    ):
        """
        Args:
            chunk_size: 目标块大小(字符数，粗略估算token)
            chunk_overlap: 相邻块的重叠量，保持上下文衔接
            separators: 分隔符优先级列表，默认按段落→换行→句号→空格
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 分隔符优先级: 先按段落切，段落太长按行切，行太长按句号切...
        self.separators = separators or ["\n\n", "\n", "。", ". ", " "]

    def chunk(self, document: Document) -> list[Chunk]:
        """
        将单个文档分块

        算法:
        1. 按最高优先级分隔符切分
        2. 如果片段仍超过chunk_size，用下一个分隔符继续切
        3. 合并小片段到chunk_size上限
        4. 添加overlap保持连贯性

        Args:
            document: 输入文档

        Returns:
            文档块列表
        """
        text = document.content
        chunks = []

        # 递归切分主逻辑
        raw_chunks = self._recursive_split(text, self.separators)

        # 添加overlap
        for i, chunk_text in enumerate(raw_chunks):
            # 如果不是第一个块，向前取overlap长度的文本
            if i > 0 and self.chunk_overlap > 0:
                overlap_text = raw_chunks[i-1][-self.chunk_overlap:]
                chunk_text = overlap_text + chunk_text

            chunks.append(Chunk(
                text=chunk_text,
                doc_id=document.doc_id,
                chunk_id=i,
                metadata={
                    **document.metadata,
                    "char_count": len(chunk_text),
                    "has_overlap": i > 0
                }
            ))

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """递归按分隔符切分文本"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 尝试当前分隔符
        sep = separators[0] if separators else " "
        parts = text.split(sep)

        # 合并小片段
        result = []
        current = ""
        for part in parts:
            if len(current) + len(part) + len(sep) <= self.chunk_size:
                current += (sep + part) if current else part
            else:
                if current:
                    result.append(current)
                # 如果单个part超过限制，用更细的分隔符继续切
                if len(part) > self.chunk_size and len(separators) > 1:
                    result.extend(self._recursive_split(part, separators[1:]))
                else:
                    current = part

        if current:
            result.append(current)

        return result


class IngestPipeline:
    """
    文档摄取流水线 - 组合Loader和Chunker

    对外提供统一的摄取接口，内部协调加载和分块步骤。
    这是典型的Pipeline模式：将多步操作封装为一个简单接口。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)

    def ingest(self, path: str, source_type: SourceType) -> list[Chunk]:
        """
        文档摄取入口: 加载 → 分块

        Args:
            path: 文件路径/URL
            source_type: 来源类型

        Returns:
            文档块列表

        用法:
            pipeline = IngestPipeline(chunk_size=512, chunk_overlap=64)
            chunks = pipeline.ingest("docs/guide.pdf", SourceType.PDF)
        """
        # Step 1: 加载原始文档
        document = self.loader.load(path, source_type)

        # Step 2: 切分为块
        chunks = self.chunker.chunk(document)

        return chunks

    def ingest_batch(self, paths: list[str], source_type: SourceType) -> list[Chunk]:
        """批量摄取多个文档，返回所有块"""
        all_chunks = []
        for path in paths:
            all_chunks.extend(self.ingest(path, source_type))
        return all_chunks
