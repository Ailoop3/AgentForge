"""
RAG模块的数据模型

定义RAG流程中各阶段的输入输出数据结构。
使用Python dataclass，轻量且自带__repr__，便于调试。
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class SourceType(Enum):
    """支持的文档来源类型"""
    PDF = "pdf"
    MARKDOWN = "md"
    WEB = "web"
    TEXT = "text"


@dataclass
class Document:
    """
    原始文档 - 从数据源加载后的中间表示

    Attributes:
        content: 文档全文内容
        source: 来源类型(pdf/web/md/text)
        metadata: 元信息(文件名、URL、作者、时间等)
        doc_id: 文档唯一标识(UUID)
    """
    content: str
    source: SourceType
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


@dataclass
class Chunk:
    """
    文档分块 - 检索的最小单元

    为什么需要分块:
    - LLM上下文窗口有限，不能塞入整篇文档
    - 细粒度分块提高检索精度
    - 每个块独立嵌入，便于精准匹配

    Attributes:
        text: 块的文本内容
        doc_id: 所属文档ID
        chunk_id: 块在文档中的序号
        metadata: 继承自文档的元信息 + 块级信息
        embedding: 文本的向量表示(后续由Embedder填充)
    """
    text: str
    doc_id: str
    chunk_id: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """
    检索结果 - 包含评分和来源信息

    Attributes:
        chunk: 匹配的文本块
        score: 相关性分数(越高越相关)
        retrieval_method: 来自哪种检索方式(vector/bm25/hybrid)
    """
    chunk: Chunk
    score: float
    retrieval_method: str = "vector"


@dataclass
class RAGContext:
    """
    RAG上下文 - 传递给LLM的最终结果

    包含检索到的文档片段和格式化后的prompt上下文。
    这是RAG Engine对Agent的统一输出格式。

    Attributes:
        query: 原始查询
        chunks: 检索并重排后的文档块
        context_text: 格式化后的上下文文本(直接拼入prompt)
        total_tokens: 上下文总token数(用于判断是否超出窗口)
    """
    query: str
    chunks: list[Chunk]
    context_text: str = ""
    total_tokens: int = 0
