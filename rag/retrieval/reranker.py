"""
重排模块 - Reranker

为什么需要重排:
1. 初步检索(向量/BM25)追求召回率，候选集可能不够精准
2. 重排用更精确的模型对候选集重新排序，提升精确率
3. 重排模型通常比检索模型更重(计算量更大)，所以只对top-N做

重排策略对比:
┌─────────────────┬──────────────┬──────────────────┐
│ 重排器          │ 原理          │ 适用场景          │
├─────────────────┼──────────────┼──────────────────┤
│ Cross-Encoder   │ 同时编码query │ 精度要求高        │
│                 │ 和doc判断相关  │                  │
│ Cohere Rerank   │ API服务       │ 快速上线          │
│ LLM Rerank      │ 让LLM判断     │ 复杂语义场景      │
│ 自定义评分      │ 规则+模型混合  │ 特定领域          │
└─────────────────┴──────────────┴──────────────────┘

学习要点:
- 召回(recall) vs 精确(precision)的权衡
- 检索系统的"粗排→精排"漏斗模式
"""

from typing import Protocol, Callable
from AgentForge.rag.models import Chunk, RetrievalResult


class Reranker(Protocol):
    """重排器协议"""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 3
    ) -> list[RetrievalResult]:
        """对检索结果重排序并截断"""
        ...


class Score加权Reranker:
    """
    加权评分重排器 - 结合多维度特征的简单重排

    综合考量:
    - 检索相关性分数 (原始score)
    - 文档新鲜度 (时间衰减)
    - 文档权威性 (source权重)

    最终分数 = α * relevance + β * freshness + γ * authority

    这是一个轻量级重排方案，不需要额外模型。
    适用于对延迟敏感或成本受限的场景。
    """

    def __init__(
        self,
        alpha: float = 0.7,     # 相关性权重
        beta: float = 0.2,      # 新鲜度权重
        gamma: float = 0.1,     # 权威性权重
        source_weights: dict[str, float] | None = None
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "权重之和必须=1"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # 来源可信度权重(可自定义)
        self.source_weights = source_weights or {
            "official_doc": 1.0,
            "blog": 0.7,
            "forum": 0.5,
            "unknown": 0.3
        }

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 3
    ) -> list[RetrievalResult]:
        """
        对候选结果重新评分并排序

        Args:
            query: 原始查询
            results: 检索引擎返回的候选
            top_k: 最终保留数

        Returns:
            重排后的结果
        """
        scored = []
        for result in results:
            relevance = result.score
            freshness = self._calc_freshness(result.chunk.metadata)
            authority = self._calc_authority(result.chunk.metadata)

            final_score = (
                self.alpha * relevance +
                self.beta * freshness +
                self.gamma * authority
            )
            scored.append(RetrievalResult(
                chunk=result.chunk,
                score=final_score,
                retrieval_method=f"{result.retrieval_method}+reranked"
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _calc_freshness(self, metadata: dict) -> float:
        """
        计算新鲜度分数

        使用指数衰减: score = e^(-λ * days_ago)
        最新的文档=1.0，越老越低
        """
        import time
        timestamp = metadata.get("timestamp", 0)
        if timestamp == 0:
            return 0.5  # 未知时间给中等分数

        days_ago = (time.time() - timestamp) / 86400
        return math.exp(-0.01 * days_ago)

    def _calc_authority(self, metadata: dict) -> float:
        """根据来源类型计算权威性"""
        source = metadata.get("source_type", "unknown")
        return self.source_weights.get(source, 0.3)


class LLM重排器:
    """
    LLM重排器 - 使用大模型判断文档与查询的相关性

    原理:
    1. 将(query, doc)对发给LLM
    2. LLM输出相关性分数(0-10)
    3. 按LLM分数排序

    优点: 语义理解能力强，能处理复杂查询
    缺点: 延迟高、token消耗大(只对top-N候选使用)

    这是"粗排→精排"架构中的精排环节。
    """

    def __init__(self, llm_fn: Callable[[str], str]):
        """
        Args:
            llm_fn: 接受prompt字符串，返回模型输出的函数
        """
        self.llm_fn = llm_fn

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 3
    ) -> list[RetrievalResult]:
        """
        用LLM对每个候选文档打分
        """
        scored = []
        for result in results:
            score = self._score_document(query, result.chunk.text)
            scored.append(RetrievalResult(
                chunk=result.chunk,
                score=score,
                retrieval_method=f"{result.retrieval_method}+llm_reranked"
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _score_document(self, query: str, doc: str) -> float:
        """
        构造prompt让LLM打分

        Prompt设计要点:
        - 明确评分标准(0-10)
        - 要求先给出分数再解释(便于解析)
        - 控制输出长度
        """
        prompt = f"""请判断文档与查询的相关性(0-10分)。

查询: {query}

文档: {doc[:500]}...

评分标准:
- 0-3: 基本不相关
- 4-6: 部分相关
- 7-9: 高度相关
- 10: 完美匹配

请只输出一个数字(0-10)，不要解释。"""

        try:
            response = self.llm_fn(prompt).strip()
            # 提取数字
            import re
            match = re.search(r'(\d+)', response)
            return float(match.group(1)) / 10.0 if match else 0.5
        except Exception:
            return 0.5  # 失败时给中等分数


# 引入math模块(上面用了但没import)
import math
