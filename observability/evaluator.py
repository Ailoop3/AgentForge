"""
评估器 - Evaluator

对Agent系统进行多维度评估:
1. 正确性 (Correctness) - 输出是否正确
2. 忠实度 (Faithfulness) - 是否有幻觉(输出是否有检索依据)
3. 延迟 (Latency) - 响应速度
4. 成本 (Cost) - API调用费用
5. 路径效率 (Path Efficiency) - 执行路径是否最优

评估框架对比:
┌─────────────────┬────────────────────┬────────────────────┐
│ 框架             │ 特点                │ 适用场景           │
├─────────────────┼────────────────────┼────────────────────┤
│ RAGAS           │ RAG专用评估         │ RAG系统            │
│ TruLens         │ 反馈驱动            │ 可观测性集成       │
│ LLM-as-Judge    │ 灵活但贵            │ 通用评估           │
│ 规则+启发式      │ 快速低成本          │ 初步筛选           │
└─────────────────┴────────────────────┴────────────────────┘

本项目采用混合策略:
- 自动指标: 延迟/成本/token(无需人工)
- 规则评估: 引用完整性/格式正确性
- LLM-as-Judge: 正确性/忠实度(按需调用)
"""

import time
import json
from typing import Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from AgentForge.observability.tracer import Tracer, TraceTree


class EvalDimension(Enum):
    """评估维度枚举"""
    CORRECTNESS = "correctness"        # 正确性
    FAITHFULNESS = "faithfulness"      # 忠实度(无幻觉)
    COMPLETENESS = "completeness"      # 完整性
    LATENCY = "latency"                # 延迟
    COST = "cost"                      # 成本
    PATH_EFFICIENCY = "path_efficiency"  # 路径效率


@dataclass
class Score:
    """
    单项评估分数

    每个维度给出分数(0-1)和解释。
    confidence表示评估结果的置信度(规则评估=高, LLM评估=中)
    """
    dimension: EvalDimension
    score: float               # 0-1, 1为最佳
    confidence: float = 1.0    # 评估置信度
    reasoning: str = ""        # 评分理由
    details: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """
    完整评估报告

    汇总所有维度的评分，给出综合评价。
    """
    trace_id: str
    task: str
    scores: list[Score] = field(default_factory=list)
    overall_score: float = 0.0
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    evaluated_at: str = ""

    def get_score(self, dimension: EvalDimension) -> Score | None:
        """获取指定维度的分数"""
        return next((s for s in self.scores if s.dimension == dimension), None)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "task": self.task,
            "overall_score": self.overall_score,
            "scores": [
                {
                    "dimension": s.dimension.value,
                    "score": s.score,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning
                }
                for s in self.scores
            ],
            "summary": self.summary,
            "recommendations": self.recommendations
        }


class RuleBasedEvaluator:
    """
    规则评估器 - 基于启发式规则的快速评估

    不需要LLM调用，速度快、成本低。
    适合:
    - 批量预筛选
    - 延迟/成本等客观指标
    - 格式正确性检查
    """

    def evaluate_latency(self, trace: TraceTree) -> Score:
        """
        延迟评估

        评分标准(基于总耗时):
        - <3s: 优秀 (1.0)
        - 3-10s: 良好 (0.8)
        - 10-30s: 一般 (0.5)
        - >30s: 差 (0.3)
        """
        latency_ms = trace.total_latency_ms
        if latency_ms < 3000:
            score, reason = 1.0, f"延迟优秀: {latency_ms:.0f}ms"
        elif latency_ms < 10000:
            score, reason = 0.8, f"延迟良好: {latency_ms:.0f}ms"
        elif latency_ms < 30000:
            score, reason = 0.5, f"延迟一般: {latency_ms:.0f}ms"
        else:
            score, reason = 0.3, f"延迟较高: {latency_ms:.0f}ms"

        return Score(
            dimension=EvalDimension.LATENCY,
            score=score,
            confidence=0.95,
            reasoning=reason,
            details={"latency_ms": latency_ms}
        )

    def evaluate_cost(self, trace: TraceTree) -> Score:
        """
        成本评估

        评分标准(基于单次查询成本):
        - <$0.01: 优秀 (1.0)
        - $0.01-$0.05: 良好 (0.8)
        - $0.05-$0.1: 一般 (0.5)
        - >$0.1: 差 (0.3)
        """
        cost = trace.total_cost
        if cost < 0.01:
            score, reason = 1.0, f"成本极低: ${cost:.4f}"
        elif cost < 0.05:
            score, reason = 0.8, f"成本合理: ${cost:.4f}"
        elif cost < 0.1:
            score, reason = 0.5, f"成本偏高: ${cost:.4f}"
        else:
            score, reason = 0.3, f"成本过高: ${cost:.4f}"

        return Score(
            dimension=EvalDimension.COST,
            score=score,
            confidence=0.95,
            reasoning=reason,
            details={"total_cost": cost}
        )

    def evaluate_path_efficiency(self, trace: TraceTree) -> Score:
        """
        路径效率评估

        评估Agent执行的路径是否最优:
        - 是否有冗余步骤
        - 是否走捷径跳过了必要环节
        - 重试次数是否过多

        指标: 实际步骤数 / 理论最优步骤数
        """
        # 获取所有LLM call span
        all_spans = trace.find_slowest_spans(100)
        llm_calls = [s for s in all_spans if s.span_type.value == "llm"]
        total_llm = len(llm_calls)

        # 简单启发: 每个Agent一次LLM调用为最优
        # 实际最优 = Agent数量, 重试会增加调用数
        optimal_calls = 4  # 假设4个Agent
        ratio = optimal_calls / max(total_llm, optimal_calls)

        score = min(ratio, 1.0)
        return Score(
            dimension=EvalDimension.PATH_EFFICIENCY,
            score=score,
            confidence=0.7,
            reasoning=f"实际LLM调用{total_llm}次，理论最优{optimal_calls}次",
            details={"actual_calls": total_llm, "optimal_calls": optimal_calls}
        )

    def evaluate_completeness(self, output: Any, task: str) -> Score:
        """
        完整性评估(基于输出结构)

        检查输出是否包含预期字段:
        - 调研类: 需要有findings/sources
        - 报告类: 需要有sections/title
        - 分析类: 需要有insights/comparisons
        """
        if not output or not isinstance(output, dict):
            return Score(
                dimension=EvalDimension.COMPLETENESS,
                score=0.2,
                confidence=0.8,
                reasoning="输出格式不正确，无法解析"
            )

        # 检查常见必需字段
        expected_keys = ["findings", "insights", "sections", "title", "output"]
        found_keys = [k for k in expected_keys if k in output]
        coverage = len(found_keys) / len(expected_keys)

        return Score(
            dimension=EvalDimension.COMPLETENESS,
            score=coverage,
            confidence=0.7,
            reasoning=f"输出包含{len(found_keys)}个预期字段: {found_keys}",
            details={"found_keys": found_keys, "coverage": coverage}
        )


class LLMEvaluator:
    """
    LLM评估器 - 用大模型作为Judge

    优势: 能理解语义，评估主观质量
    劣势: 有一定成本和延迟，自身也有偏差

    设计要点:
    1. 明确的评分标准(reduces subjectivity)
    2. 结构化输出(parsable)
    3. 少量示例(可选, improves consistency)
    """

    JUDGE_PROMPT = """评估以下Agent输出质量。

原始任务: {task}

Agent输出:
{output}

请从以下维度评分(0-10):
1. 正确性: 信息是否准确、逻辑是否正确
2. 忠实度: 是否有事实依据、是否有幻觉
3. 完整性: 是否覆盖任务所有要求
4. 清晰度: 表达是否清楚、结构是否合理

输出JSON格式:
{{
    "correctness": {{"score": 0-10, "reason": "理由"}},
    "faithfulness": {{"score": 0-10, "reason": "理由"}},
    "completeness": {{"score": 0-10, "reason": "理由"}},
    "clarity": {{"score": 0-10, "reason": "理由"}}
}}
只输出JSON。"""

    def __init__(self, llm_fn: Callable[[str, str], str]):
        self.llm_fn = llm_fn

    def evaluate_correctness(self, output: Any, task: str) -> Score:
        """LLM评判正确性"""
        return self._judge(output, task, EvalDimension.CORRECTNESS)

    def evaluate_faithfulness(self, output: Any, context: str, task: str) -> Score:
        """
        评估忠实度(幻觉检测)

        原理: 检查输出中的事实是否能在检索到的context中找到依据
        如果有context中没有的信息，可能是幻觉。

        Args:
            output: Agent输出
            context: 检索到的参考文档
            task: 原始任务
        """
        prompt = f"""检查以下输出是否完全基于提供的参考来源。

参考来源:
{context[:1000]}

Agent输出:
{str(output)[:500]}

是否有参考来源中未提及的信息(幻觉)?
输出JSON: {{"faithful": true/false, "hallucinations": ["..."], "reason": "理由"}}"""

        try:
            response = self.llm_fn("你是事实核查专家。", prompt)
            json_str = response[response.find("{"):response.find("}")+1]
            result = json.loads(json_str)

            is_faithful = result.get("faithful", True)
            hallucinations = result.get("hallucinations", [])

            return Score(
                dimension=EvalDimension.FAITHFULNESS,
                score=1.0 if is_faithful else max(0, 1.0 - len(hallucinations) * 0.3),
                confidence=0.7,
                reasoning=result.get("reason", "无异常"),
                details={"hallucinations": hallucinations}
            )
        except Exception:
            return Score(
                dimension=EvalDimension.FAITHFULNESS,
                score=0.5,
                confidence=0.3,
                reasoning="评估失败，无法判断"
            )

    def _judge(self, output: Any, task: str, dimension: EvalDimension) -> Score:
        """通用LLM评判"""
        prompt = self.JUDGE_PROMPT.format(
            task=task,
            output=str(output)[:1000]
        )
        try:
            response = self.llm_fn("你是一个严格的评估专家。", prompt)
            json_str = response[response.find("{"):response.find("}")+1]
            result = json.loads(json_str)

            dim_key = dimension.value
            if dim_key in result:
                item = result[dim_key]
                return Score(
                    dimension=dimension,
                    score=item.get("score", 5) / 10,
                    confidence=0.7,
                    reasoning=item.get("reason", ""),
                    details=result
                )
        except Exception:
            pass

        return Score(
            dimension=dimension,
            score=0.5,
            confidence=0.3,
            reasoning="评估失败"
        )


class Evaluator:
    """
    统一评估器 - 组合规则评估和LLM评估

    策略:
    1. 客观指标(延迟/成本)始终用规则评估(快且准)
    2. 主观指标(正确性/忠实度)可选LLM评估
    3. 支持按需评估(只评估关心的维度)
    """

    def __init__(
        self,
        llm_fn: Callable | None = None,
        use_llm: bool = False,
        dimensions: list[EvalDimension] | None = None
    ):
        """
        Args:
            llm_fn: LLM函数(用于LLM-as-Judge)
            use_llm: 是否启用LLM评估
            dimensions: 评估维度(None表示全部)
        """
        self.rule_evaluator = RuleBasedEvaluator()
        self.llm_evaluator = LLMEvaluator(llm_fn) if llm_fn and use_llm else None
        self.dimensions = dimensions or list(EvalDimension)

    def evaluate(
        self,
        trace: TraceTree,
        agent_output: Any = None,
        retrieved_context: str = ""
    ) -> EvalReport:
        """
        执行完整评估

        Args:
            trace: 执行链路
            agent_output: 最终Agent输出(用于质量评估)
            retrieved_context: 检索到的上下文(用于忠实度评估)

        Returns:
            完整评估报告
        """
        scores = []

        # 规则评估(始终执行)
        if EvalDimension.LATENCY in self.dimensions:
            scores.append(self.rule_evaluator.evaluate_latency(trace))

        if EvalDimension.COST in self.dimensions:
            scores.append(self.rule_evaluator.evaluate_cost(trace))

        if EvalDimension.PATH_EFFICIENCY in self.dimensions:
            scores.append(self.rule_evaluator.evaluate_path_efficiency(trace))

        if agent_output and EvalDimension.COMPLETENESS in self.dimensions:
            scores.append(self.rule_evaluator.evaluate_completeness(agent_output, trace.task))

        # LLM评估(可选)
        if self.llm_evaluator and agent_output:
            if EvalDimension.CORRECTNESS in self.dimensions:
                scores.append(
                    self.llm_evaluator.evaluate_correctness(agent_output, trace.task)
                )

            if EvalDimension.FAITHFULNESS in self.dimensions and retrieved_context:
                scores.append(
                    self.llm_evaluator.evaluate_faithfulness(
                        agent_output, retrieved_context, trace.task
                    )
                )

        # 计算综合分数(加权平均)
        weights = {
            EvalDimension.CORRECTNESS: 0.3,
            EvalDimension.FAITHFULNESS: 0.25,
            EvalDimension.LATENCY: 0.15,
            EvalDimension.COST: 0.1,
            EvalDimension.PATH_EFFICIENCY: 0.1,
            EvalDimension.COMPLETENESS: 0.1,
        }

        weighted_sum = 0
        weight_total = 0
        for s in scores:
            w = weights.get(s.dimension, 0.1)
            weighted_sum += s.score * w
            weight_total += w

        overall = weighted_sum / weight_total if weight_total > 0 else 0

        # 生成建议
        recommendations = self._generate_recommendations(scores)

        return EvalReport(
            trace_id=trace.trace_id,
            task=trace.task,
            scores=scores,
            overall_score=overall,
            summary=self._generate_summary(overall, scores),
            recommendations=recommendations,
            evaluated_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def _generate_recommendations(self, scores: list[Score]) -> list[str]:
        """基于低分项给出改进建议"""
        recs = []
        for s in scores:
            if s.score < 0.5:
                if s.dimension == EvalDimension.LATENCY:
                    recs.append("延迟过高: 考虑缓存、并行化、或减少Agent数量")
                elif s.dimension == EvalDimension.COST:
                    recs.append("成本过高: 考虑使用小模型或减少LLM调用次数")
                elif s.dimension == EvalDimension.FAITHFULNESS:
                    recs.append("存在幻觉: 加强RAG检索质量，增加引用验证")
                elif s.dimension == EvalDimension.CORRECTNESS:
                    recs.append("正确性不足: 检查Agent的system prompt和推理逻辑")
        return recs

    def _generate_summary(self, overall: float, scores: list[Score]) -> str:
        """生成评估摘要"""
        if overall >= 0.8:
            grade = "优秀"
        elif overall >= 0.6:
            grade = "良好"
        elif overall >= 0.4:
            grade = "一般"
        else:
            grade = "需改进"

        lowest = min(scores, key=lambda s: s.score) if scores else None
        return f"综合评分: {overall:.2f} ({grade})。最低分项: {lowest.dimension.value if lowest else 'N/A'}"
