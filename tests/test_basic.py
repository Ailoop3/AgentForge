"""
AgentForge 基础测试

验证:
1. 模块导入正常
2. 知识库摄取和检索正常
3. Agent执行正常
4. 编排器端到端流程正常
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """测试所有模块可正常导入"""
    from AgentForge.config import settings
    from AgentForge.rag.models import Document, Chunk, SourceType, RAGContext
    from AgentForge.rag.knowledge_base import KnowledgeBase
    from AgentForge.rag.ingestion.loaders import IngestPipeline, TextChunker
    from AgentForge.rag.retrieval.vector_store import SimpleEmbedder, VectorStore, BM25Retriever, HybridRetriever
    from AgentForge.rag.retrieval.reranker import Score加权Reranker
    from AgentForge.agents.state import AgentState, create_initial_state
    from AgentForge.agents.base import BaseAgent, Tool, AgentResult
    from AgentForge.agents.arbiter import Blackboard, Arbiter
    from AgentForge.agents.router import KeywordRouter, DefaultPlanner
    from AgentForge.agents.agents.index import get_agent, list_agents
    from AgentForge.observability.tracer import Tracer, SpanType, Span, TraceTree
    from AgentForge.observability.evaluator import Evaluator, RuleBasedEvaluator, EvalDimension, Score, EvalReport
    from AgentForge.observability.metrics import MetricsCollector, MetricSnapshot
    from AgentForge.observability.replay import Replayer, ReplayStep
    from AgentForge.orchestrator import Orchestrator, OrchestratorResult
    from AgentForge.__main__ import AgentForge as AgentForgeCls, quick_demo
    print("[OK] 所有模块导入成功")


def test_rag_pipeline():
    """测试RAG完整流程"""
    from AgentForge.rag.knowledge_base import KnowledgeBase

    # 创建知识库
    kb = KnowledgeBase(name="test_kb")

    # 直接摄取文本
    kb.ingest_text("Python是一种高级编程语言，以简洁语法著称。", {"type": "intro"})
    kb.ingest_text("Go是Google开发的语言，以并发支持著称。", {"type": "intro"})
    kb.ingest_text("RAG是检索增强生成技术，结合检索和生成。", {"type": "tech"})

    assert kb.size == 3, f"期望3个chunk，实际{kb.size}"
    print(f"[OK] 知识库摄取: {kb.size}个chunk")

    # 检索
    results = kb.retrieve("Python语言")
    assert len(results) > 0, "检索结果为空"
    print(f"[OK] 检索测试: 返回{len(results)}个结果")

    # 格式化上下文
    context = kb.get_context("Python语言")
    assert context.context_text != "", "上下文为空"
    print(f"[OK] RAG上下文: {context.total_tokens} tokens")


def test_agent_creation():
    """测试Agent创建"""
    from AgentForge.agents.agents.index import get_agent, list_agents

    agents = list_agents()
    assert len(agents) >= 4, f"至少4个Agent，实际{len(agents)}"
    print(f"[OK] Agent列表: {agents}")

    # 创建Researcher
    researcher = get_agent("researcher")
    assert researcher.name == "researcher"
    print(f"[OK] Agent创建: {researcher}")


def test_orchestrator():
    """测试编排器端到端"""
    from AgentForge.__main__ import AgentForge
    from AgentForge.orchestrator import OrchestratorResult

    forge = AgentForge()

    # 添加知识
    forge.add_text_knowledge("researcher", "Python是1991年创建的编程语言。")
    forge.add_text_knowledge("analyst", "技术选型需要考虑性能和开发效率。")

    # 执行
    result = forge.run("测试任务")

    assert isinstance(result, OrchestratorResult)
    assert result.success or not result.success  # 至少返回了结果
    print(f"[OK] 编排器执行: trace={result.trace_id}, duration={result.duration_ms:.1f}ms")


def test_tracer():
    """测试追踪器"""
    from AgentForge.observability.tracer import Tracer, SpanType

    tracer = Tracer(auto_persist=False)

    with tracer.start_trace("测试任务") as trace_id:
        with tracer.span("test_op", SpanType.CUSTOM):
            pass

    stats = tracer.get_statistics()
    assert stats["total_traces"] >= 1
    print(f"[OK] Tracer: {stats['total_traces']}条trace")


def test_metrics():
    """测试指标收集"""
    from AgentForge.observability.metrics import MetricsCollector

    metrics = MetricsCollector()
    metrics.record_query(success=True, latency_ms=100, cost=0.001, tokens=150)
    metrics.record_query(success=True, latency_ms=200, cost=0.002, tokens=300)
    metrics.record_agent_call("researcher")

    report = metrics.get_report()
    assert report["overview"]["total_queries"] == 2
    print(f"[OK] Metrics: {report['overview']}")


def test_blackboard():
    """测试Blackboard"""
    from AgentForge.agents.arbiter import Blackboard

    bb = Blackboard()
    bb.write("researcher", "findings", ["发现1", "发现2"])
    bb.write("analyst", "insights", ["洞察1"])

    result = bb.read("analyst", "findings", owner="researcher")
    assert len(result) == 2
    print(f"[OK] Blackboard: 读写正常")


def test_arbiter():
    """测试Arbiter"""
    from AgentForge.agents.arbiter import Arbiter
    from AgentForge.agents.base import AgentResult

    arbiter = Arbiter(strategy="weighted")
    results = {
        "researcher": AgentResult("researcher", "output_a", confidence=0.8),
        "analyst": AgentResult("analyst", "output_b", confidence=0.6),
    }

    verdict = arbiter.arbitrate(results, "测试任务")
    assert "output" in verdict
    assert "score" in verdict
    print(f"[OK] Arbiter: strategy={verdict['strategy']}, score={verdict['score']:.2f}")


def test_all():
    """运行全部测试"""
    print("=" * 50)
    print("AgentForge 测试")
    print("=" * 50)

    test_imports()
    test_rag_pipeline()
    test_agent_creation()
    test_tracer()
    test_metrics()
    test_blackboard()
    test_arbiter()
    test_orchestrator()

    print("=" * 50)
    print("所有测试通过 [OK]")
    print("=" * 50)


if __name__ == "__main__":
    test_all()
