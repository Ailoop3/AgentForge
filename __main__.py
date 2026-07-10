"""
AgentForge - 主入口

整合三大模块的一站式Agent平台。

使用方式:
1. 编程使用:
    from agentforge import AgentForge
    forge = AgentForge(llm_fn=your_llm_function)
    result = forge.run("你的任务")
    print(result.output)

2. 加载知识库:
    forge.add_knowledge("researcher", "docs/", SourceType.MARKDOWN)

3. 查看统计:
    forge.dashboard()  # 启动Streamlit面板

4. 回放trace:
    forge.replay(trace_id)

快速开始:
    python -m agentforge
"""

import os
import sys
from typing import Callable, Any

from .config import settings
from .orchestrator import Orchestrator, OrchestratorResult
from .rag.knowledge_base import KnowledgeBase
from .rag.models import SourceType
from .observability.tracer import Tracer
from .observability.replay import Replayer
from .observability.evaluator import Evaluator
from .observability.metrics import MetricsCollector


class AgentForge:
    """
    AgentForge - 统一的Agent平台接口

    封装了三大模块的所有功能，提供简洁的顶层API。
    用户只需与这个类交互，无需关心内部模块。

    用法示例:
        forge = AgentForge(llm_fn=call_openai)

        # 可选: 为Agent添加知识库
        forge.add_knowledge_base("researcher", "docs/research/")
        forge.add_knowledge_base("analyst", "docs/analysis/")

        # 执行任务
        result = forge.run("分析Python和Go的优劣势")
        print(result.output)
        print(f"Trace ID: {result.trace_id}")
        print(f"总耗时: {result.duration_ms:.0f}ms")
        print(f"综合评分: {result.overall_score:.2f}")

        # 查看Dashboard
        forge.launch_dashboard()
    """

    def __init__(self, llm_fn: Callable[[str, str], str] | None = None):
        """
        初始化AgentForge平台

        Args:
            llm_fn: LLM调用函数，签名为 (system_prompt, user_prompt) -> str
                   推荐使用OpenAI/兼容API:

                   def call_openai(system, user):
                       from openai import OpenAI
                       client = OpenAI()
                       response = client.chat.completions.create(
                           model="gpt-4o-mini",
                           messages=[
                               {"role": "system", "content": system},
                               {"role": "user", "content": user}
                           ]
                       )
                       return response.choices[0].message.content
        """
        # 初始化可观测性组件
        self.tracer = Tracer()
        self.replayer = Replayer(self.tracer)
        self.metrics = MetricsCollector()
        self.evaluator = Evaluator(llm_fn=llm_fn, use_llm=False)

        # 知识库注册表
        self._knowledge_bases: dict[str, KnowledgeBase] = {}

        # 编排器(延迟初始化，等知识库注册完毕)
        self._orchestrator = None
        self.llm_fn = llm_fn

    def add_knowledge_base(
        self,
        agent_name: str,
        path: str,
        source_type: SourceType = SourceType.MARKDOWN
    ) -> 'AgentForge':
        """
        为指定Agent添加知识库

        支持链式调用:
            forge.add_knowledge_base("researcher", "docs/a.md")\\
                 .add_knowledge_base("analyst", "docs/b.md")

        Args:
            agent_name: Agent名称(researcher/analyst/writer/reviewer)
            path: 文档路径(文件or目录)
            source_type: 文档类型

       :
            self(支持链式调用)
        """
        # 创建知识库实例
        kb = KnowledgeBase(name=f"{agent_name}_kb")
        self._knowledge_bases[agent_name] = kb

        # 摄取文档
        if os.path.isdir(path):
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                if os.path.isfile(filepath):
                    try:
                        kb.ingest(filepath, source_type)
                    except Exception as e:
                        print(f" 跳过 {filename}: {e}")
        else:
            kb.ingest(path, source_type)

        print(f"知识库加载: agent={agent_name}, chunks={kb.size}")
        return self

    def add_text_knowledge(self, agent_name: str, text: str, metadata: dict = None) -> 'AgentForge':
        """
        直接为Agent添加文本知识

        用于运行时动态注入知识，如用户提供的上下文。

        Args:
            agent_name: Agent名称
            text: 知识文本
            metadata: 附加元信息
        """
        kb_name = f"{agent_name}_kb"
        if kb_name not in self._knowledge_bases:
            self._knowledge_bases[kb_name] = KnowledgeBase(name=kb_name)

        self._knowledge_bases[kb_name].ingest_text(text, metadata)
        return self

    def run(self, task: str) -> OrchestratorResult:
        """
        执行任务 - 主要入口

        自动初始化编排器(首次调用时)，然后执行完整流程。

        Args:
            task: 任务描述

        Returns:
            OrchestratorResult(含输出/trace/评估/指标)
        """
        if self._orchestrator is None:
            self._init_orchestrator()

        return self._orchestrator.run(task)

    def _init_orchestrator(self):
        """初始化编排器(延迟加载)"""
        self._orchestrator = Orchestrator(
            llm_fn=self.llm_fn,
            tracer=self.tracer,
            evaluator=self.evaluator,
            metrics=self.metrics,
            knowledge_bases=self._knowledge_bases
        )

    def replay(self, trace_id: str, verbose: bool = True):
        """
        回放指定trace

        Args:
            trace_id: trace ID
            verbose: 是否打印详情
        """
        return self.replayer.replay(trace_id, verbose)

    def get_statistics(self) -> dict:
        """获取系统统计信息"""
        return {
            "traces": self.tracer.get_statistics(),
            "metrics": self.metrics.get_report()
        }

    def launch_dashboard(self):
        """
        启动Streamlit Dashboard

        需要在终端运行:
            streamlit run -m agentforge
        """
        try:
            import subprocess
            dashboard_path = os.path.join(
                os.path.dirname(__file__), "observability", "dashboard", "app.py"
            )
            subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
        except ImportError:
            print("请先安装Streamlit: pip install streamlit")
        except Exception as e:
            print(f"Dashboard启动失败: {e}")


def quick_demo():
    """
    快速演示 - 无需外部依赖

    展示AgentForge的基本功能，使用内置的占位LLM。
    """
    print("=" * 60)
    print("AgentForge 快速演示")
    print("=" * 60)

    # 创建实例(使用默认占位LLM)
    forge = AgentForge()

    # 添加一些演示知识
    forge.add_text_knowledge("researcher", """
    Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
    它以简洁、易读的语法著称，广泛应用于数据科学、AI、Web开发等领域。
    Python 拥有丰富的标准库和第三方生态系统（如NumPy、Pandas、PyTorch）。

    Go（又称Golang）是Google开发的静态类型编程语言，于2009年发布。
    它以并发支持（goroutine）、编译速度快、部署简单而受到欢迎。
    特别适合云服务、微服务、网络编程等领域。
    """)

    forge.add_text_knowledge("analyst", """
    技术选型需要考虑以下维度:
    1. 性能: Go在并发和计算性能上优于Python
    2. 开发效率: Python开发速度更快，代码更简洁
    3. 生态系统: Python在AI/数据领域生态更丰富
    4. 学习曲线: Python更易入门，Go的并发模型需要额外学习
    5. 部署: Go编译为静态二进制，部署更简单
    """)

    # 执行任务
    task = "分析Python和Go的优劣势，给出技术选型建议"
    print(f"\n任务: {task}\n")

    result = forge.run(task)

    # 展示结果
    print(f"\n{'='*60}")
    print(f"执行结果:")
    print(f"{'='*60}")
    print(f"成功: {result.success}")
    print(f"耗时: {result.duration_ms:.1f}ms")
    print(f"综合评分: {result.overall_score:.2f}")
    print(f"Trace ID: {result.trace_id}")
    print(f"\n输出: {result.output}")
    print(f"\nAgent执行详情:")
    for agent, detail in result.agent_results.items():
        status = "[OK]" if detail.get("success") else "[FAIL]"
        print(f"  {status} {agent}: {detail.get('duration_ms', 0):.1f}ms")

    # 查看统计
    stats = forge.get_statistics()
    print(f"\n系统统计: {stats['traces']}")

    return result


# CLI入口
if __name__ == "__main__":
    quick_demo()
