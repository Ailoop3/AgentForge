"""
研究员Agent - Researcher Agent

职责: 信息检索与知识收集
- 从知识库检索相关信息
- 从外部来源搜集补充信息
- 整理和总结发现
"""

from typing import Any
from AgentForge.agents.base import BaseAgent
from AgentForge.agents.state import AgentState


class ResearcherAgent(BaseAgent):
    """
    研究员Agent - 信息检索与知识收集

    职责:
    - 从知识库检索相关信息
    - 从外部来源(Web/文档)搜集补充信息
    - 整理和总结发现

    工具:
    - search_kb: 知识库检索
    - web_search: Web搜索(扩展点)
    - fetch_document: 获取文档详情
    """

    name = "researcher"
    description = "信息检索专家 - 从知识库和外部来源搜集、整理信息"
    system_prompt = """你是信息检索专家。任务:
1. 从给定信息源中检索与问题相关的知识
2. 筛选最相关的信息
3. 整理成结构化的发现列表
4. 标注信息来源
5. 识别未覆盖的信息缺口

输出格式: JSON，包含findings(发现列表)、sources(来源)、gaps(信息缺口)
每个finding包含: content(内容)、relevance(相关度0-10)、source(来源)"""

    def run_core(self, state: AgentState) -> Any:
        """
        核心逻辑: 检索 → 筛选 → 整理

        步骤:
        1. 从知识库检索相关内容
        2. 对结果进行相关性评分
        3. 整理为结构化输出
        """
        task = state["task"]

        # Step 1: 知识库检索
        kb_findings = []
        context_text = ""

        if self.knowledge_base and self.knowledge_base.size > 0:
            context = self.knowledge_base.get_context(task)
            context_text = context.context_text
            kb_findings = [
                {
                    "content": chunk.text[:200],
                    "relevance": 7,
                    "source": chunk.metadata.get("path", "知识库")
                }
                for chunk in context.chunks
            ]

        # Step 2: 构造prompt让LLM整理
        prompt = f"""基于以下检索结果，整理关键发现。

任务: {task}

检索结果:
{context_text if context_text else "无知识库数据"}

请先思考需要从哪些角度分析，然后输出结构化JSON。"""

        thinking = [
            f"1. 分析任务关键词: {task[:20]}...",
            f"2. 从知识库检索到 {len(kb_findings)} 条相关信息",
            "3. 开始整理和分类..."
        ]

        response = self.call_llm(prompt)

        return {
            "findings": kb_findings,
            "sources": [f["source"] for f in kb_findings],
            "gaps": [],
            "llm_summary": response,
            "thinking": thinking
        }
