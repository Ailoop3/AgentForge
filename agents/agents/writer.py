"""
作家Agent - Writer Agent

职责: 内容撰写、结构化输出
- 将分析和发现转化为可读内容
- 按照指定格式组织文档
- 确保逻辑连贯、表达清晰
- 添加引用溯源
"""

from typing import Any
from AgentForge.agents.base import BaseAgent
from AgentForge.agents.state import AgentState


class WriterAgent(BaseAgent):
    """
    作家Agent - 内容撰写、结构化输出

    职责:
    - 将分析和发现转化为可读内容
    - 按照指定格式组织文档
    - 确保逻辑连贯、表达清晰
    - 添加引用溯源

    写作原则:
    - 金字塔原理: 结论先行，分层展开
    - 简洁精准: 一句话能说清的不用两句
    - 引用完整: 每个关键论点都标注来源
    - 可读性强: 适当使用列表、对比、总结
    """

    name = "writer"
    description = "内容创作专家 - 撰写报告、文章、文档"
    system_prompt = """你是内容创作专家。任务:
1. 将分析结果转化为结构清晰、逻辑连贯的文本
2. 按照指定格式组织文档结构
3. 确保专业准确的表达
4. 添加来源引用(溯源)

写作原则:
- 金字塔原理: 结论先行，分层展开
- 简洁精准: 一句话能说清的不用两句
- 引用完整: 每个关键论点都标注来源
- 可读性强: 适当使用列表、对比、总结

输出格式: JSON，包含标题、章节、引用。"""

    def run_core(self, state: AgentState) -> Any:
        """
        核心逻辑: 构骨架 → 填内容 → 润色

        依赖: Researcher + Analyst的输出
        """
        task = state["task"]
        researcher_output = state.get("results", {}).get("researcher", {}).get("output", {})
        analyst_output = state.get("results", {}).get("analyst", {}).get("output", {})

        findings = researcher_output.get("findings", [])
        insights = analyst_output.get("insights", [])
        patterns = analyst_output.get("patterns", [])
        llm_analysis = analyst_output.get("llm_analysis", "")

        prompt = f"""基于以下素材，撰写一份结构化的报告。

任务: {task}

素材:
- 关键发现: {findings[:5] if findings else '无'}
- 分析洞察: {insights if insights else '无'}
- 发现模式: {patterns if patterns else '无'}
- 分析内容: {str(llm_analysis)[:300]}

报告应包含:
1. 摘要(3句话总结核心结论)
2. 详细分析
3. 关键发现
4. 建议/结论

请输出结构化的JSON格式。"""

        response = self.call_llm(prompt)

        return {
            "title": f"《{task[:20]}》分析报告",
            "sections": [
                {"title": "摘要", "content": "核心结论摘要..."},
                {"title": "详细分析", "content": response},
                {"title": "建议", "content": "基于分析的建议..."}
            ],
            "citations": researcher_output.get("sources", []),
            "word_count": len(response),
            "input_from": ["researcher", "analyst"]
        }
