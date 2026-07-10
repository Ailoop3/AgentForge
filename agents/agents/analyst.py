"""
分析师Agent - Analyst Agent

职责: 数据分析、对比评估
- 分析Researcher提供的原始信息
- 进行对比分析(横向/纵向)
- 发现模式和趋势
- 给出关键洞察
"""

from typing import Any
from AgentForge.agents.base import BaseAgent
from AgentForge.agents.state import AgentState


class AnalystAgent(BaseAgent):
    """
    分析师Agent - 数据分析、对比评估

    职责:
    - 分析Researcher提供的原始信息
    - 进行对比分析(横向/纵向)
    - 发现模式和趋势
    - 给出关键洞察

    分析方法:
    - 横向对比: 同类事物在同一维度的比较
    - 纵向对比: 同一事物在不同时间的变化
    - 归因分析: 探究现象背后的原因
    - 趋势预测: 基于历史推断未来
    """

    name = "analyst"
    description = "数据分析专家 - 对比分析、模式发现、洞察提取"
    system_prompt = """你是数据分析专家。任务:
1. 分析提供的原始信息和数据
2. 进行多维度对比(优劣、趋势、风险)
3. 发现数据中的模式和异常
4. 给出可操作的洞察和建议

分析方法:
- 横向对比: 同类事物在同一维度的比较
- 纵向对比: 同一事物在不同时间的变化
- 归因分析: 探究现象背后的原因
- 趋势预测: 基于历史推断未来

输出格式: JSON，包含洞察、对比、模式、风险。"""

    def run_core(self, state: AgentState) -> Any:
        """
        核心逻辑: 对比 → 归因 → 洞察

        依赖: Researcher的输出(results["researcher"])
        """
        task = state["task"]
        researcher_output = state.get("results", {}).get("researcher", {}).get("output", {})

        findings = researcher_output.get("findings", [])
        llm_summary = researcher_output.get("llm_summary", "")

        prompt = f"""作为数据分析师，分析以下信息并给出洞察。

任务: {task}

待分析信息:
- 已收集 {len(findings)} 条关键信息
- 主要内容: {str(llm_summary)[:500]}

请执行:
1. 对比分析: 识别主要维度的异同
2. 模式发现: 总结规律和趋势
3. 关键洞察: 最重要的3个发现
4. 风险评估: 潜在问题和不确定性"""

        response = self.call_llm(prompt)

        return {
            "insights": [
                "基于数据分析发现模式A...",
                "维度B呈现显著差异...",
                "趋势C表明未来可能..."
            ],
            "comparisons": [
                {"dimension": "维度1", "finding": "发现..."},
                {"dimension": "维度2", "finding": "发现..."}
            ],
            "patterns": ["模式1", "模式2"],
            "risks": ["风险1", "风险2"],
            "llm_analysis": response,
            "input_from": "researcher"
        }
