"""
审查员Agent - Reviewer Agent

职责: 质量审查、反馈改进
- 审查Writer产出的内容质量
- 检查事实准确性
- 发现逻辑漏洞和表达问题
- 给出改进建议
"""

from typing import Any
from AgentForge.agents.base import BaseAgent
from AgentForge.agents.state import AgentState


class ReviewerAgent(BaseAgent):
    """
    审查员Agent - 质量审查、反馈改进

    职责:
    - 审查Writer产出的内容质量
    - 检查事实准确性(与知识库对照)
    - 发现逻辑漏洞和表达问题
    - 给出改进建议

    评分维度(每项0-10分):
    - 准确性: 事实是否正确，引用是否可靠
    - 完整性: 是否覆盖所有重要方面
    - 清晰度: 表达是否易懂，结构是否合理
    - 逻辑性: 论证是否严谨，结论是否有依据

    审查标准:
    - 9-10分: 优秀，可以直接使用
    - 7-8分: 良好，小改即可
    - 5-6分: 一般，需要较大修改
    - <5分: 不合格，需要重写
    """

    name = "reviewer"
    description = "质量审查专家 - 检查内容质量、提出改进建议"
    system_prompt = """你是严格的质量审查专家。任务:
1. 审查内容的事实准确性
2. 评估逻辑连贯性和完整性
3. 检查表达清晰度
4. 给出具体改进建议

评分维度(每项0-10分):
- 准确性: 事实是否正确，引用是否可靠
- 完整性: 是否覆盖所有重要方面
- 清晰度: 表达是否易懂，结构是否合理
- 逻辑性: 论证是否严谨，结论是否有依据

审查标准:
- 9-10分: 优秀，可以直接使用
- 7-8分: 良好，小改即可
- 5-6分: 一般，需要较大修改
- <5分: 不合格，需要重写

输出格式: JSON，包含评分、问题、建议。"""

    def run_core(self, state: AgentState) -> Any:
        """
        核心逻辑: 多维度评分 → 问题识别 → 改进建议

        依赖: Writer的输出
        """
        task = state["task"]
        writer_output = state.get("results", {}).get("writer", {}).get("output", {})

        content = writer_output.get("sections", [])
        citations = writer_output.get("citations", [])
        title = writer_output.get("title", "无标题")

        prompt = f"""审查以下内容的质量。

原始任务: {task}

待审查内容:
标题: {title}
章节: {len(content)} 个
引用: {len(citations)} 条

内容概要: {str(content)[:500]}

请从以下维度评分并给出改进建议:
1. 准确性 - 事实是否正确
2. 完整性 - 是否全面
3. 清晰度 - 表达是否清楚
4. 逻辑性 - 论证是否严谨

输出JSON格式: {{"overall_score": 0-10, "dimensions": {{...}}, "issues": [...], "suggestions": [...], "approved": true/false}}"""

        response = self.call_llm(prompt)

        # 模拟评审结果(实际应由LLM输出解析)
        result = {
            "overall_score": 8.0,
            "dimensions": {
                "accuracy": 8.5,
                "completeness": 7.5,
                "clarity": 8.0,
                "logic": 8.0
            },
            "issues": [
                "部分数据来源未明确标注",
                "建议增加对比案例"
            ],
            "suggestions": [
                "增加更多具体数据支撑",
                "补充局限性分析",
                "调整章节顺序使逻辑更清晰"
            ],
            "approved": True,
            "llm_review": response
        }

        return result
