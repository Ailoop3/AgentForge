"""
具体Agent实现

四个核心Agent:
1. Researcher - 信息检索与收集
2. Analyst - 数据分析与对比
3. Writer - 内容撰写与生成
4. Reviewer - 质量审查与改进
"""

from AgentForge.agents.agents.researcher import ResearcherAgent
from AgentForge.agents.agents.analyst import AnalystAgent
from AgentForge.agents.agents.writer import WriterAgent
from AgentForge.agents.agents.reviewer import ReviewerAgent
from AgentForge.agents.base import BaseAgent

# Agent注册表 - 供Orchestrator查找实例
AGENT_REGISTRY = {
    "researcher": ResearcherAgent,
    "analyst": AnalystAgent,
    "writer": WriterAgent,
    "reviewer": ReviewerAgent,
}


def get_agent(name: str, **kwargs) -> BaseAgent:
    """
    工厂函数 - 根据名称创建Agent实例
    """
    if name not in AGENT_REGISTRY:
        raise ValueError(f"未知Agent: {name}。可用: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[name](**kwargs)
