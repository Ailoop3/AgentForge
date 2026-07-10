"""
Agent注册表工厂

提供统一的Agent查找和创建接口。
"""

from AgentForge.agents.agents.researcher import ResearcherAgent
from AgentForge.agents.agents.analyst import AnalystAgent
from AgentForge.agents.agents.writer import WriterAgent
from AgentForge.agents.agents.reviewer import ReviewerAgent
from AgentForge.agents.base import BaseAgent

# Agent注册表 - 供Orchestrator查找实例
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "researcher": ResearcherAgent,
    "analyst": AnalystAgent,
    "writer": WriterAgent,
    "reviewer": ReviewerAgent,
}


def get_agent(name: str, **kwargs) -> BaseAgent:
    """
    工厂函数 - 根据名称创建Agent实例

    Args:
        name: Agent名称
        **kwargs: 传递给Agent构造函数的参数

    Returns:
        Agent实例

    Raises:
        ValueError: 未找到该名称的Agent
    """
    if name not in AGENT_REGISTRY:
        raise ValueError(
            f"未知Agent: {name}。可用: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[name](**kwargs)


def list_agents() -> list[str]:
    """列出所有可用Agent"""
    return list(AGENT_REGISTRY.keys())
