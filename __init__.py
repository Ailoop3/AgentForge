"""
AgentForge - 3-in-1 Agent Learning Platform

整合:
1. RAG Engine - 知识检索增强
2. Multi-Agent Core - 多智能体协作
3. Eval & Observability - 评估与可观测性
"""

from .orchestrator import Orchestrator, OrchestratorResult
from .rag.knowledge_base import KnowledgeBase
from .rag.models import SourceType
from .config import settings
from .__main__ import AgentForge, quick_demo

__version__ = "0.1.0"
__all__ = [
    "AgentForge",
    "Orchestrator",
    "OrchestratorResult",
    "KnowledgeBase",
    "SourceType",
    "settings",
    "quick_demo",
]
