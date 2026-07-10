"""
Router & Planner - 任务路由与分解

Router: 根据用户意图选择合适的Agent组合
Planner: 将复杂任务分解为DAG(有向无环图)

分解示例:
用户输入: "帮我调研竞品并写一份对比报告"

Router选择: [Researcher, Analyst, Writer, Reviewer]
Planner生成DAG:
  Researcher ──→ Analyst ──→ Writer ──→ Reviewer
       │                         ↑
       └─────────────────────────┘ (研究者也提供原始数据)

学习要点:
1. 路由本质是分类问题(可规则/可模型)
2. 任务分解需要理解任务间依赖关系
3. DAG拓扑排序决定执行顺序
"""

from typing import Protocol
from dataclasses import dataclass, field
from enum import Enum
from AgentForge.agents.state import AgentState


class IntentType(Enum):
    """预定义的任务意图类型"""
    RESEARCH = "research"        # 调研类
    ANALYSIS = "analysis"        # 分析类
    WRITING = "writing"          # 写作类
    REVIEW = "review"            # 评审类
    COMPLEX = "complex"          # 综合类(需要多Agent)


class Router(Protocol):
    """路由器协议"""

    def route(self, task: str) -> list[str]:
        """根据任务选择参与的Agent列表"""
        ...


class Planner(Protocol):
    """规划器协议"""

    def plan(self, task: str, agent_names: list[str]) -> 'DAG':
        """生成执行DAG"""
        ...


class KeywordRouter:
    """
    基于关键词的路由器 - 简单高效的规则路由

    原理: 根据任务描述中的关键词匹配，决定需要哪些Agent。
    适合意图明确、类别有限的场景。

    扩展方向:
    - 用LLM做意图分类(更灵活但慢)
    - 用Embedding相似度匹配(介之间)
    - 混合策略: 优先规则，兜底LLM
    """

    # 关键词 → Agent映射表
    KEYWORD_MAP = {
        "researcher": ["调研", "搜索", "查找", "搜集", "research", "search", "find"],
        "analyst": ["分析", "对比", "评估", "研究", "analyze", "compare", "evaluate"],
        "writer": ["写", "撰写", "生成", "报告", "write", "draft", "compose"],
        "reviewer": ["审查", "评审", "检查", "review", "check", "evaluate"],
    }

    def route(self, task: str) -> list[str]:
        """
        关键词匹配路由

        匹配逻辑:
        1. 遍历每个Agent的关键词列表
        2. 统计命中数
        3. 命中>0则选中该Agent
        4. 多Agent命中则全选(表示复杂任务)

        Args:
            task: 用户任务

        Returns:
            选中的Agent名称列表
        """
        task_lower = task.lower()
        selected = []
        scores = {}

        for agent, keywords in self.KEYWORD_MAP.items():
            # 计算匹配分数(命中关键词数)
            match_count = sum(1 for kw in keywords if kw in task_lower)
            if match_count > 0:
                selected.append(agent)
                scores[agent] = match_count

        # 如果没有匹配任何关键词，默认走完整流程
        if not selected:
            selected = ["researcher", "analyst", "writer", "reviewer"]

        # 按分数排序(优先级高的先执行)
        selected.sort(key=lambda a: scores.get(a, 0), reverse=True)

        return selected


class LLMRouter:
    """
    LLM路由器 - 用大模型做意图分类

    相比关键词路由的优势:
    - 理解语义而非表面词汇
    - 可处理模糊/隐含意图
    - 支持更丰富的分类

    劣势:
    - 调用有延迟和成本
    - LLM可能判断错误
    """

    ROUTING_PROMPT = """分析以下任务，判断需要哪些Agent参与。

可选Agent:
- researcher: 信息搜索、资料搜集、知识查询
- analyst: 数据分析、对比评估、模式发现
- writer: 文本生成、报告撰写、内容创作
- reviewer: 质量审查、错误检查、改进建议

任务: {task}

请输出JSON格式: {{"agents": ["researcher", "writer"], "reason": "解释"}}
只输出JSON。"""

    def __init__(self, llm_fn: callable):
        self.llm_fn = llm_fn

    def route(self, task: str) -> list[str]:
        """LLM意图分类"""
        prompt = self.ROUTING_PROMPT.format(task=task)
        try:
            response = self.llm_fn("你是一个路由分析专家。", prompt)
            import json
            # 提取JSON
            json_str = response[response.find("{"):response.find("}")+1]
            result = json.loads(json_str)
            return result.get("agents", ["researcher", "writer"])
        except Exception:
            # 失败时回退到完整流程
            return ["researcher", "analyst", "writer", "reviewer"]


@dataclass
class DAGNode:
    """
    DAG节点 - 表示一个Agent执行步骤

    属性:
    - agent: 执行该步骤的Agent名称
    - dependencies: 前置步骤列表(必须等这些完成才能执行)
    - condition: 是否执行该节点的条件函数
    - retry_policy: 失败重试策略
    - fallback: 失败时替代执行的Agent

    DAG示例:
        A(Researcher) ─┬─→ B(Analyst) ─→ C(Writer) ─→ D(Reviewer)
                      └──────────→ C
    """
    agent: str
    dependencies: list[str] = field(default_factory=list)
    condition: callable = None
    retry_policy: dict = field(default_factory=lambda: {"max_retries": 2, "backoff": 1.0})
    fallback: str | None = None


@dataclass
class DAG:
    """
    有向无环图 - Agent执行的编排计划

    拓扑排序后的节点列表决定执行顺序。
    无依赖的节点可并行执行。
    """
    nodes: list[DAGNode] = field(default_factory=list)

    def get_execution_order(self) -> list[list[DAGNode]]:
        """
        拓扑排序 - 分层返回执行计划

        返回分层结构，同一层的节点无依赖关系可并行。

        示例返回:
        [
            [Node(researcher)],           # 第1层: 无依赖，先执行
            [Node(analyst)],              # 第2层: 依赖researcher
            [Node(writer)],               # 第3层: 依赖analyst
            [Node(reviewer)],             # 第4层: 依赖writer
        ]

        Returns:
            按执行顺序分层的节点列表
        """
        # 构建依赖图
        node_map = {n.agent: n for n in self.nodes}
        in_degree = {n.agent: len(n.dependencies) for n in self.nodes}

        layers = []
        remaining = set(n.agent for n in self.nodes)

        while remaining:
            # 找到入度为0的节点(无未满足依赖)
            layer = [node_map[a] for a in remaining if in_degree[a] == 0]
            if not layer:
                raise ValueError("DAG存在环，无法拓扑排序")

            layers.append(layer)
            for node in layer:
                remaining.remove(node.agent)
                # 减少依赖当前节点的后续节点入度
                for other in self.nodes:
                    if node.agent in other.dependencies:
                        in_degree[other.agent] -= 1

        return layers

    def get_dependencies(self, agent: str) -> list[str]:
        """获取某个Agent的所有前置依赖"""
        node = next((n for n in self.nodes if n.agent == agent), None)
        return node.dependencies if node else []


class DefaultPlanner:
    """
    默认规划器 - 根据Agent列表生成合理的DAG

    规划策略(领域知识):
    - Researcher总是最先(数据源)
    - Analyst依赖Researcher(需要原始数据)
    - Writer依赖Analyst(需要分析结果)
    - Reviewer总是最后(审查最终输出)

    这种领域知识确保了DAG的合理性。
    更智能的Planner应让LLM参与分解。
    """

    # 标准执行流水线(领域知识)
    STANDARD_PIPELINE = ["researcher", "analyst", "writer", "reviewer"]

    def plan(self, task: str, agent_names: list[str]) -> DAG:
        """
        生成执行DAG

        规则:
        1. 按标准流水线顺序排列选中的Agent
        2. 每个Agent依赖前面所有Agent(确保信息流)
        3. Writer额外依赖Researcher(需要原始数据与分析结果)

        Args:
            task: 任务描述
            agent_names: Router选中的Agent列表

        Returns:
            执行DAG
        """
        # 按标准顺序过滤
        ordered = [a for a in self.STANDARD_PIPELINE if a in agent_names]

        nodes = []
        for i, agent in enumerate(ordered):
            # 前面所有Agent都是依赖
            deps = ordered[:i]

            # Writer额外直接依赖Researcher
            if agent == "writer" and "researcher" in [d for d in deps]:
                pass  # 已经包含在ordered[:i]中

            nodes.append(DAGNode(
                agent=agent,
                dependencies=deps
            ))

        return DAG(nodes=nodes)


class LLMDAGPlanner:
    """
    LLM驱动的DAG规划器

    让LLM理解任务语义，生成合理的执行计划。
    适合开放域、非标准化任务。
    """

    PLAN_PROMPT = """将以下任务分解为执行计划。

可用角色:
- researcher: 搜索和搜集信息
- analyst: 分析和对比数据
- writer: 撰写报告/文档
- reviewer: 审查和改进输出

任务: {task}

已选角色: {agents}

请生成DAG依赖关系(哪些角色需要在哪些角色之后执行)。
输出JSON: {{"nodes": [{{"agent": "researcher", "dependencies": []}}, ...]}}
只输出JSON。"""

    def __init__(self, llm_fn: callable):
        self.llm_fn = llm_fn

    def plan(self, task: str, agent_names: list[str]) -> DAG:
        """LLM生成DAG"""
        prompt = self.PLAN_PROMPT.format(task=task, agents=agent_names)
        try:
            response = self.llm_fn("你是一个任务规划专家。", prompt)
            import json
            json_str = response[response.find("{"):response.find("}")+1]
            data = json.loads(json_str)
            nodes = [DAGNode(**n) for n in data["nodes"]]
            return DAG(nodes=nodes)
        except Exception:
            # 失败时使用默认规划
            return DefaultPlanner().plan(task, agent_names)
