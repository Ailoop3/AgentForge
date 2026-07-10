"""
Agent基类 - BaseAgent

所有具体Agent的抽象基类，定义通用行为和接口。
使用模板方法模式: 子类只实现核心逻辑，基类处理通用流程。

Agent生命周期:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ prepare  │───→│  think   │───→│  act     │───→│ reflect  │
│ (准备)    │    │ (推理)    │    │ (执行)   │    │ (反思)   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
      ↑                                              │
      └───────────── 循环直到完成或超限 ────────────────┘

学习要点:
1. 模板方法模式: 基类定义骨架，子类填充细节
2. 每个Agent有独立知识库 → 知识隔离
3. Trace上下文全程传递 → 可观测性
4. 重试策略 → 容错能力
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from datetime import datetime
import uuid
import time

from AgentForge.agents.state import AgentState
from AgentForge.rag.knowledge_base import KnowledgeBase


@dataclass
class Tool:
    """
    Agent工具定义

    工具是Agent与外界交互的能力扩展。
    每个工具包含:
    - 名称和描述(LLM用于决定是否调用)
    - 参数schema(LLM用于构造调用参数)
    - 执行函数(实际逻辑)

    设计参考了OpenAI Function Calling / LangChain Tools
    """
    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)

    def execute(self, **kwargs) -> Any:
        """执行工具并返回结果"""
        return self.func(**kwargs)


@dataclass
class AgentResult:
    """
    Agent执行结果

    标准化输出格式，包含:
    - 实际输出内容
    - 中间过程(思考、工具调用等)
    - 性能指标(耗时、token)
    - 置信度(用于Arbiter裁决)
    """
    agent_name: str
    output: Any
    thinking: list[str] = field(default_factory=list)      # 推理过程
    tool_calls: list[dict] = field(default_factory=list)   # 工具调用记录
    confidence: float = 0.5      # 自我置信度(0-1)
    duration: float = 0.0        # 执行耗时(秒)
    success: bool = True
    error: str = ""


class BaseAgent(ABC):
    """
    Agent基类 - 所有智能体的公共父类

    子类需要实现:
    - name: Agent名称
    - system_prompt: 系统提示词(定义角色和行为)
    - run_core(): 核心执行逻辑

    基类提供:
    - 状态管理
    - 工具调用框架
    - 知识库查询
    - 错误处理和重试
    - 执行追踪
    """

    # 子类覆盖这些属性
    name: str = "base"
    description: str = "基础Agent"
    system_prompt: str = "你是一个智能助手。"

    def __init__(
        self,
        llm_fn: Callable[[str, str], str] | None = None,
        knowledge_base: KnowledgeBase | None = None,
        tools: list[Tool] | None = None,
        max_iterations: int = None
    ):
        """
        Args:
            llm_fn: LLM调用函数 (system_prompt, user_prompt) → response
            knowledge_base: 专属知识库
            tools: 可用工具列表
            max_iterations: 最大思考-行动循环次数
        """
        self.llm_fn = llm_fn or self._default_llm
        self.knowledge_base = knowledge_base
        self.tools = tools or []
        self.max_iterations = max_iterations or 10

    @abstractmethod
    def run_core(self, state: AgentState) -> Any:
        """
        核心执行逻辑 - 子类必须实现

        不同类型的Agent有不同的实现:
        - Researcher: 检索→筛选→整理信息
        - Analyst: 对比→归因→发现模式
        - Writer: 结构化→生成→润色
        - Reviewer: 审查→打分→建议

        Args:
            state: 当前全局状态(可读取其他Agent的结果)

        Returns:
            Agent的输出内容
        """
        pass

    def run(self, state: AgentState) -> tuple[AgentState, AgentResult]:
        """
        完整的Agent执行入口(模板方法)

        流程:
        1. 状态注入(更新current_agent)
        2. 准备阶段(查询知识库等)
        3. 核心执行
        4. 结果包装
        5. 状态更新

        Args:
            state: 全局状态

        Returns:
            (更新后的状态, Agent执行结果)
        """
        start_time = time.time()

        # 记录当前执行的Agent
        state["current_agent"] = self.name
        state["current_step"] = f"{self.name}.running"

        try:
            # Step 1: 准备 - 收集上下文
            context = self._prepare(state)

            # Step 2: 核心执行(子类实现)
            output = self.run_core(state)

            # Step 3: 包装结果
            result = AgentResult(
                agent_name=self.name,
                output=output,
                duration=time.time() - start_time,
                success=True
            )

            # Step 4: 更新全局状态
            state["results"][self.name] = {
                "output": output,
                "confidence": result.confidence,
                "duration": result.duration,
                "timestamp": datetime.now().isoformat()
            }
            state["messages"].append({
                "role": "agent",
                "agent": self.name,
                "content": str(output),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            # 错误处理: 记录但不让整个系统崩溃
            result = AgentResult(
                agent_name=self.name,
                output=None,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
            state["errors"].append({
                "agent": self.name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

        state["current_step"] = f"{self.name}.done"
        return state, result

    def _prepare(self, state: AgentState) -> dict:
        """
        准备阶段 - 收集Agent需要的上下文

        包括:
        - 从知识库查询相关知识
        - 从blackboard读取其他Agent的结果
        - 构建prompt上下文

        Args:
            state: 当前状态

        Returns:
            上下文字典
        """
        context = {
            "task": state["task"],
            "subtasks": state.get("subtasks", {}),
            "other_results": {
                k: v for k, v in state.get("results", {}).items()
                if k != self.name
            }
        }

        # 知识库查询(如果有)
        if self.knowledge_base and self.knowledge_base.size > 0:
            kb_context = self.knowledge_base.get_context(state["task"])
            context["kb_chunks"] = kb_context.context_text
            context["kb_token_count"] = kb_context.total_tokens

        return context

    def call_llm(self, user_prompt: str, system_prompt: str = None) -> str:
        """
        封装LLM调用，统一处理异常

        Args:
            user_prompt: 用户提示
            system_prompt: 系统提示(默认使用Agent自身的)

        Returns:
            LLM返回的文本
        """
        sp = system_prompt or self.system_prompt
        try:
            return self.llm_fn(sp, user_prompt)
        except Exception as e:
            return f"[LLM调用失败: {e}]"

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用指定工具

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if tool is None:
            raise ValueError(f"工具未找到: {tool_name}。可用工具: {[t.name for t in self.tools]}")
        return tool.execute(**kwargs)

    def should_retry(self, state: AgentState, error: str) -> bool:
        """
        判断是否值得重试

        策略:
        - 网络/超时错误 → 重试
        - LLM输出格式错误 → 重试
        - 逻辑错误 → 不重试(重试也没用)

        Args:
            state: 当前状态
            error: 错误信息

        Returns:
            是否重试
        """
        retryable_keywords = ["timeout", "connection", "rate limit", "格式错误"]
        return any(kw in error.lower() for kw in retryable_keywords)

    @property
    def available_tools(self) -> list[str]:
        """返回可用工具名称列表"""
        return [t.name for t in self.tools]

    def _default_llm(self, system: str, user: str) -> str:
        """
        默认LLM - 占位实现

        实际使用时应注入真实的LLM调用函数，如:
        - OpenAI ChatCompletion
        - 本地Ollama / vLLM
        - 其他兼容API
        """
        return f"[默认响应] 收到提示: {user[:50]}..."

    def __repr__(self):
        return f"{self.name}(kb={self.knowledge_base}, tools={self.available_tools})"
