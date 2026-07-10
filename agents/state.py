"""
全局状态定义 - AgentState

LangGraph风格的状态管理，使用TypedDict定义状态schema。
所有Agent共享状态，通过Annotated的reducer函数控制字段更新方式。

状态流转:
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Router  │───→│ Planner  │───→│ Executor │
└──────────┘    └──────────┘    └──────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │  Arbiter     │
                              │ (汇总/裁决)   │
                              └──────────────┘

学习要点:
1. TypedDict兼顾类型安全和运行时性能
2. Annotated reducer控制并发写入语义
3. 支持checkpoint(中断恢复)和human-in-the-loop
"""

from typing import TypedDict, Annotated, Any
from datetime import datetime


def append_list(existing: list, new: list) -> list:
    """列表累加reducer - Agent输出按追加合并"""
    if existing is None:
        return new
    return existing + new


def update_dict(existing: dict, new: dict) -> dict:
    """字典更新reducer - 合并结果"""
    if existing is None:
        return new
    return {**existing, **new}


def replace_value(existing: Any, new: Any) -> Any:
    """替换reducer - 直接覆盖"""
    return new


class AgentState(TypedDict):
    """
    全局状态 - 所有Agent共享的工作上下文

    设计原则:
    - 读多写少的字段用replace(如task, trace_id)
    - Agent间协作字段用append/update(如messages, results)
    - 每个Agent只写自己的命名空间(如results["researcher"])

    字段分类:
    ┌─────────────┬──────────────┬─────────────────────────┐
    │ 字段         │ 更新方式      │ 用途                     │
    ├─────────────┼──────────────┼─────────────────────────┤
    │ messages    │ append       │ 对话历史(所有消息)        │
    │ task        │ replace      │ 原始任务描述              │
    │ subtasks    │ replace      │ Planner分解的子任务       │
    │ results     │ update       │ 各Agent的输出结果         │
    │ current_step│ replace      │ 当前执行步骤(调试用)      │
    │ trace_id    │ replace      │ 链路追踪ID               │
    │ metadata    │ update       │ 运行时元信息              │
    │ errors      │ append       │ 错误记录(用于失败分析)    │
    └─────────────┴──────────────┴─────────────────────────┘
    """
    # 对话历史 - append模式(每轮追加)
    messages: Annotated[list[dict], append_list]

    # 任务定义 - replace模式(不变)
    task: Annotated[str, replace_value]
    subtasks: Annotated[dict[str, str], replace_value]

    # Agent输出 - update模式(按agent名合并)
    results: Annotated[dict[str, Any], update_dict]

    # 执行控制
    current_step: Annotated[str, replace_value]
    current_agent: Annotated[str, replace_value]

    # 追踪
    trace_id: Annotated[str, replace_value]
    created_at: Annotated[str, replace_value]

    # 元数据(存储各种运行时信息)
    metadata: Annotated[dict, update_dict]

    # 错误记录
    errors: Annotated[list[dict], append_list]

    # 裁决结果
    final_output: Annotated[Any, replace_value]
    final_score: Annotated[float, replace_value]


def create_initial_state(task: str, trace_id: str) -> AgentState:
    """
    创建初始状态 - 统一的状态工厂

    使用工厂函数确保所有必需字段都有初始值，
    避免不同Agent读到None导致错误。

    Args:
        task: 用户输入的任务
        trace_id: 链路追踪ID

    Returns:
        初始化的AgentState
    """
    return AgentState(
        messages=[{
            "role": "user",
            "content": task,
            "timestamp": datetime.now().isoformat()
        }],
        task=task,
        subtasks={},
        results={},
        current_step="init",
        current_agent="",
        trace_id=trace_id,
        created_at=datetime.now().isoformat(),
        metadata={},
        errors=[],
        final_output=None,
        final_score=0.0
    )
