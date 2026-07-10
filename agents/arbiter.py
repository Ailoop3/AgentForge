"""
Blackboard & Arbiter - 共享工作区与结果裁决

Blackboard: Agent间共享数据的发布-订阅协作空间
Arbiter: 汇总/仲裁多Agent输出，产生最终结果

协作模式对比:
┌─────────────────┬────────────────────┬────────────────────┐
│ 模式             │ 通信方式            │ 适用场景           │
├─────────────────┼────────────────────┼────────────────────┤
│ 消息传递         │ 直接点对点          │ 简单协作           │
│ 黑板(Blackboard) │ 共享读写空间        │ 问题求解/诊断      │
│ 发布-订阅        │ 事件驱动            │ 异步/解耦系统      │
│ 层级汇报         │ 上下级汇报          │ 管理式架构          │
└─────────────────┴────────────────────┴────────────────────┘

本项目Blackboard结合共享空间和发布-订阅:
- Agent写入结果到共享空间
- Agent订阅自己关心的数据更新
"""

from typing import Any, Callable
from collections import defaultdict
from dataclasses import dataclass, field
import time


class Blackboard:
    """
    黑板 - Agent间共享工作区

    设计灵感来自"黑板系统"AI架构(经典如Hearsay-II语音识别系统):
    - 多个知识源(Agent)共享一个数据板
    - 每个Agent独立工作，读取黑板上的信息
    - Agent将自己的发现写回黑板
    - 整个系统逐步推进问题求解

    实现特点:
    1. 分区写入: 每个Agent有自己的命名空间(避免冲突)
    2. 订阅通知: Agent可订阅特定key的数据变化
    3. 版本追踪: 每次写入递增版本号(支持冲突检测)
    4. 读取隔离: Agent只能写自己的分区(除非显式授权)
    """

    def __init__(self):
        # 主存储: {owner: {key: {"value": Any, "version": int, "timestamp": float}}}
        self._data: dict[str, dict[str, dict]] = defaultdict(dict)
        # 订阅注册表: {key_pattern: [callback_fn, ...]}
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        # 访问日志(用于审计和调试)
        self._access_log: list[dict] = []

    def write(self, agent: str, key: str, value: Any) -> int:
        """
        向黑板写入数据

        Args:
            agent: 写入者
            key: 数据键(如"findings", "analysis")
            value: 数据内容

        Returns:
            写入的版本号
        """
        owner_space = self._data[agent]

        # 版本号递增
        current_version = owner_space.get(key, {}).get("version", 0)
        new_version = current_version + 1

        owner_space[key] = {
            "value": value,
            "version": new_version,
            "timestamp": time.time(),
            "agent": agent
        }

        # 记录访问
        self._access_log.append({
            "action": "write",
            "agent": agent,
            "key": key,
            "version": new_version,
            "timestamp": time.time()
        })

        # 通知订阅者
        self._notify(key, value, agent)

        return new_version

    def read(self, agent: str, key: str, owner: str = None) -> Any:
        """
        从黑板读取数据

        Args:
            agent: 读取者(用于审计)
            key: 数据键
            owner: 指定读取哪个Agent的数据(None表示搜索所有)

        Returns:
            数据值，未找到返回None
        """
        # 如果指定了owner，直接读取
        if owner:
            data = self._data.get(owner, {}).get(key)
            if data:
                self._access_log.append({
                    "action": "read",
                    "agent": agent,
                    "key": key,
                    "owner": owner,
                    "version": data["version"],
                    "timestamp": time.time()
                })
                return data["value"]
            return None

        # 否则搜索所有Agent的数据(取最新版本)
        latest = None
        latest_version = -1
        for owner_name, space in self._data.items():
            if key in space and space[key]["version"] > latest_version:
                latest = space[key]["value"]
                latest_version = space[key]["version"]

        return latest

    def read_all(self, agent: str) -> dict[str, Any]:
        """
        读取黑板上的所有数据(Agent视角)

        返回结构: {owner: {key: value}}
        """
        return {
            owner: {k: v["value"] for k, v in space.items()}
            for owner, space in self._data.items()
        }

    def subscribe(self, key_pattern: str, callback: Callable) -> None:
        """
        订阅数据变更通知

        Args:
            key_pattern: 关注的key(支持精确匹配)
            callback: 回调函数(key, value, writer)
        """
        self._subscribers[key_pattern].append(callback)

    def _notify(self, key: str, value: Any, writer: str) -> None:
        """通知所有匹配的订阅者"""
        for pattern, callbacks in self._subscribers.items():
            if pattern == key or pattern == "*":
                for cb in callbacks:
                    try:
                        cb(key, value, writer)
                    except Exception as e:
                        pass  # 订阅者失败不影响主流程

    def get_history(self) -> list[dict]:
        """获取操作历史日志"""
        return self._access_log.copy()

    def clear(self):
        """清空黑板"""
        self._data.clear()
        self._access_log.clear()


class Arbiter:
    """
    仲裁者 - 汇总、评估、裁决多Agent的输出

    核心功能:
    1. 聚合: 将多Agent输出融合为统一结果
    2. 冲突检测: 发现Agent间结果矛盾
    3. 质量评分: 评估每个Agent的输出质量
    4. 辩论触发: 对低置信度结果启动辩论流程

    裁决策略:
    ┌─────────────────┬─────────────────────────────┐
    │ 策略             │ 描述                         │
    ├─────────────────┼─────────────────────────────┤
    │ 拼接合并         │ 简单拼接各Agent输出            │
    │ 置信度加权       │ 按Agent自评置信度加权融合       │
    │ 投票             │ 多数同意的结果胜出             │
    │ LLM裁决         │ 让LLM评判哪个Agent输出更好      │
    │ 元评估           │ Reviewer打分最高者胜出         │
    └─────────────────┴─────────────────────────────┘
    """

    def __init__(self, strategy: str = "weighted", llm_fn: Callable = None):
        """
        Args:
            strategy: 裁决策略 (weighted/voting/llm_reviewer)
            llm_fn: LLM函数(用于llm_reviewer策略)
        """
        self.strategy = strategy
        self.llm_fn = llm_fn

    def arbitrate(self, results: dict, task: str) -> dict:
        """
        仲裁入口 - 根据策略选择裁决方式

        Args:
            results: {agent_name: AgentResult}
            task: 原始任务

        Returns:
            {
                "output": 最终输出,
                "score": 质量评分,
                "strategy": 使用的策略,
                "details": 裁决详情,
                "conflicts": 发现的冲突
            }
        """
        if not results:
            return {"output": "", "score": 0, "strategy": "none", "details": "无结果可裁决"}

        if self.strategy == "weighted":
            return self._weighted_merge(results, task)
        elif self.strategy == "voting":
            return self._voting(results, task)
        elif self.strategy == "llm_reviewer":
            return self._llm_review(results, task)
        else:
            return self._simple_concat(results)

    def detect_conflicts(self, results: dict) -> list[dict]:
        """
        检测Agent间结果冲突

        简单实现: 检查数值型结论是否不一致
        高级实现: 用LLM判断语义矛盾

        Args:
            results: Agent输出集合

        Returns:
            冲突列表 [{agent_a, agent_b, issue}]
        """
        conflicts = []
        agent_names = list(results.keys())

        # 比较每对Agent的输出
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                a, b = agent_names[i], agent_names[j]
                # 简单启发: 如果输出完全不同类型，可能冲突
                output_a = results[a].output
                output_b = results[b].output

                if type(output_a) != type(output_b):
                    conflicts.append({
                        "agent_a": a,
                        "agent_b": b,
                        "issue": "输出类型不一致",
                        "severity": "low"
                    })

        return conflicts

    def _weighted_merge(self, results: dict, task: str) -> dict:
        """
        置信度加权合并

        原理: 置信度高的Agent输出占更大权重
        final = Σ(conf_i * output_i) / Σ(conf_i)

        适用: Agent输出可量化的场景
        """
        total_conf = sum(r.confidence for r in results.values())
        if total_conf == 0:
            total_conf = 1  # 避免除零

        # 加权平均分数
        avg_score = total_conf / len(results)

        # 按置信度排序，选择主要输出
        sorted_results = sorted(results.items(), key=lambda x: x[1].confidence, reverse=True)

        # 主输出取最高置信度Agent的结果
        primary = sorted_results[0][1]

        return {
            "output": primary.output,
            "score": avg_score,
            "strategy": "weighted",
            "details": {
                "primary_agent": sorted_results[0][0],
                "confidences": {name: r.confidence for name, r in results.items()},
                "total_agents": len(results)
            },
            "conflicts": self.detect_conflicts(results)
        }

    def _voting(self, results: dict, task: str) -> dict:
        """
        投票裁决

        原理: 多个Agent给出相似结论时，多数意见胜出
        适用: 有明确对错的场景(如事实性问题)
        """
        # 简化实现: 按confidence投票(高confidence算多票)
        vote_count = defaultdict(float)
        for name, result in results.items():
            # 输出的哈希作为"选项"(相同输出算同一选项)
            option_key = str(result.output)[:100]  # 取前100字符作为key
            vote_count[option_key] += result.confidence

        # 票数最多的胜出
        winner = max(vote_count, key=vote_count.get)

        return {
            "output": winner,
            "score": vote_count[winner] / sum(vote_count.values()),
            "strategy": "voting",
            "details": {"votes": dict(vote_count)}
        }

    def _llm_review(self, results: dict, task: str) -> dict:
        """LLM作为裁判评判最佳输出"""
        if not self.llm_fn:
            return self._weighted_merge(results, task)

        # 构造评审prompt
        options = "\n\n".join([
            f"--- {name} ---\n{result.output}"
            for name, result in results.items()
        ])

        prompt = f"""任务: {task}

以下是不同Agent的输出，请选择最佳的一个并说明理由:

{options}

请输出JSON: {{"best_agent": "名称", "score": 0-10, "reason": "理由"}}"""

        try:
            response = self.llm_fn("你是一个公正的评审官。", prompt)
            import json
            json_str = response[response.find("{"):response.find("}")+1]
            verdict = json.loads(json_str)

            best_agent = verdict.get("best_agent", "")
            best_result = results.get(best_agent)

            return {
                "output": best_result.output if best_result else str(results),
                "score": verdict.get("score", 5) / 10,
                "strategy": "llm_reviewer",
                "details": verdict
            }
        except Exception:
            return self._weighted_merge(results, task)

    def _simple_concat(self, results: dict) -> dict:
        """简单拼接(保底策略)"""
        parts = [f"[{name}]: {result.output}" for name, result in results.items()]
        return {
            "output": "\n\n".join(parts),
            "score": sum(r.confidence for r in results.values()) / len(results),
            "strategy": "concat"
        }
