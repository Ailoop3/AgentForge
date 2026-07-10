"""
回放器 - Replay

基于存储的trace回放Agent执行过程，用于:
1. 调试 - 查看每一步的输入输出
2. 教学 - 理解Agent决策过程
3. 审计 - 检查执行是否符合预期
4. 对比 - 不同版本/配置的差异对比

回放模式:
┌─────────────────┬────────────────────────────────────┐
│ 模式             │ 说明                                │
├─────────────────┼────────────────────────────────────┤
│ 逐步回放         │ 按span顺序逐步展示(带时间间隔)       │
│ 快速回放         │ 一次性展示所有步骤                    │
│ 对比回放         │ 并排比较两条trace                     │
│ 单点调试         │ 聚焦某个span的详细信息               │
└─────────────────┴────────────────────────────────────┘

学习要点:
1. 回放是可观测性的"最后一公里"
2. 好的回放工具能大幅降低Agent调试难度
3. trace的结构化存储是回放的基础
"""

import json
import os
import time
from typing import Any
from dataclasses import dataclass

from AgentForge.observability.tracer import Tracer, TraceTree, Span, SpanType


@dataclass
class ReplayStep:
    """
    回放步骤 - 单条span的可展示格式

    将技术性的Span转换为适合展示的步骤信息。
    """
    step_number: int
    name: str
    agent: str
    span_type: str
    duration_ms: float
    cost: float
    input_summary: str
    output_summary: str
    status: str           # success / error / pending
    timestamp: float


class Replayer:
    """
    回放器 - 基于trace数据的执行回放

    功能:
    1. 加载trace数据
    2. 按时间顺序重放
    3. 生成逐步报告
    4. 支持查询特定环节
    """

    def __init__(self, tracer: Tracer):
        """
        Args:
            tracer: Tracer实例(用于访问存储的trace)
        """
        self.tracer = tracer

    def load_trace(self, trace_id: str) -> dict | None:
        """
        加载指定trace的完整数据

        Args:
            trace_id: trace ID

        Returns:
            trace原始数据字典
        """
        return self.tracer.get_trace_detail(trace_id)

    def replay(self, trace_id: str, verbose: bool = True) -> list[ReplayStep]:
        """
        回放指定trace

        Args:
            trace_id: trace ID
            verbose: 是否打印到控制台

        Returns:
            按时间排序的回放步骤列表
        """
        data = self.load_trace(trace_id)
        if not data:
            return []

        steps = self._build_steps(data)

        if verbose:
            self._print_replay(steps, data.get("task", "未知任务"))

        return steps

    def _build_steps(self, trace_data: dict) -> list[ReplayStep]:
        """
        从trace数据构建回放步骤

        主要做格式转换: 原始span → 展示用ReplayStep
        """
        steps = []
        root = trace_data.get("root_span", {})
        if not root:
            return steps

        # 递归处理所有span
        self._process_span(root, steps, counter=[0])
        return sorted(steps, key=lambda s: s.timestamp)

    def _process_span(self, span_data: dict, steps: list, counter: list):
        """递归处理span树"""
        counter[0] += 1

        status = "error" if span_data.get("error") else "success"
        error_info = f" [错误: {span_data['error']}]" if span_data.get("error") else ""

        step = ReplayStep(
            step_number=counter[0],
            name=span_data.get("name", "unknown"),
            agent=span_data.get("agent", ""),
            span_type=span_data.get("span_type", "custom"),
            duration_ms=span_data.get("duration_ms", 0),
            cost=span_data.get("cost", 0),
            input_summary=self._summarize(span_data.get("input")),
            output_summary=self._summarize(span_data.get("output")) + error_info,
            status=status,
            timestamp=span_data.get("start_time", 0)
        )
        steps.append(step)

    def _summarize(self, data: Any, max_len: int = 80) -> str:
        """生成数据的简短摘要"""
        if data is None:
            return "无"
        text = str(data)
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text

    def _print_replay(self, steps: list['ReplayStep'], task: str):
        """打印回放过程到控制台"""
        print(f"\n{'='*60}")
        print(f"Trace 回放")
        print(f"任务: {task}")
        print(f"总步骤: {len(steps)}")
        print(f"{'='*60}\n")

        for step in steps:
            # 状态标记
            icon = "✓" if step.status == "success" else "✗"
            # 耗时着色(概念上，这里用文字标注)
            speed = self._speed_label(step.duration_ms)

            print(f"{icon} Step {step.step_number:2d} | {step.name:<25s} | {step.duration_ms:8.1f}ms {speed} | ${step.cost:.5f}")
            print(f"  Input:  {step.input_summary}")
            print(f"  Output: {step.output_summary}")
            print()

        # 汇总
        total_cost = sum(s.cost for s in steps)
        total_time = sum(s.duration_ms for s in steps)
        errors = sum(1 for s in steps if s.status == "error")

        print(f"{'='*60}")
        print(f"汇总: {len(steps)}步骤 | {total_time:.0f}ms | ${total_cost:.5f} | {errors}个错误")
        print(f"{'='*60}\n")

    def _speed_label(self, ms: float) -> str:
        """根据耗时返回速度标签"""
        if ms < 100:
            return "(快)"
        elif ms < 1000:
            return "(中)"
        else:
            return "(慢)"

    def compare_traces(self, trace_id_a: str, trace_id_b: str) -> dict:
        """
        对比两条trace的差异

        用于:
        - 前后版本对比(改了prompt后的效果)
        - 不同配置对比(换了模型后的差异)
        - 成功vs失败对比

        Returns:
            包含各项对比指标的字典
        """
        data_a = self.load_trace(trace_id_a)
        data_b = self.load_trace(trace_id_b)

        if not data_a or not data_b:
            return {"error": "无法加载trace数据"}

        metrics_a = data_a.get("root_span", {})
        metrics_b = data_b.get("root_span", {})

        return {
            "task_a": data_a.get("task"),
            "task_b": data_b.get("task"),
            "duration": {
                "a_ms": metrics_a.get("duration_ms", 0),
                "b_ms": metrics_b.get("duration_ms", 0),
                "diff_ms": metrics_b.get("duration_ms", 0) - metrics_a.get("duration_ms", 0)
            },
            "cost": {
                "a": data_a.get("total_cost", 0),
                "b": data_b.get("total_cost", 0),
                "diff": data_b.get("total_cost", 0) - data_a.get("total_cost", 0)
            }
        }

    def get_trace_timeline(self, trace_id: str) -> list[dict]:
        """
        生成时间线数据(供Dashboard可视化)

        返回按时间排序的span列表，每个包含:
        - 开始时间
        - 持续时间
        - 名称
        - Agent
        - 类型
        """
        data = self.load_trace(trace_id)
        if not data:
            return []

        steps = self._build_steps(data)
        return [
            {
                "name": s.name,
                "agent": s.agent,
                "type": s.span_type,
                "start": s.timestamp,
                "duration_ms": s.duration_ms,
                "status": s.status
            }
            for s in steps
        ]
