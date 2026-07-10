"""
指标定义与聚合 - Metrics

定义Agent系统的关键指标并提供聚合分析。

指标体系:
┌─────────────────┬──────────────┬─────────────────────────────┐
│ 指标             │ 类型          │ 说明                         │
├─────────────────┼──────────────┼─────────────────────────────┤
│ query_count     │ Counter      │ 总请求数                     │
│ success_rate    │ Gauge        │ 成功率                       │
│ avg_latency     │ Histogram    │ 平均延迟                     │
│ p95_latency     │ Histogram    │ P95延迟                      │
│ total_cost      │ Counter      │ 总成本                       │
│ tokens_per_query│ Gauge        │ 平均token消耗                │
│ retry_rate      │ Gauge        │ 重试率                       │
│ agent_usage     │ Counter      │ 各Agent使用频率              │
└─────────────────┴──────────────┴─────────────────────────────┘

学习要点:
1. RED方法: Rate(请求率), Error(错误率), Duration(延迟)
2. 百分位数(P50/P95/P99)比平均值更有意义
3. 指标聚合支持窗口计算(1min/5min/1h)
"""

from typing import Any
from collections import defaultdict
from dataclasses import dataclass, field
import time
import statistics


@dataclass
class MetricSnapshot:
    """某一时刻的指标快照"""
    timestamp: float
    query_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    agent_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    retry_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.query_count == 0:
            return 1.0
        return self.success_count / self.query_count

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def avg_cost_per_query(self) -> float:
        if self.query_count == 0:
            return 0.0
        return self.total_cost / self.query_count


class MetricsCollector:
    """
    指标收集器 - 实时收集和存储运行时指标

    用法:
        metrics = MetricsCollector()
        metrics.record_query(success=True, latency_ms=1500, cost=0.02)
        metrics.record_agent_call("researcher")
        report = metrics.get_report()
    """

    def __init__(self, window_seconds: int = 3600):
        """
        Args:
            window_seconds: 滑动窗口大小(默认1小时)
        """
        self.window_seconds = window_seconds
        # 使用多快照实现滑动窗口
        self._snapshots: list[MetricSnapshot] = []
        self._current = MetricSnapshot(timestamp=time.time())
        self._agent_usage: dict[str, int] = defaultdict(int)

    def record_query(
        self,
        success: bool,
        latency_ms: float,
        cost: float = 0.0,
        tokens: int = 0,
        retries: int = 0
    ):
        """
        记录一次查询

        Args:
            success: 是否成功
            latency_ms: 耗时(毫秒)
            cost: 成本(美元)
            tokens: token消耗
            retries: 重试次数
        """
        self._current.query_count += 1
        self._current.total_latency_ms += latency_ms
        self._current.latencies.append(latency_ms)
        self._current.total_cost += cost
        self._current.total_tokens += tokens
        self._current.retry_count += retries

        if success:
            self._current.success_count += 1
        else:
            self._current.error_count += 1

        # 限制latencies列表大小(避免内存无限增长)
        if len(self._current.latencies) > 10000:
            self._current.latencies = self._current.latencies[-5000:]

    def record_agent_call(self, agent_name: str):
        """记录一次Agent调用"""
        self._current.agent_calls[agent_name] += 1
        self._agent_usage[agent_name] += 1

    def get_report(self) -> dict:
        """
        生成指标报告

        Returns:
            包含所有关键指标的字典
        """
        return {
            "overview": {
                "total_queries": self._current.query_count,
                "success_rate": f"{self._current.success_rate:.1%}",
                "error_count": self._current.error_count,
            },
            "latency": {
                "avg_ms": round(self._current.avg_latency_ms, 1),
                "p50_ms": round(self._current.p50_latency_ms, 1),
                "p95_ms": round(self._current.p95_latency_ms, 1),
            },
            "cost": {
                "total": f"${self._current.total_cost:.4f}",
                "avg_per_query": f"${self._current.avg_cost_per_query:.6f}",
                "total_tokens": self._current.total_tokens,
            },
            "agents": dict(self._current.agent_calls),
            "quality": {
                "retry_rate": (
                    self._current.retry_count / self._current.query_count
                    if self._current.query_count > 0 else 0
                ),
            }
        }

    def get_agent_distribution(self) -> dict[str, float]:
        """获取各Agent的使用分布(百分比)"""
        total = sum(self._agent_usage.values())
        if total == 0:
            return {}
        return {name: count / total for name, count in self._agent_usage.items()}

    def reset(self):
        """重置指标"""
        self._snapshots.append(self._current)
        self._current = MetricSnapshot(timestamp=time.time())
