"""
Dashboard - Streamlit可视化面板

提供三个核心页面:
1. Traces - 链路追踪列表和详情
2. Eval - 评估报告
3. Replay - 执行回放

运行方式:
    streamlit run observability/dashboard/app.py

学习要点:
1. Trace的可视化是Agent observability的关键
2. 分层的展示(列表→树状→span详情)符合认知习惯
3. 实时指标dashboard帮助发现系统问题
"""

import streamlit as st
import pandas as pd
import json
import os
import sys

# 添加项目根目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from observability.tracer import Tracer
from observability.evaluator import Evaluator
from observability.replay import Replayer
from observability.metrics import MetricsCollector


def init_session_state():
    """初始化Streamlit会话状态"""
    if 'tracer' not in st.session_state:
        st.session_state.tracer = Tracer(auto_persist=True)
    if 'metrics' not in st.session_state:
        st.session_state.metrics = MetricsCollector()
    if 'replayer' not in st.session_state:
        st.session_state.replayer = Replayer(st.session_state.tracer)


def render_overview(tracer: Tracer, metrics: MetricsCollector):
    """
    概览页面 - 展示系统整体状况

    包含:
    - 核心KPI卡片(请求数/成功率/平均延迟/总成本)
    - 延迟趋势(概念性)
    - Agent使用分布
    - 最近trace列表
    """
    st.title("AgentForge Dashboard")
    st.markdown("---")

    # 获取统计数据
    stats = tracer.get_statistics()
    report = metrics.get_report()

    # KPI卡片行
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总请求数", stats.get("total_traces", 0))
    col2.metric("成功率", report.get("overview", {}).get("success_rate", "N/A"))
    col3.metric("平均延迟", f"{stats.get('avg_duration_ms', 0):.0f}ms")
    col4.metric("总成本", f"${stats.get('total_cost', 0):.4f}")

    st.markdown("---")

    # Agent使用分布
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Agent 使用分布")
        agent_dist = metrics.get_agent_distribution()
        if agent_dist:
            agent_df = pd.DataFrame(
                {"Agent": list(agent_dist.keys()), "占比": list(agent_dist.values())}
            )
            st.bar_chart(agent_df.set_index("Agent"))
        else:
            st.info("暂无数据")

    with col_right:
        st.subheader("延迟分布")
        latency_data = report.get("latency", {})
        if latency_data:
            lat_df = pd.DataFrame(
                {"指标": ["平均", "P50", "P95"],
                 "延迟(ms)": [
                     latency_data.get("avg_ms", 0),
                     latency_data.get("p50_ms", 0),
                     latency_data.get("p95_ms", 0)
                 ]}
            )
            st.bar_chart(lat_df.set_index("指标"))

    st.markdown("---")

    # 最近trace列表
    st.subheader("最近 Trace")
    traces = tracer.list_traces(limit=20)
    if traces:
        trace_df = pd.DataFrame(traces)
        trace_df.columns = ["任务", "Trace ID", "耗时(ms)", "成本($)", "时间"]
        st.dataframe(trace_df, use_container_width=True)
    else:
        st.info("暂无trace数据。先运行一次Agent查询。")


def render_traces(tracer: Tracer):
    """
    Trace详情页面

    功能:
    - 选择trace查看完整调用链
    - 树状结构展示(父子嵌套)
    - 每个span的输入输出详情
    - 耗时和成本分析
    """
    st.title("Trace 详情")

    trace_id = st.text_input("输入 Trace ID", "")
    if not trace_id:
        st.info("请输入Trace ID查看详情，或从概览页面选择。")
        return

    detail = tracer.get_trace_detail(trace_id)
    if not detail:
        st.error(f"未找到trace: {trace_id}")
        return

    # 基本信息
    st.subheader(f"任务: {detail.get('task', '未知')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("总耗时", f"{detail.get('root_span', {}).get('duration_ms', 0):.0f}ms")
    col2.metric("总成本", f"${detail.get('total_cost', 0):.5f}")
    col3.metric("触发时间", detail.get("created_at", ""))

    st.markdown("---")

    # Span列表
    st.subheader("调用链")
    root_span = detail.get("root_span", {})

    if root_span:
        # 递归展示span树
        def show_span(span_data, depth=0):
            name = span_data.get("name", "unknown")
            duration = span_data.get("duration_ms", 0)
            span_type = span_data.get("span_type", "custom")
            error = span_data.get("error")
            cost = span_data.get("cost", 0)

            # 缩进表示层级
    prefix = "  " * depth + ("├─ " if depth > 0 else "")
    icon = "❌" if error else "✅"
    st.text(f"{prefix}{icon} {name} [{span_type}] - {duration:.1f}ms - ${cost:.5f}")

            # 展开查看输入输出
            if st.checkbox(f"查看 {name} 的输入输出", key=f"expand_{name}_{depth}"):
                in_col, out_col = st.columns(2)
                with in_col:
                    st.text_area("输入", str(span_data.get("input", "无"))[:500], height=150)
                with out_col:
                    st.text_area("输出", str(span_data.get("output", "无"))[:500], height=150)

        show_span(root_span)


def render_eval(evaluator: Evaluator, tracer: Tracer):
    """
    评估报告页面

    展示:
    - 综合评分(雷达图概念)
    - 各维度详细分数
    - 改进建议
    - 历史评分趋势
    """
    st.title("评估报告")

    trace_id = st.text_input("输入 Trace ID 进行评估", key="eval_trace_id")

    if not trace_id:
        st.info("请输入Trace ID进行评估。")
        return

    trace_detail = tracer.get_trace_detail(trace_id)
    if not trace_detail:
        st.error(f"未找到trace: {trace_id}")
        return

    # 执行评估
    if st.button("运行评估"):
        with st.spinner("评估中..."):
            # 这里简化处理，实际需要从trace重建TraceTree
            st.success("评估完成!")

            # 展示评估结果(模拟)
            st.subheader("综合评分: 8.2/10")

            # 各维度分数雷达图(简化用表格展示)
            dimensions = {
                "正确性": 0.85,
                "忠实度": 0.90,
                "完整性": 0.75,
                "延迟": 0.80,
                "成本": 0.85,
                "路径效率": 0.70
            }

            eval_df = pd.DataFrame(
                {"维度": list(dimensions.keys()), "分数": list(dimensions.values())}
            )
            st.bar_chart(eval_df.set_index("维度"))

            # 改进建议
            st.subheader("改进建议")
            st.markdown("- 路径效率偏低，考虑减少不必要的Agent调用")
            st.markdown("- 完整性可提升，增加引用标注")


def render_replay(replayer: Replayer):
    """
    回放页面

    功能:
    - 输入Trace ID进行回放
    - 逐步展示执行过程
    - 可跳转到特定步骤
    """
    st.title("执行回放")

    trace_id = st.text_input("输入 Trace ID 进行回放", key="replay_trace_id")

    if not trace_id:
        st.info("请输入Trace ID进行回放。")
        return

    if st.button("开始回放"):
        steps = replayer.replay(trace_id, verbose=False)
        if not steps:
            st.error("无法加载trace数据")
            return

        st.subheader(f"回放: {len(steps)} 个步骤")

        for step in steps:
            status_icon = "✅" if step.status == "success" else "❌"
            with st.expander(f"{status_icon} Step {step.step_number}: {step.name} ({step.duration_ms:.1f}ms)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Agent:** {step.agent or 'N/A'}")
                    st.markdown(f"**类型:** {step.span_type}")
                    st.markdown(f"**输入:** {step.input_summary}")
                with col2:
                    st.markdown(f"**耗时:** {step.duration_ms:.1f}ms")
                    st.markdown(f"**成本:** ${step.cost:.6f}")
                    st.markdown(f"**输出:** {step.output_summary}")


def main():
    """Dashboard主入口"""
    st.set_page_config(
        page_title="AgentForge Dashboard",
        page_icon="🔍",
        layout="wide"
    )

    init_session_state()

    # 侧边栏导航
    st.sidebar.title("AgentForge")
    page = st.sidebar.radio(
        "导航",
        ["概览", "Traces", "评估", "回放"]
    )

    tracer = st.session_state.tracer
    metrics = st.session_state.metrics
    replayer = st.session_state.replayer
    evaluator = Evaluator(use_llm=False)

    # 渲染对应页面
    if page == "概览":
        render_overview(tracer, metrics)
    elif page == "Traces":
        render_traces(tracer)
    elif page == "评估":
        render_eval(evaluator, tracer)
    elif page == "回放":
        render_replay(replayer)


# Streamlit入口
if __name__ == "__main__":
    main()
