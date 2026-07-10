# AgentForge 🔨

> 一站式Agent开发学习平台 — 整合RAG、多Agent协作、评估与可观测性

## 项目概述

AgentForge 是一个面向学习的Python项目，覆盖Agent开发的三大核心领域：

| 模块 | 核心知识 | 文件位置 |
|------|---------|---------|
| **RAG Engine** | 文档分块、向量化、混合检索(BM25+向量)、RRF融合、重排 | `rag/` |
| **Multi-Agent Core** | 路由、DAG规划、Agent基类、Blackboard、Arbiter | `agents/` |
| **Eval & Observability** | 全链路Trace、多维度评估、指标收集、回放、Dashboard | `observability/` |

## 快速开始

### 1. 安装

```bash
git clone <repo_url>
cd AgentForge
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# 无需API Key，使用内置占位LLM
python -m agentforge
```

### 3. 编程使用

```python
from agentforge import AgentForge

# 创建实例(传入你的LLM函数)
forge = AgentForge(llm_fn=your_openai_call)

# 添加知识库
forge.add_knowledge_base("researcher", "docs/research/")

# 执行任务
result = forge.run("分析Python和Go的优劣势")
print(result.output)
print(f"Trace: {result.trace_id}")
print(f"评分: {result.overall_score}")
```

### 4. 启动Dashboard

```bash
streamlit run observability/dashboard/app.py
```

## 项目结构

```
AgentForge/
├── config.py                  # 全局配置
├── orchestrator.py            # 编排器(整合所有模块)
├── __main__.py                # 主入口 + AgentForge类
│
├── rag/                       # 模块1: RAG Engine
│   ├── models.py              # 数据模型(Document/Chunk/RAGContext)
│   ├── knowledge_base.py      # 统一知识库接口
│   ├── ingestion/             # 文档摄取
│   │   └── loaders.py         # 加载器 + 分块器
│   └── retrieval/             # 检索引擎
│       ├── vector_store.py    # 向量存储 + BM25 + 混合检索
│       └── reranker.py        # 重排器
│
├── agents/                    # 模块2: Multi-Agent Core
│   ├── state.py               # 全局状态定义
│   ├── base.py                # Agent基类 + Tool定义
│   ├── router.py              # 路由器 + 规划器 + DAG
│   ├── arbiter.py             # Blackboard + Arbiter
│   └── agents/                # 具体Agent实现
│       ├── researcher.py      # 研究员
│       ├── analyst.py         # 分析师
│       ├── writer.py          # 作家
│       └── reviewer.py        # 审查员
│
├── observability/             # 模块3: Eval & Observability
│   ├── tracer.py              # 全链路追踪
│   ├── evaluator.py           # 多维度评估
│   ├── metrics.py             # 指标收集
│   ├── replay.py              # 执行回放
│   └── dashboard/             # Streamlit面板
│       └── app.py
│
├── knowledge_bases/           # 向量数据库存储
├── traces/                    # Trace持久化存储
├── requirements.txt
└── README.md
```

## 架构设计

```
用户请求
  │
  ▼
Router ──意图识别──→ Planner ──分解──→ DAG
                                          │
              ┌───────────────────────────┤
              ▼                           ▼
        Agent A (RAG)              Agent B (RAG)
              │                           │
              ▼                           ▼
        KnowledgeBase A            KnowledgeBase B
              │                           │
              └─────────┬─────────────────┘
                        ▼
                    Blackboard
                        │
                        ▼
                     Arbiter ──→ 最终输出

全程: Tracer记录每个span → Evaluator打分 → Dashboard展示
```

## 核心概念

### RAG Engine
- **混合检索**: 向量检索(语义) + BM25(关键词) → RRF融合
- **重排漏斗**: 粗排(召回) → 精排(精确)
- **知识隔离**: 每个Agent独立知识库

### Multi-Agent Core
- **DAG编排**: 拓扑排序决定执行顺序，无依赖节点可并行
- **Blackboard**: 发布-订阅模式的共享工作区
- **Arbiter**: 置信度加权/投票/LLM裁决多种策略

### Eval & Observability
- **全链路Trace**: 树状Span结构，记录每次LLM调用/工具调用/检索
- **多维度评估**: 正确性/忠实度/延迟/成本/路径效率
- **回放调试**: 基于trace逐步回放执行过程

## 扩展方向

- [ ] 接入真实LLM API (OpenAI/Claude/本地模型)
- [ ] 替换为Chroma/Qdrant向量库
- [ ] 实现并行DAG执行
- [ ] 增加更多Agent角色
- [ ] 接入LangSmith/Phoenix实现生产级可观测性
- [ ] 添加Web搜索工具
- [ ] 实现Human-in-the-loop

## 学习路径建议

1. **第一周**: 理解RAG Engine，修改分块策略和检索参数
2. **第二周**: 理解Multi-Agent Core，尝试添加自定义Agent
3. **第三周**: 理解Observability，分析trace找出瓶颈
4. **第四周**: 整合优化，接入真实LLM完成端到端流程

## License

MIT
