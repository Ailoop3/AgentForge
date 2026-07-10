"""
AgentForge - 3-in-1 Agent Learning Platform

整合三大核心模块：
1. RAG Engine - 知识检索增强
2. Multi-Agent Core - 多智能体协作
3. Eval & Observability - 评估与可观测性

架构设计原则：
- 每个模块可独立使用，也可组合使用
- 全链路可观测（每个操作都被追踪）
- 模块化设计，便于扩展和替换组件
"""

# 全局配置类 - 管理所有模块的配置参数
class Settings:
    """
    集中管理整个平台的配置。
    实际项目中可从环境变量或配置文件加载。
    """
    # 模型配置
    LLM_MODEL: str = "gpt-4o-mini"          # 默认使用的LLM模型
    LLM_TEMPERATURE: float = 0.7            # 生成温度（控制随机性）
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # 向量嵌入模型

    # RAG配置
    CHUNK_SIZE: int = 512                   # 文本分块大小(token数)
    CHUNK_OVERLAP: int = 64                 # 块之间重叠量(保持语义连贯)
    RETRIEVAL_TOP_K: int = 5                # 默认检索返回的文档数
    RERANK_TOP_K: int = 3                   # 重排后最终保留的文档数

    # 向量数据库配置
    VECTOR_DB_PATH: str = "./knowledge_bases"  # 向量数据库存储路径

    # 多Agent配置
    MAX_AGENT_ITERATIONS: int = 10          # 单个Agent最大迭代次数(防止无限循环)
    AGENT_TIMEOUT: int = 120                # 单次Agent执行超时(秒)

    # 可观测性配置
    TRACE_STORAGE_PATH: str = "./traces"    # Trace数据存储路径
    ENABLE_COST_TRACKING: bool = True       # 是否追踪API调用成本

    # 成本参考(美元/1K tokens) - 用于计算调用成本
    COST_PER_1K_TOKENS: dict = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "text-embedding-3-small": {"input": 0.00002, "output": 0},
    }


# 全局单例配置，所有模块共享
settings = Settings()
