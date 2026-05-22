"""DogCode 记忆模块 - 跨会话知识持久化系统。

该模块提供完整的记忆管理功能，包括：
- 记忆类型定义（YAML 模板驱动）
- 文件系统存储（Markdown + YAML 元数据头）
- 记忆提取管线（LLM 辅助提取、去重、合并）
- 语义索引与检索（SQLite + Embedding）
- 热度生命周期管理
- 会话启动记忆注入

设计原则：
1. 自用优先，MVP 级别实现
2. 零外部依赖（除 PyYAML），Embedding 可配置
3. 清晰的接口边界，便于集成到 ReuleauxCoder-ezcode
"""

from __future__ import annotations

from dogcode_memory.config import MemoryConfig, ContextConfig
from dogcode_memory.schema import MemoryField, MemoryTypeSchema, MergeOp, OperationMode
from dogcode_memory.registry import MemoryTypeRegistry
from dogcode_memory.store import MemoryStore
from dogcode_memory.format import serialize_memory, deserialize_memory
from dogcode_memory.merge import merge_memory, patch_merge, sum_merge, immutable_merge, append_merge
from dogcode_memory.extractor import MemoryExtractor, CandidateMemory
from dogcode_memory.deduplicator import MemoryDeduplicator, DedupDecision
from dogcode_memory.updater import MemoryUpdater, MemoryOperation
from dogcode_memory.index import MemoryIndex, MemoryRecord
from dogcode_memory.retriever import MemoryRetriever, MemoryResult
from dogcode_memory.lifecycle import hotness_score, MemoryArchiver
from dogcode_memory.injector import MemoryInjector
from dogcode_memory.pipeline import MemoryPipeline
from dogcode_memory.adapter import (
    ReuleauxLLMAdapter,
    ReuleauxEmbeddingAdapter,
    adapt_llm,
)

__all__ = [
    # 配置
    "MemoryConfig",
    "ContextConfig",
    # Schema
    "MemoryField",
    "MemoryTypeSchema",
    "MergeOp",
    "OperationMode",
    # 注册表
    "MemoryTypeRegistry",
    # 存储
    "MemoryStore",
    # 格式
    "serialize_memory",
    "deserialize_memory",
    # 合并
    "merge_memory",
    "patch_merge",
    "sum_merge",
    "immutable_merge",
    "append_merge",
    # 提取
    "MemoryExtractor",
    "CandidateMemory",
    # 去重
    "MemoryDeduplicator",
    "DedupDecision",
    # 更新
    "MemoryUpdater",
    "MemoryOperation",
    # 索引
    "MemoryIndex",
    "MemoryRecord",
    # 检索
    "MemoryRetriever",
    "MemoryResult",
    # 生命周期
    "hotness_score",
    "MemoryArchiver",
    # 注入
    "MemoryInjector",
    # 管线
    "MemoryPipeline",
    # 适配器
    "ReuleauxLLMAdapter",
    "ReuleauxEmbeddingAdapter",
    "adapt_llm",
]
