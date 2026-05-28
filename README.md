# dogcode-memory

DogCode 记忆模块 - 跨会话知识持久化系统。

## 概述

`dogcode-memory` 是 DogCode 的记忆管理子系统，解决以下问题：

- **跨会话知识丢失**：每次会话独立，编码偏好、项目上下文无法继承
- **上下文恢复缓慢**：长会话需要重放所有消息
- **无语义检索**：无法按语义相关性检索历史信息

## 架构

```
dogcode_memory/
├── config.py          # MemoryConfig, ContextConfig
├── schema.py          # 记忆类型 Schema 定义
├── registry.py        # YAML 模板注册表
├── store.py           # 文件系统存储
├── format.py          # Markdown + YAML 序列化/反序列化
├── merge.py           # 字段级合并策略（PATCH/SUM/IMMUTABLE/APPEND）
├── extractor.py       # LLM 辅助记忆提取
├── deduplicator.py    # 两阶段去重（向量预过滤 + LLM 决策）
├── updater.py         # 写入/编辑/删除操作执行
├── index.py           # SQLite + Embedding 语义索引
├── retriever.py       # L0/L1/L2 分层检索 + 热度加权
├── lifecycle.py       # 热度评分 + 冷归档
├── injector.py        # 会话启动记忆注入
├── pipeline.py        # 完整生命周期管线编排
└── schemas/           # 内置 YAML 模板
    ├── profile.yaml
    ├── preferences.yaml
    ├── project.yaml
    ├── tools.yaml
    ├── patterns.yaml
    └── errors.yaml
```

## 快速开始

```python
from dogcode_memory import MemoryPipeline

# 创建管线
pipeline = MemoryPipeline.create(
    storage_dir="~/.dogcode/memories",
    llm=your_llm_client,              # 可选
    embedding_provider=your_embedder,  # 可选
)

# 会话启动：注入记忆
memory_prompt = pipeline.on_session_start(
    session_id="session_123",
    context="Working on DogCode project using Python"
)
# 将 memory_prompt 追加到 System Prompt

# 会话结束：提取记忆
stats = pipeline.on_session_end(
    session_id="session_123",
    messages=conversation_messages,
)
print(f"提取: {stats['extracted']}, 创建: {stats['created']}, 合并: {stats['merged']}")

# 维护：归档冷记忆
maintenance = pipeline.run_maintenance()
```

## 存储结构

```
~/.dogcode/memories/
  user/
    profile.md
    preferences/
      naming-style.md
    projects/
      dogcode.md
  agent/
    tools/
      bash.md
    patterns/
      error-handling.md
    errors/
      import-error.md
  .index.db          # SQLite 索引数据库
```

## 依赖

- Python >= 3.10
- PyYAML >= 6.0


