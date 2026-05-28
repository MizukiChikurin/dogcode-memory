"""上下文管理子模块 - 会话归档、摘要缓存与恢复。

该模块实现架构文档中 Phase 1 的短期上下文管理增强：
- 消息摘要缓存，避免重复调 LLM
- 会话归档：将旧消息移入归档存储
- 会话恢复：从归档加载摘要而非完整消息
- 归档压缩策略（Layer 2.5）

使用方式：
    from dogcode_memory.context import SessionArchiver, SessionRestorer, SummaryCache
"""

from __future__ import annotations

from dogcode_memory.context.archiver import ArchiveRecord, ArchiveWriter, SessionArchiver
from dogcode_memory.context.compression_enhanced import ArchiveCompressStrategy, generate_structured_summary
from dogcode_memory.context.restorer import SessionRestorer
from dogcode_memory.context.summary_cache import SummaryCache, compute_message_hash

__all__ = [
    "ArchiveRecord",
    "ArchiveWriter",
    "SessionArchiver",
    "SessionRestorer",
    "SummaryCache",
    "compute_message_hash",
    "ArchiveCompressStrategy",
    "generate_structured_summary",
]
