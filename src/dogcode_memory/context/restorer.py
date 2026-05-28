"""会话恢复器 - 从归档和记忆中组装恢复上下文。

恢复流程：
1. 加载会话的所有归档摘要
2. 检索相关长期记忆
3. 在 token 预算内组装上下文
4. 返回恢复后的消息列表（由调用方应用到 Agent）

注意：本模块不直接修改服务端 Agent 对象，只返回可注入的消息列表，
由上层（Pipeline 或调用方）决定如何应用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from dogcode_memory.context.archiver import ArchiveRecord, SessionArchiver


class MemoryRetrieverLike(Protocol):
    """记忆检索器协议。"""

    def retrieve(
        self,
        query: str,
        types: list[str] | None = None,
        limit: int = 10,
        token_budget: int = 2000,
    ) -> list[Any]:
        """检索相关记忆。"""
        ...


@dataclass
class RestoreResult:
    """会话恢复结果。"""

    messages: list[dict[str, Any]] = field(default_factory=list)
    archive_records_used: list[ArchiveRecord] = field(default_factory=list)
    memory_uris_used: list[str] = field(default_factory=list)
    total_tokens: int = 0
    info: dict[str, Any] = field(default_factory=dict)


class SessionRestorer:
    """会话恢复器。"""

    def __init__(
        self,
        archiver: SessionArchiver,
        token_budget: int = 8000,
        archive_budget_ratio: float = 0.5,
        memory_budget_ratio: float = 0.25,
    ):
        """
        初始化恢复器。

        Args:
            archiver: 会话归档器
            token_budget: 恢复上下文总 token 预算
            archive_budget_ratio: 分配给归档摘要的预算比例
            memory_budget_ratio: 分配给记忆注入的预算比例
        """
        self._archiver = archiver
        self._token_budget = token_budget
        self._archive_budget = int(token_budget * archive_budget_ratio)
        self._memory_budget = int(token_budget * memory_budget_ratio)

    def restore(
        self,
        session_id: str,
        query: str = "",
        memory_retriever: MemoryRetrieverLike | None = None,
    ) -> RestoreResult:
        """
        恢复会话上下文。

        Args:
            session_id: 会话 ID
            query: 检索记忆的查询文本，空字符串时使用归档摘要拼接
            memory_retriever: 可选的记忆检索器

        Returns:
            恢复结果，包含组装好的消息列表
        """
        # 1. 加载归档摘要
        archive_text, archive_records = self._archiver.build_restore_context(
            session_id=session_id,
            token_budget=self._archive_budget,
        )

        # 2. 检索相关记忆
        memory_messages: list[dict[str, Any]] = []
        memory_uris: list[str] = []
        if memory_retriever and query:
            memory_messages, memory_uris = self._retrieve_memories(
                query=query,
                retriever=memory_retriever,
            )

        # 3. 组装上下文
        context_messages = self._assemble_context(
            archive_text=archive_text,
            memory_messages=memory_messages,
        )

        total_tokens = _estimate_tokens(context_messages)

        return RestoreResult(
            messages=context_messages,
            archive_records_used=archive_records,
            memory_uris_used=memory_uris,
            total_tokens=total_tokens,
            info={
                "session_id": session_id,
                "archive_text_length": len(archive_text),
                "archive_records_count": len(archive_records),
                "memory_messages_count": len(memory_messages),
                "query": query,
            },
        )

    def restore_with_active_messages(
        self,
        session_id: str,
        active_messages: list[dict[str, Any]],
        query: str = "",
        memory_retriever: MemoryRetrieverLike | None = None,
    ) -> RestoreResult:
        """
        恢复会话上下文，并与当前活跃消息合并。

        返回的消息列表格式：
        [归档摘要消息, 记忆注入消息, 活跃消息...]

        Args:
            session_id: 会话 ID
            active_messages: 当前活跃消息（如从 SessionStore 加载的）
            query: 检索查询
            memory_retriever: 可选的记忆检索器

        Returns:
            恢复结果
        """
        base = self.restore(session_id, query, memory_retriever)

        # 合并活跃消息
        all_messages = list(base.messages)

        # 去重：避免归档摘要和活跃消息中重复
        # 简单策略：如果活跃消息开头与归档摘要内容重叠，跳过前几条
        skip_count = self._count_overlap(all_messages, active_messages)
        merged = all_messages + active_messages[skip_count:]

        total_tokens = _estimate_tokens(merged)

        return RestoreResult(
            messages=merged,
            archive_records_used=base.archive_records_used,
            memory_uris_used=base.memory_uris_used,
            total_tokens=total_tokens,
            info={
                **base.info,
                "active_messages_count": len(active_messages),
                "skip_count": skip_count,
            },
        )

    def _retrieve_memories(
        self,
        query: str,
        retriever: MemoryRetrieverLike,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        检索记忆并格式化为消息。

        Returns:
            (消息列表, 使用的 URI 列表)
        """
        try:
            results = retriever.retrieve(
                query=query,
                limit=10,
                token_budget=self._memory_budget,
            )
        except Exception:
            return [], []

        if not results:
            return [], []

        parts = ["## 历史记忆\n"]
        uris: list[str] = []
        for result in results:
            uris.append(getattr(result, "uri", "") or "")
            abstract = getattr(result, "abstract", "") or ""
            content = getattr(result, "overview", "") or getattr(result, "content", "") or ""
            parts.append(f"### {abstract}\n{content}\n")

        memory_msg = {
            "role": "user",
            "content": "\n".join(parts),
        }
        assistant_ack = {
            "role": "assistant",
            "content": "Got it, I have the historical context.",
        }
        return [memory_msg, assistant_ack], uris

    def _assemble_context(
        self,
        archive_text: str,
        memory_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """组装恢复上下文消息列表。"""
        messages: list[dict[str, Any]] = []

        if archive_text:
            messages.append({
                "role": "user",
                "content": f"[Previous session context]\n{archive_text}",
            })
            messages.append({
                "role": "assistant",
                "content": "I have the context from previous sessions.",
            })

        messages.extend(memory_messages)
        return messages

    @staticmethod
    def _count_overlap(
        prefix_messages: list[dict[str, Any]],
        active_messages: list[dict[str, Any]],
    ) -> int:
        """
        估算前缀消息与活跃消息的重叠数量。

        简单启发式：如果 active_messages 前几条是归档摘要的重复内容，
        则跳过这些重复项。
        """
        if not prefix_messages or not active_messages:
            return 0

        # 检查活跃消息前两条是否与归档摘要重复
        skip = 0
        for i in range(min(2, len(active_messages))):
            active_role = active_messages[i].get("role", "")
            active_content = active_messages[i].get("content", "") or ""
            for pm in prefix_messages:
                if pm.get("role") == active_role:
                    pm_content = pm.get("content", "") or ""
                    # 如果内容相似度超过 80%，认为是重复
                    if _text_similarity(pm_content, active_content) > 0.8:
                        skip += 1
                        break
        return skip


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """粗略估算消息列表的 token 数。"""
    total_chars = 0
    for m in messages:
        content = m.get("content", "") or ""
        total_chars += len(content)
    return int(total_chars / 3.0)


def _text_similarity(a: str, b: str) -> float:
    """简单文本相似度（基于公共子串比例）。"""
    if not a or not b:
        return 0.0
    a_norm = a.lower().strip()
    b_norm = b.lower().strip()
    if a_norm == b_norm:
        return 1.0

    # 使用简单字符集合重叠度
    set_a = set(a_norm)
    set_b = set(b_norm)
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
