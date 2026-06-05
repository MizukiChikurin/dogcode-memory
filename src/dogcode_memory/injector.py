"""记忆注入 - 会话启动时将相关记忆注入上下文。"""

from __future__ import annotations

from typing import Any

from dogcode_memory.config import MemoryConfig
from dogcode_memory.retriever import MemoryRetriever, MemoryResult


class MemoryInjector:
    """记忆注入器 - 为新会话注入历史记忆。"""

    def __init__(
        self,
        retriever: MemoryRetriever,
        config: MemoryConfig | None = None,
    ):
        """
        初始化注入器。

        Args:
            retriever: 记忆检索器
            config: 记忆配置
        """
        self._retriever = retriever
        self._config = config or MemoryConfig()

    def inject_for_session(
        self,
        session_context: str = "",
        query: str = "",
    ) -> dict[str, Any]:
        """
        为新会话注入记忆，返回结构化的注入结果。

        Args:
            session_context: 会话上下文描述（如当前项目、任务）
            query: 检索查询，空字符串时使用 session_context

        Returns:
            注入结果字典：
            {
                "text": "格式化的记忆注入文本",
                "uris": ["user/profile.md", ...],
                "count": 3,
                "total_tokens": 120,
            }
            无相关记忆时返回 {"text": "", "uris": [], "count": 0, "total_tokens": 0}
        """
        search_query = query or session_context
        if not search_query:
            return {"text": "", "uris": [], "count": 0, "total_tokens": 0}

        # 检索相关记忆
        memories = self._retriever.retrieve(
            query=search_query,
            limit=10,
            token_budget=self._config.injection_token_budget,
        )

        if not memories:
            return {"text": "", "uris": [], "count": 0, "total_tokens": 0}

        # 在预算内选择最相关的记忆
        selected = self._select_relevant_memories(memories, self._config.injection_token_budget)

        if not selected:
            return {"text": "", "uris": [], "count": 0, "total_tokens": 0}

        text = self._format_memory_prompt(selected)
        uris = [m.uri for m in selected]
        total_tokens = self._estimate_injection_tokens(text)

        return {
            "text": text,
            "uris": uris,
            "count": len(selected),
            "total_tokens": total_tokens,
        }

    def _select_relevant_memories(
        self,
        memories: list[MemoryResult],
        token_budget: int,
    ) -> list[MemoryResult]:
        """
        在 token 预算内选择最相关的记忆。

        简单贪心选择：按分数排序，依次加入直到预算耗尽。
        """
        selected: list[MemoryResult] = []
        used_tokens = 0
        chars_per_token = 3.0

        for memory in memories:
            content = memory.overview if memory.level <= 1 else memory.content
            estimated_tokens = int(len(content) / chars_per_token)

            if used_tokens + estimated_tokens <= token_budget or not selected:
                selected.append(memory)
                used_tokens += estimated_tokens
            else:
                break

        return selected

    def _format_memory_prompt(self, memories: list[MemoryResult]) -> str:
        """将记忆格式化为 System Prompt 增量。"""
        parts = ["## 历史记忆", ""]

        for memory in memories:
            parts.append(f"### {memory.abstract}")
            content = memory.overview if memory.level <= 1 else memory.content
            if content:
                parts.append(content)
            parts.append("")

        return "\n".join(parts)

    def _estimate_injection_tokens(self, prompt: str) -> int:
        """估算注入内容的 token 数。"""
        return len(prompt) // 3
