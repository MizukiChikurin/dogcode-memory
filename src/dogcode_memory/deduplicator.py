"""去重模块 - 向量预过滤 + LLM 决策的两阶段去重。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dogcode_memory.config import MemoryConfig
from dogcode_memory.extractor import CandidateMemory


class SimilarityProvider(Protocol):
    """相似度提供者协议。"""

    def search_similar(
        self,
        query_text: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """搜索相似记忆，返回 (uri, score) 列表。"""
        ...


@dataclass
class DedupDecision:
    """去重决策结果。"""

    action: str   # "skip", "create", "merge"
    target_uri: str = ""   # merge 时的目标 URI
    reason: str = ""       # 决策理由

    def to_dict(self) -> dict[str, str]:
        """序列化为字典。"""
        return {
            "action": self.action,
            "target_uri": self.target_uri,
            "reason": self.reason,
        }


DEDUP_SYSTEM_PROMPT = """\
You are a memory deduplication system. Decide whether a candidate memory should be created, merged, or skipped.

Given:
1. A candidate memory (newly extracted)
2. A list of existing similar memories

For each existing memory, decide one of:
- skip: The candidate is fully covered by an existing memory, no action needed
- create: The candidate is genuinely new and should be created as a new memory
- merge: The candidate contains new information that should be merged into an existing memory

Return a JSON object:
{
  "decision": "skip|create|merge",
  "target_uri": "uri of existing memory (for merge)",
  "reason": "brief explanation"
}

Rules:
- If candidate and existing are ~90%+ similar in meaning → skip
- If candidate adds meaningful new details to existing → merge
- If candidate is about a different topic/entity → create
- Prefer merge over create when topics overlap
"""


class MemoryDeduplicator:
    """记忆去重器 - 两阶段去重（向量预过滤 + LLM 决策）。"""

    def __init__(
        self,
        similarity_provider: SimilarityProvider | None = None,
        llm: Any | None = None,
        config: MemoryConfig | None = None,
    ):
        """
        初始化去重器。

        Args:
            similarity_provider: 相似度搜索提供者
            llm: LLM 客户端，用于决策
            config: 记忆配置
        """
        self._similarity = similarity_provider
        self._llm = llm
        self._config = config or MemoryConfig()

    def deduplicate(
        self,
        candidate: CandidateMemory,
        existing_uris: list[str] | None = None,
    ) -> DedupDecision:
        """
        对单个候选执行去重决策。

        Args:
            candidate: 候选记忆
            existing_uris: 可选的已有记忆 URI 列表（若提供则跳过向量搜索）

        Returns:
            去重决策
        """
        # 阶段一：查找相似记忆
        similar = self._find_similar(candidate)

        if not similar:
            # 无相似记忆，直接创建
            return DedupDecision(
                action="create",
                reason="No similar memories found",
            )

        # 阶段二：LLM 决策
        if self._llm:
            return self._llm_decide(candidate, similar)

        # 无 LLM 时，使用简单的相似度阈值决策
        best_uri, best_score = similar[0]
        if best_score >= self._config.dedup_similarity_threshold:
            return DedupDecision(
                action="merge",
                target_uri=best_uri,
                reason=f"High similarity score: {best_score:.3f}",
            )
        return DedupDecision(
            action="create",
            reason=f"Best similarity score too low: {best_score:.3f}",
        )

    def _find_similar(
        self,
        candidate: CandidateMemory,
    ) -> list[tuple[str, float]]:
        """查找与候选记忆相似的记忆。"""
        if not self._similarity:
            return []

        query = f"{candidate.abstract} {candidate.overview}".strip()
        if not query:
            return []

        try:
            return self._similarity.search_similar(
                query_text=query,
                type_filter=candidate.type,
                limit=5,
            )
        except Exception:
            return []

    def _llm_decide(
        self,
        candidate: CandidateMemory,
        similar_memories: list[tuple[str, float]],
    ) -> DedupDecision:
        """使用 LLM 决策去重策略。"""
        try:
            # 构造决策提示
            candidate_text = f"Type: {candidate.type}\nAbstract: {candidate.abstract}\nOverview: {candidate.overview}\nContent: {candidate.content[:500]}"
            similar_text = "\n\n".join([
                f"URI: {uri}\nRelevance: {score:.3f}"
                for uri, score in similar_memories[:3]
            ])

            prompt = f"""Candidate memory:
{candidate_text}

Similar existing memories:
{similar_text}

Decide: skip, create, or merge?"""

            resp = self._llm.chat(
                messages=[
                    {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.content if hasattr(resp, "content") else str(resp)

            # 解析 JSON 决策
            import json
            # 尝试提取 JSON
            decision_data = self._parse_json_decision(content)
            if decision_data:
                return DedupDecision(
                    action=decision_data.get("decision", "create"),
                    target_uri=decision_data.get("target_uri", ""),
                    reason=decision_data.get("reason", "LLM decision"),
                )
        except Exception:
            pass

        # 降级为相似度阈值
        best_uri, best_score = similar_memories[0]
        if best_score >= self._config.dedup_similarity_threshold:
            return DedupDecision(action="merge", target_uri=best_uri, reason="Fallback: high similarity")
        return DedupDecision(action="create", reason="Fallback: low similarity")

    @staticmethod
    def _parse_json_decision(text: str) -> dict[str, str] | None:
        """从文本中解析 JSON 决策。"""
        import re
        text = text.strip()

        # 尝试直接解析
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                pass

        # 从代码块中提取
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        # 查找花括号
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass

        return None
