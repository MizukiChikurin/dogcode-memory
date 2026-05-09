"""分层检索 - L0/L1/L2 三级检索与热度加权排序。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dogcode_memory.config import MemoryConfig
from dogcode_memory.format import deserialize_memory
from dogcode_memory.index import MemoryIndex
from dogcode_memory.lifecycle import hotness_score
from dogcode_memory.store import MemoryStore


@dataclass
class MemoryResult:
    """检索结果。"""

    uri: str
    level: int          # 0=L0 摘要, 1=L1 概览, 2=L2 详情
    abstract: str       # L0 内容
    overview: str       # L1 内容
    content: str        # L2 内容
    score: float        # 综合得分
    hotness: float = 0.0  # 热度得分

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "uri": self.uri,
            "level": self.level,
            "abstract": self.abstract,
            "overview": self.overview,
            "content": self.content[:500] if self.content else "",
            "score": self.score,
            "hotness": self.hotness,
        }


class MemoryRetriever:
    """记忆检索器 - 分层检索与热度加权。"""

    def __init__(
        self,
        store: MemoryStore,
        index: MemoryIndex,
        registry: Any | None = None,
        config: MemoryConfig | None = None,
    ):
        """
        初始化检索器。

        Args:
            store: 记忆存储
            index: 记忆索引
            registry: 记忆类型注册表（可选）
            config: 记忆配置
        """
        self._store = store
        self._index = index
        self._registry = registry
        self._config = config or MemoryConfig()

    def retrieve(
        self,
        query: str,
        types: list[str] | None = None,
        limit: int = 10,
        token_budget: int = 2000,
    ) -> list[MemoryResult]:
        """
        分层检索入口。

        检索流程：
        1. L0 检索：获取候选 URI 列表（仅摘要）
        2. L1 检索：读取概览层内容
        3. L2 检索：按需读取完整内容
        4. 热度加权排序

        Args:
            query: 查询文本
            types: 记忆类型过滤列表
            limit: 返回结果数量上限
            token_budget: Token 预算（用于控制 L2 加载量）

        Returns:
            检索结果列表
        """
        # L0: 获取候选 URI
        l0_results = self._retrieve_l0(query, types, limit * 3)
        if not l0_results:
            return []

        # L1: 加载概览
        l1_results = self._retrieve_l1(l0_results, limit * 2)

        # L2: 按需加载完整内容（在 token 预算内）
        l2_results = self._retrieve_l2(l1_results, token_budget, limit)

        # 热度加权排序
        ranked = self._score_and_rank(l2_results, query)
        return ranked[:limit]

    def retrieve_by_uri(self, uri: str) -> MemoryResult | None:
        """按 URI 直接读取完整记忆。"""
        content = self._store.read(uri)
        if not content:
            return None

        memory = deserialize_memory(content)
        self._index.update_activity(uri)

        return MemoryResult(
            uri=uri,
            level=2,
            abstract=memory.type or uri,
            overview=memory.content[:200] if memory.content else "",
            content=memory.content,
            score=1.0,
        )

    def _retrieve_l0(
        self,
        query: str,
        types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """
        L0 检索：返回候选 URI 及语义分数。

        使用索引的向量/FTS 搜索。
        """
        # 如果没有类型过滤，直接搜索
        if not types:
            return self._index.search(query, limit=limit)

        # 有类型过滤时，对每个类型搜索并合并
        all_results: list[tuple[str, float]] = []
        for type_name in types:
            results = self._index.search(query, type_filter=type_name, limit=limit)
            all_results.extend(results)

        # 去重并保留最高分数
        seen: dict[str, float] = {}
        for uri, score in all_results:
            if uri not in seen or seen[uri] < score:
                seen[uri] = score

        sorted_results = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    def _retrieve_l1(
        self,
        l0_results: list[tuple[str, float]],
        limit: int,
    ) -> list[MemoryResult]:
        """
        L1 检索：加载概览层。

        读取记忆文件，提取摘要和概览。
        """
        results: list[MemoryResult] = []
        for uri, semantic_score in l0_results[:limit]:
            content = self._store.read(uri)
            if not content:
                continue

            memory = deserialize_memory(content)
            overview = memory.content[:400] if memory.content else memory.abstract

            results.append(MemoryResult(
                uri=uri,
                level=1,
                abstract=memory.abstract or memory.type or uri,
                overview=overview,
                content=memory.content,
                score=semantic_score,
            ))

        return results

    def _retrieve_l2(
        self,
        l1_results: list[MemoryResult],
        token_budget: int,
        limit: int,
    ) -> list[MemoryResult]:
        """
        L2 检索：在 token 预算内加载完整内容。

        简单估算：每字符约 0.3 token。
        """
        results: list[MemoryResult] = []
        used_tokens = 0
        chars_per_token = 3.0

        for result in l1_results[:limit]:
            content_len = len(result.content) if result.content else 0
            estimated_tokens = int(content_len / chars_per_token)

            if used_tokens + estimated_tokens > token_budget and results:
                # 预算不足，停止加载完整内容
                # 但仍保留 L1 级别的结果
                result.level = 1
                result.content = ""
            else:
                result.level = 2
                used_tokens += estimated_tokens
                # 更新访问计数
                self._index.update_activity(result.uri)

            results.append(result)

        return results

    def _score_and_rank(
        self,
        results: list[MemoryResult],
        query: str,
    ) -> list[MemoryResult]:
        """
        语义分数 × 热度加权排序。

        final_score = 0.8 * semantic_score + 0.2 * hotness_score
        """
        scored_results = []
        for result in results:
            # 计算热度
            hotness = self._calculate_hotness(result.uri)
            result.hotness = hotness
            result.score = 0.8 * result.score + 0.2 * hotness
            scored_results.append(result)

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results

    def _calculate_hotness(self, uri: str) -> float:
        """计算单条记忆的热度。"""
        # 从索引获取访问数据
        try:
            import sqlite3
            with sqlite3.connect(self._index._db_path) as conn:
                cursor = conn.execute(
                    "SELECT active_count, updated_at FROM memories WHERE uri = ?",
                    (uri,),
                )
                row = cursor.fetchone()
                if row:
                    active_count, updated_at = row
                    return hotness_score(
                        active_count=active_count or 0,
                        updated_at=updated_at or "",
                        half_life_days=self._config.hotness_half_life_days,
                    )
        except Exception:
            pass
        return 0.5  # 默认值
