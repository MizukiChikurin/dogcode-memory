"""语义索引 - SQLite + 可选 Embedding 的向量索引。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Embedding 提供者协议。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表。"""
        ...


@dataclass
class MemoryRecord:
    """索引记录模型。"""

    uri: str
    abstract: str
    content: str
    active_count: int
    updated_at: str
    embedding: list[float] | None = None


class MemoryIndex:
    """记忆索引 - 基于 SQLite 的语义索引。"""

    def __init__(
        self,
        db_path: str,
        embedding_provider: EmbeddingProvider | None = None,
        dimension: int = 1536,
    ):
        """
        初始化索引。

        Args:
            db_path: SQLite 数据库文件路径
            embedding_provider: Embedding 提供者，None 时仅使用 FTS 全文搜索
            dimension: 向量维度
        """
        self._db_path = db_path
        self._embedding_provider = embedding_provider
        self._dimension = dimension
        self._embedding_cache: dict[str, list[float]] = {}
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表结构。"""
        self._ensure_dir(Path(self._db_path).parent)
        with sqlite3.connect(self._db_path) as conn:
            # 主表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    uri TEXT PRIMARY KEY,
                    abstract TEXT,
                    content TEXT,
                    active_count INTEGER DEFAULT 0,
                    updated_at TEXT,
                    embedding_json TEXT
                )
            """)
            # FTS5 全文搜索表（用于 fallback）
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    uri, abstract, content,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
            # 触发器保持 FTS 同步
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, uri, abstract, content)
                    VALUES (new.rowid, new.uri, new.abstract, new.content);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, uri, abstract, content)
                    VALUES ('delete', old.rowid, old.uri, old.abstract, old.content);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, uri, abstract, content)
                    VALUES ('delete', old.rowid, old.uri, old.abstract, old.content);
                    INSERT INTO memories_fts(rowid, uri, abstract, content)
                    VALUES (new.rowid, new.uri, new.abstract, new.content);
                END
            """)
            # 访问日志表（用于热度计算）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    uri TEXT,
                    accessed_at TEXT,
                    query TEXT
                )
            """)
            conn.commit()

    def index_memory(
        self,
        uri: str,
        abstract: str,
        content: str,
        updated_at: str = "",
    ) -> None:
        """
        索引单条记忆。

        Args:
            uri: 记忆 URI
            abstract: 摘要文本
            content: 完整内容
            updated_at: 更新时间
        """
        embedding = None
        if self._embedding_provider:
            text = f"{abstract} {content[:500]}".strip()
            embedding = self._get_or_create_embedding(text)

        embedding_json = json.dumps(embedding) if embedding else None

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO memories (uri, abstract, content, active_count, updated_at, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(uri) DO UPDATE SET
                    abstract=excluded.abstract,
                    content=excluded.content,
                    updated_at=excluded.updated_at,
                    embedding_json=excluded.embedding_json
                """,
                (uri, abstract, content, 0, updated_at or _now_iso(), embedding_json),
            )
            conn.commit()

    def search(
        self,
        query: str,
        type_filter: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """
        搜索记忆。

        优先使用向量搜索（如果有 embedding provider），否则回退到 FTS。

        Args:
            query: 查询文本
            type_filter: 可选的类型过滤（通过 URI 前缀匹配）
            limit: 返回数量限制

        Returns:
            (uri, score) 列表，按相关性排序
        """
        if self._embedding_provider:
            return self._vector_search(query, type_filter, limit)
        return self._fts_search(query, type_filter, limit)

    def search_similar(
        self,
        query_text: str,
        type_filter: str | None = None,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """
        搜索相似记忆 —— `SimilarityProvider` 协议方法。

        是 `search()` 的别名，参数签名与 `SimilarityProvider` 保持一致，
        供 `MemoryDeduplicator` 在去重阶段调用。

        Args:
            query_text: 查询文本
            type_filter: 可选的类型过滤
            limit: 返回数量限制

        Returns:
            (uri, score) 列表，按相关性排序
        """
        return self.search(query=query_text, type_filter=type_filter, limit=limit)

    def _vector_search(
        self,
        query: str,
        type_filter: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """向量相似度搜索。"""
        query_embedding = self._get_or_create_embedding(query)
        if not query_embedding:
            return self._fts_search(query, type_filter, limit)

        results: list[tuple[str, float]] = []
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT uri, abstract, content, embedding_json FROM memories")
            for row in cursor:
                uri, abstract, content, emb_json = row
                if type_filter and type_filter not in uri:
                    continue
                if not emb_json:
                    continue
                try:
                    mem_embedding = json.loads(emb_json)
                    score = _cosine_similarity(query_embedding, mem_embedding)
                    results.append((uri, score))
                except Exception:
                    continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _fts_search(
        self,
        query: str,
        type_filter: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """FTS5 全文搜索（fallback）。"""
        # 转义 FTS 特殊字符
        safe_query = query.replace('"', '""')
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT memories.uri, rank
                FROM memories_fts
                JOIN memories ON memories.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
            )
            results = [(row[0], max(0.0, 1.0 + row[1])) for row in cursor]  # rank 为负数

        # 如果 FTS 无结果，尝试 LIKE 模糊匹配
        if not results:
            with sqlite3.connect(self._db_path) as conn:
                pattern = f"%{query}%"
                cursor = conn.execute(
                    "SELECT uri FROM memories WHERE abstract LIKE ? OR content LIKE ? LIMIT ?",
                    (pattern, pattern, limit),
                )
                results = [(row[0], 0.5) for row in cursor]

        # 应用类型过滤
        if type_filter:
            results = [(uri, score) for uri, score in results if type_filter in uri]

        return results

    def update_activity(self, uri: str, query: str = "") -> None:
        """
        更新记忆的访问计数。

        Args:
            uri: 被访问的记忆 URI
            query: 触发访问的查询文本（可选）
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE memories SET active_count = active_count + 1 WHERE uri = ?",
                (uri,),
            )
            conn.execute(
                "INSERT INTO access_log (uri, accessed_at, query) VALUES (?, ?, ?)",
                (uri, _now_iso(), query),
            )
            conn.commit()

    def remove(self, uri: str) -> None:
        """从索引中移除记忆。"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM memories WHERE uri = ?", (uri,))
            conn.execute("DELETE FROM access_log WHERE uri = ?", (uri,))
            conn.commit()

    def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息。"""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(active_count) FROM memories")
            total, total_access = cursor.fetchone()
            return {
                "total_memories": total or 0,
                "total_access": total_access or 0,
                "has_embedding": self._embedding_provider is not None,
                "dimension": self._dimension,
                "db_path": self._db_path,
            }

    def _get_or_create_embedding(self, text: str) -> list[float] | None:
        """获取或创建文本的 embedding，带缓存。"""
        if not self._embedding_provider:
            return None

        cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            embeddings = self._embedding_provider.embed([text])
            if embeddings and embeddings[0]:
                embedding = embeddings[0]
                self._embedding_cache[cache_key] = embedding
                return embedding
        except Exception:
            pass
        return None

    @staticmethod
    def _ensure_dir(directory: Path) -> None:
        """确保目录存在。"""
        directory.mkdir(parents=True, exist_ok=True)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _now_iso() -> str:
    """获取当前 ISO 格式时间字符串。"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
