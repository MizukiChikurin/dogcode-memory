"""语义索引 - SQLite + 可选 Embedding 的向量索引。

并发安全说明：
- SQLite 启用 WAL（Write-Ahead Logging）模式，支持读者与写者并发。
- WAL 模式下，读取不会被写入阻塞，写入也不会被读取阻塞。
- 多进程并发写入时 SQLite 内部自动序列化，无需额外锁。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
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
    """记忆索引 - 基于 SQLite 的语义索引。

    启用 WAL 模式以支持并发读写：
    - 读者不会阻塞写者
    - 写者不会阻塞读者
    - 多进程写入由 SQLite 内部自动序列化
    """

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
        # 内存中的向量索引：uri -> embedding
        self._embedding_index: dict[str, list[float]] = {}
        self._index_lock = threading.RLock()
        self._init_db()
        self._preload_embeddings()

    def _init_db(self) -> None:
        """初始化数据库表结构并启用 WAL 模式。"""
        self._ensure_dir(Path(self._db_path).parent)
        with sqlite3.connect(self._db_path) as conn:
            # 启用 WAL 模式，支持读写并发
            conn.execute("PRAGMA journal_mode=WAL")
            # 自动检查点间隔：1000 页（平衡性能与恢复速度）
            conn.execute("PRAGMA wal_autocheckpoint=1000")
            # 同步模式 NORMAL：WAL 模式下安全且更快
            conn.execute("PRAGMA synchronous=NORMAL")

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
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    uri, abstract, content,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    uri TEXT,
                    accessed_at TEXT,
                    query TEXT
                )
            """)
            conn.commit()

    def _preload_embeddings(self) -> None:
        """启动时将所有 embedding 从 SQLite 加载到内存字典。"""
        if not self._embedding_provider:
            return
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT uri, embedding_json FROM memories WHERE embedding_json IS NOT NULL")
            for row in cursor:
                uri, emb_json = row
                try:
                    embedding = json.loads(emb_json)
                    if embedding and len(embedding) == self._dimension:
                        with self._index_lock:
                            self._embedding_index[uri] = embedding
                except Exception:
                    continue

    def index_memory(
        self,
        uri: str,
        abstract: str,
        content: str,
        updated_at: str = "",
    ) -> None:
        """索引单条记忆，同步维护内存向量索引。"""
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

        # 同步更新内存索引
        with self._index_lock:
            if embedding:
                self._embedding_index[uri] = embedding
            else:
                self._embedding_index.pop(uri, None)

    def search(
        self,
        query: str,
        type_filter: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """
        搜索记忆。

        优先使用向量搜索（如果有 embedding provider），否则回退到 FTS。
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
        """搜索相似记忆 —— SimilarityProvider 协议方法。"""
        return self.search(query=query_text, type_filter=type_filter, limit=limit)

    def _vector_search(
        self,
        query: str,
        type_filter: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """向量相似度搜索（内存遍历优化版）。

        从内存中的 _embedding_index 字典遍历计算余弦相似度，
        避免每次搜索都对 SQLite 执行全表扫描。
        若内存索引为空，自动回退到 FTS 搜索。
        """
        query_embedding = self._get_or_create_embedding(query)
        if not query_embedding:
            return self._fts_search(query, type_filter, limit)

        results: list[tuple[str, float]] = []
        with self._index_lock:
            index_snapshot = dict(self._embedding_index)

        for uri, mem_embedding in index_snapshot.items():
            if type_filter and type_filter not in uri:
                continue
            if len(mem_embedding) != self._dimension:
                continue
            score = _cosine_similarity(query_embedding, mem_embedding)
            results.append((uri, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _fts_search(
        self,
        query: str,
        type_filter: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """FTS5 全文搜索（fallback）。"""
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
            results = [(row[0], max(0.0, 1.0 + row[1])) for row in cursor]

        if not results:
            with sqlite3.connect(self._db_path) as conn:
                pattern = f"%{query}%"
                cursor = conn.execute(
                    "SELECT uri FROM memories WHERE abstract LIKE ? OR content LIKE ? LIMIT ?",
                    (pattern, pattern, limit),
                )
                results = [(row[0], 0.5) for row in cursor]

        if type_filter:
            results = [(uri, score) for uri, score in results if type_filter in uri]

        return results

    def update_activity(self, uri: str, query: str = "") -> None:
        """更新记忆的访问计数。"""
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
        """从索引中移除记忆，同步清理内存向量索引。"""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM memories WHERE uri = ?", (uri,))
            conn.execute("DELETE FROM access_log WHERE uri = ?", (uri,))
            conn.commit()

        with self._index_lock:
            self._embedding_index.pop(uri, None)

    def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息。"""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(active_count) FROM memories")
            total, total_access = cursor.fetchone()

        with self._index_lock:
            index_size = len(self._embedding_index)

        return {
            "total_memories": total or 0,
            "total_access": total_access or 0,
            "has_embedding": self._embedding_provider is not None,
            "dimension": self._dimension,
            "db_path": self._db_path,
            "memory_index_size": index_size,
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
    """计算两个向量的余弦相似度（优化版）。"""
    if len(a) != len(b):
        return 0.0

    # 局部变量绑定减少属性查找开销
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a_sq += x * x
        norm_b_sq += y * y

    norm_a = norm_a_sq ** 0.5
    norm_b = norm_b_sq ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _now_iso() -> str:
    """获取当前 ISO 格式时间字符串。"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
