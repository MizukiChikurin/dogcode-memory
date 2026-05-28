"""管线编排 - 将记忆模块组件编排为完整的会话生命周期管线。

并发安全说明：
- on_session_end 中的写操作受 MemoryStore 文件锁保护，
  确保多会话同时结束不会导致文件写入冲突。
- SQLite 索引通过 WAL 模式天然支持读写并发，无需额外锁。
"""

from __future__ import annotations

from typing import Any

from dogcode_memory.config import MemoryConfig
from dogcode_memory.context.archiver import SessionArchiver
from dogcode_memory.context.compression_enhanced import generate_structured_summary
from dogcode_memory.context.restorer import RestoreResult, SessionRestorer
from dogcode_memory.context.summary_cache import SummaryCache
from dogcode_memory.deduplicator import MemoryDeduplicator
from dogcode_memory.extractor import CandidateMemory, MemoryExtractor
from dogcode_memory.index import MemoryIndex
from dogcode_memory.injector import MemoryInjector
from dogcode_memory.lifecycle import MemoryArchiver
from dogcode_memory.retriever import MemoryRetriever
from dogcode_memory.store import MemoryStore
from dogcode_memory.updater import MemoryOperation, MemoryUpdater


class MemoryPipeline:
    """
    记忆管线 - 编排记忆模块的完整生命周期。

    提供统一的接口，便于 ReuleauxCoder-ezcode 集成：
    - 会话启动：注入历史记忆
    - 会话运行：可选的实时提取
    - 会话结束：归档 + 提取记忆（写操作受文件锁保护）
    """

    def __init__(
        self,
        store: MemoryStore,
        index: MemoryIndex,
        extractor: MemoryExtractor,
        deduplicator: MemoryDeduplicator,
        updater: MemoryUpdater,
        retriever: MemoryRetriever,
        injector: MemoryInjector,
        archiver: MemoryArchiver,
        session_archiver: SessionArchiver | None = None,
        session_restorer: SessionRestorer | None = None,
        config: MemoryConfig | None = None,
    ):
        """
        初始化管线。

        Args:
            store: 记忆存储
            index: 记忆索引
            extractor: 记忆提取器
            deduplicator: 去重器
            updater: 更新器
            retriever: 检索器
            injector: 注入器
            archiver: 冷记忆归档器（生命周期管理）
            session_archiver: 会话归档器（上下文管理）
            session_restorer: 会话恢复器
            config: 记忆配置
        """
        self._store = store
        self._index = index
        self._extractor = extractor
        self._deduplicator = deduplicator
        self._updater = updater
        self._retriever = retriever
        self._injector = injector
        self._archiver = archiver
        self._session_archiver = session_archiver
        self._session_restorer = session_restorer
        self._config = config or MemoryConfig()
        self._enabled = self._config.enabled

    @classmethod
    def create(
        cls,
        storage_dir: str,
        llm: Any | None = None,
        embedding_provider: Any | None = None,
        config: MemoryConfig | None = None,
    ) -> "MemoryPipeline":
        """
        工厂方法 - 使用默认配置快速创建管线实例。

        Args:
            storage_dir: 记忆存储根目录
            llm: LLM 客户端
            embedding_provider: Embedding 提供者
            config: 可选的配置覆盖

        Returns:
            配置好的 MemoryPipeline 实例
        """
        from dogcode_memory.registry import MemoryTypeRegistry

        cfg = config or MemoryConfig()
        if not cfg.storage_dir:
            cfg.storage_dir = storage_dir

        store = MemoryStore(cfg.storage_dir)
        registry = MemoryTypeRegistry()

        db_path = f"{cfg.storage_dir}/.index.db"
        index = MemoryIndex(
            db_path=db_path,
            embedding_provider=embedding_provider,
            dimension=cfg.embedding_dimension,
        )

        extractor = MemoryExtractor(llm=llm, registry=registry, config=cfg)
        deduplicator = MemoryDeduplicator(
            similarity_provider=index if embedding_provider else None,
            llm=llm,
            config=cfg,
        )
        updater = MemoryUpdater(store=store, registry=registry)
        retriever = MemoryRetriever(store=store, index=index, registry=registry, config=cfg)
        injector = MemoryInjector(retriever=retriever, config=cfg)
        archiver = MemoryArchiver(store=store, config=cfg)

        # 会话归档与恢复
        import os
        archive_dir = os.path.join(cfg.storage_dir, "sessions")
        session_archiver = SessionArchiver(
            archive_dir=archive_dir,
            summary_fn=None,
            structured_summary_fn=generate_structured_summary,
        )
        session_restorer = SessionRestorer(
            archiver=session_archiver,
            token_budget=8000,
        )

        return cls(
            store=store,
            index=index,
            extractor=extractor,
            deduplicator=deduplicator,
            updater=updater,
            retriever=retriever,
            injector=injector,
            archiver=archiver,
            session_archiver=session_archiver,
            session_restorer=session_restorer,
            config=cfg,
        )

    def on_session_start(
        self,
        session_id: str,
        context: str = "",
    ) -> str:
        """
        会话启动：检索并注入相关历史记忆。

        读操作无需加锁（文件原子读 + SQLite WAL 模式）。

        Args:
            session_id: 新会话 ID
            context: 会话上下文描述

        Returns:
            格式化的记忆注入文本（供 System Prompt 使用）
        """
        if not self._enabled:
            return ""
        return self._injector.inject_for_session(session_context=context)

    def on_session_end(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        会话结束：提取长期记忆并更新存储。

        整个写入流程受文件锁保护，防止多会话同时结束造成冲突。

        Args:
            session_id: 会话 ID
            messages: 会话消息列表

        Returns:
            操作结果统计
        """
        if not self._enabled or not messages:
            return {"extracted": 0, "created": 0, "merged": 0, "skipped": 0}

        # 提取候选记忆（无需锁，纯内存操作）
        candidates = self._extractor.extract(session_id, messages)

        # 去重并生成操作（无需锁，纯内存操作）
        operations: list[MemoryOperation] = []
        stats = {"extracted": len(candidates), "created": 0, "merged": 0, "skipped": 0}

        for candidate in candidates:
            decision = self._deduplicator.deduplicate(candidate)

            if decision.action == "skip":
                stats["skipped"] += 1
                continue

            uri = self._updater.generate_uri(
                memory_type=candidate.type,
                name=candidate.abstract[:40],
                space="user" if candidate.type in ("profile", "preferences", "project") else "agent",
            )

            candidate_dict = {
                "type": candidate.type,
                "abstract": candidate.abstract,
                "overview": candidate.overview,
                "content": candidate.content,
                "fields": candidate.fields,
                "source_session": candidate.source_session,
                "space": "user" if candidate.type in ("profile", "preferences", "project") else "agent",
            }

            if decision.action == "create":
                operations.append(MemoryOperation(op_type="write", uri=uri, candidate=candidate_dict))
                stats["created"] += 1
            elif decision.action == "merge" and decision.target_uri:
                operations.append(MemoryOperation(
                    op_type="edit",
                    uri=decision.target_uri,
                    candidate=candidate_dict,
                ))
                stats["merged"] += 1
            else:
                operations.append(MemoryOperation(op_type="write", uri=uri, candidate=candidate_dict))
                stats["created"] += 1

        # 持有文件锁执行所有写入操作（文件存储 + 索引更新）
        with self._store.lock:
            modified = self._updater.apply_operations(operations)

            for uri in modified:
                content = self._store.read(uri)
                if content:
                    from dogcode_memory.format import deserialize_memory
                    memory = deserialize_memory(content)
                    self._index.index_memory(
                        uri=uri,
                        abstract=memory.abstract or memory.type,
                        content=memory.content,
                        updated_at=memory.updated_at,
                    )

        return stats

    def run_maintenance(self) -> dict[str, Any]:
        """
        运行维护任务：冷记忆归档、索引优化。

        归档操作受文件锁保护。

        Returns:
            维护结果统计
        """
        if not self._enabled:
            return {"archived": 0}

        with self._store.lock:
            archived = self._archiver.archive_cold_memories()
        return {"archived": len(archived), "archived_uris": archived}

    def archive_session(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        将会话消息归档到存储。

        归档后旧消息可从活跃列表中移除，仅保留归档摘要用于恢复。

        Args:
            session_id: 会话 ID
            messages: 待归档的消息列表
            metadata: 可选的会话元数据

        Returns:
            归档结果信息
        """
        if not self._enabled or not messages or not self._session_archiver:
            return {"archived": False, "reason": "disabled_or_empty"}

        try:
            record = self._session_archiver.archive(
                session_id=session_id,
                messages=messages,
                metadata=metadata,
            )
            return {
                "archived": True,
                "archive_id": record.archive_id,
                "message_count": record.message_count,
                "token_count": record.token_count,
                "summary_length": len(record.summary),
            }
        except Exception as e:
            return {"archived": False, "reason": str(e)}

    def restore_session(
        self,
        session_id: str,
        active_messages: list[dict[str, Any]] | None = None,
        query: str = "",
    ) -> RestoreResult:
        """
        从归档和相关记忆中恢复会话上下文。

        Args:
            session_id: 会话 ID
            active_messages: 当前已加载的活跃消息（如从 SessionStore 加载的）
            query: 检索记忆的查询文本

        Returns:
            RestoreResult，包含组装好的消息列表
        """
        if not self._enabled or not self._session_restorer:
            return RestoreResult(
                messages=active_messages or [],
                info={"reason": "disabled_or_no_restorer"},
            )

        if active_messages:
            return self._session_restorer.restore_with_active_messages(
                session_id=session_id,
                active_messages=active_messages,
                query=query,
                memory_retriever=self._retriever,
            )

        return self._session_restorer.restore(
            session_id=session_id,
            query=query,
            memory_retriever=self._retriever,
        )

    def get_session_archives(self, session_id: str) -> list[dict[str, Any]]:
        """
        获取会话的所有归档记录（只读）。

        Args:
            session_id: 会话 ID

        Returns:
            归档记录字典列表
        """
        if not self._session_archiver:
            return []
        records = self._session_archiver.load_archives(session_id)
        return [r.to_dict() for r in records]

    def delete_session_archive(self, session_id: str, archive_id: str) -> bool:
        """删除指定会话归档。"""
        if not self._session_archiver:
            return False
        return self._session_archiver.delete_archive(session_id, archive_id)

    def get_stats(self) -> dict[str, Any]:
        """获取管线统计信息。

        读操作无需加锁。
        """
        stats = {
            "enabled": self._enabled,
            "store": self._store.get_stats(),
            "index": self._index.get_stats(),
        }
        if self._session_archiver:
            stats["sessions_with_archives"] = len(self._session_archiver.list_session_ids())
        return stats
