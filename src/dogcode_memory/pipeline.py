"""管线编排 - 将记忆模块组件编排为完整的会话生命周期管线。

并发安全说明：
- on_session_end 中的写操作受 MemoryStore 文件锁保护，
  确保多会话同时结束不会导致文件写入冲突。
- SQLite 索引通过 WAL 模式天然支持读写并发，无需额外锁。
"""

from __future__ import annotations

import logging
from typing import Any

from dogcode_memory.async_ops import AsyncTask, AsyncTaskPool, ExtractionTask
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

logger = logging.getLogger(__name__)


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
        async_pool: AsyncTaskPool | None = None,
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
            async_pool: 异步任务池，None 时延迟创建
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
        self._async_pool = async_pool
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

        async_pool = AsyncTaskPool(max_workers=2)

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
            async_pool=async_pool,
            config=cfg,
        )

    def on_session_start(
        self,
        session_id: str,
        context: str = "",
    ) -> dict[str, Any]:
        """
        会话启动：检索并注入相关历史记忆。

        读操作无需加锁（文件原子读 + SQLite WAL 模式）。

        Args:
            session_id: 新会话 ID
            context: 会话上下文描述

        Returns:
            注入结果字典 {"text": "...", "uris": [...], "count": N, "total_tokens": T}
        """
        if not self._enabled:
            logger.debug("on_session_start: 记忆系统已禁用，跳过注入 (session=%s)", session_id)
            return {"text": "", "uris": [], "count": 0, "total_tokens": 0}

        logger.info("on_session_start: 开始注入记忆 (session=%s, context=%r)", session_id, context[:100] if context else "")
        result = self._injector.inject_for_session(session_context=context)

        # 记录最近一次注入元数据，供前端查询
        if result and result.get("count", 0) > 0:
            import time
            self._last_injection_meta = {
                "last_injection": result,
                "timestamp": time.time(),
            }
            logger.info("on_session_start: 注入完成 (session=%s, count=%d, tokens=%d)",
                        session_id, result.get("count", 0), result.get("total_tokens", 0))
        else:
            logger.debug("on_session_start: 未找到相关记忆 (session=%s)", session_id)

        return result

    def get_injected_memory_meta(self) -> dict[str, Any]:
        """
        获取最近一次注入记忆的元数据（供前端查询）。

        Returns:
            {"last_injection": {"text": ..., "uris": [...], "count": ...}, "timestamp": ...}
            或空字典（尚未注入）
        """
        return getattr(self, "_last_injection_meta", {})

    def on_session_end(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        会话结束：提取长期记忆并更新存储。

        整个写入流程受文件锁保护，防止多会话同时结束造成冲突。
        索引更新失败不会回滚文件写入（索引可通过重建修复）。

        Args:
            session_id: 会话 ID
            messages: 会话消息列表

        Returns:
            操作结果统计
        """
        if not self._enabled or not messages:
            return {"extracted": 0, "created": 0, "merged": 0, "skipped": 0}

        logger.info("on_session_end: 开始处理 (session=%s, messages=%d)", session_id, len(messages))

        # 提取候选记忆（无需锁，纯内存操作）
        try:
            candidates = self._extractor.extract(session_id, messages)
        except Exception as e:
            logger.warning("on_session_end: 记忆提取失败 (session=%s) - %s", session_id, e)
            return {"extracted": 0, "created": 0, "merged": 0, "skipped": 0, "error": "extraction_failed"}

        logger.debug("on_session_end: 提取到 %d 个候选记忆 (session=%s)", len(candidates), session_id)

        # 去重并生成操作（无需锁，纯内存操作）
        operations: list[MemoryOperation] = []
        stats: dict[str, Any] = {"extracted": len(candidates), "created": 0, "merged": 0, "skipped": 0}

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

        if not operations:
            logger.debug("on_session_end: 无操作需要执行 (session=%s)", session_id)
            return stats

        # 持有文件锁执行所有写入操作（文件存储 + 索引更新）
        modified: list[str] = []
        index_errors: list[str] = []
        with self._store.lock:
            try:
                modified = self._updater.apply_operations(operations)
            except Exception as e:
                logger.error("on_session_end: 存储写入失败 (session=%s) - %s", session_id, e)
                stats["error"] = "store_write_failed"
                return stats

            for uri in modified:
                try:
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
                except Exception as e:
                    logger.warning("on_session_end: 索引更新失败 (uri=%s) - %s", uri, e)
                    index_errors.append(uri)

        if index_errors:
            stats["index_errors"] = len(index_errors)
            logger.warning("on_session_end: %d 条记忆索引更新失败 (session=%s)", len(index_errors), session_id)

        logger.info("on_session_end: 处理完成 (session=%s, created=%d, merged=%d, skipped=%d)",
                    session_id, stats["created"], stats["merged"], stats["skipped"])
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

        logger.info("run_maintenance: 开始维护任务")

        try:
            with self._store.lock:
                archived = self._archiver.archive_cold_memories()
            logger.info("run_maintenance: 归档完成，%d 条冷记忆已归档", len(archived))
            return {"archived": len(archived), "archived_uris": archived}
        except Exception as e:
            logger.exception("run_maintenance: 维护任务失败")
            return {"archived": 0, "error": "maintenance_failed"}

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
            logger.debug("archive_session: 跳过归档 (session=%s, enabled=%s, has_archiver=%s)",
                         session_id, self._enabled, self._session_archiver is not None)
            return {"archived": False, "reason": "disabled_or_empty"}

        logger.info("archive_session: 开始归档 (session=%s, messages=%d)", session_id, len(messages))

        try:
            record = self._session_archiver.archive(
                session_id=session_id,
                messages=messages,
                metadata=metadata,
            )
            logger.info("archive_session: 归档成功 (session=%s, archive_id=%s, messages=%d)",
                        session_id, record.archive_id, record.message_count)
            return {
                "archived": True,
                "archive_id": record.archive_id,
                "message_count": record.message_count,
                "token_count": record.token_count,
                "summary_length": len(record.summary),
            }
        except PermissionError as e:
            logger.error("archive_session: 无权限归档 (session=%s) - %s", session_id, e)
            return {"archived": False, "reason": "permission_denied"}
        except OSError as e:
            logger.error("archive_session: 归档失败 (session=%s) - %s", session_id, e)
            return {"archived": False, "reason": "io_error"}
        except Exception as e:
            logger.exception("archive_session: 归档异常 (session=%s)", session_id)
            return {"archived": False, "reason": "internal_error"}

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
        if self._async_pool:
            stats["pending_tasks"] = len(self._async_pool.get_pending())
        return stats

    # ───────────────────────────────────────────────
    # Phase 2: 异步提取
    # ───────────────────────────────────────────────

    def on_session_end_async(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """
        异步触发会话结束的记忆提取。

        将提取任务提交到后台线程池，立即返回任务 ID，
        不阻塞调用方。提取完成后可通过 wait_for_task 查询结果。

        Args:
            session_id: 会话 ID
            messages: 会话消息列表

        Returns:
            任务 ID，可用于后续查询
        """
        if not self._enabled or not messages:
            return ""

        if self._async_pool is None:
            self._async_pool = AsyncTaskPool(max_workers=2)

        task = self._async_pool.submit(
            task_type="memory_extraction",
            fn=self._do_session_end,
            payload={"session_id": session_id},
            session_id=session_id,
            messages=messages,
        )
        return task.task_id

    def _do_session_end(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """后台执行会话结束提取的内部方法。"""
        return self.on_session_end(session_id, messages)

    def get_pending_tasks(self) -> list[str]:
        """
        获取正在执行中的任务 ID 列表。

        Returns:
            任务 ID 列表
        """
        if self._async_pool is None:
            return []
        return self._async_pool.get_pending()

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """
        获取指定任务的当前状态。

        Args:
            task_id: 任务 ID

        Returns:
            任务状态字典，任务不存在返回 None
        """
        if self._async_pool is None:
            return None
        task = self._async_pool.get_task(task_id)
        return task.to_dict() if task else None

    def wait_for_task(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """
        等待指定任务完成并返回结果。

        Args:
            task_id: 任务 ID
            timeout: 等待超时（秒），None 表示无限等待

        Returns:
            任务结果字典（包含 result 和 error），任务不存在返回 None
        """
        if self._async_pool is None:
            return None
        task = self._async_pool.wait_for(task_id, timeout=timeout)
        if task is None:
            return None
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "is_done": task.is_done(),
            "has_error": task.error is not None,
            "error_message": str(task.error) if task.error else None,
            "result": task.result,
        }

    def wait_all_tasks(self, timeout: float | None = None) -> None:
        """
        等待所有已提交的异步任务完成。

        Args:
            timeout: 等待超时（秒）
        """
        if self._async_pool is not None:
            self._async_pool.wait_all(timeout=timeout)

    def shutdown_async_pool(self, wait: bool = True) -> None:
        """
        关闭异步任务池，释放线程资源。

        Args:
            wait: 是否等待所有任务完成后再关闭
        """
        if self._async_pool is not None:
            self._async_pool.shutdown(wait=wait)
            self._async_pool = None
