"""操作执行器 - 将去重决策转换为存储操作并执行。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dogcode_memory.format import MemoryData, deserialize_memory, serialize_memory
from dogcode_memory.merge import merge_memory
from dogcode_memory.schema import MemoryTypeSchema
from dogcode_memory.store import MemoryStore


@dataclass
class MemoryOperation:
    """单个记忆操作定义。"""

    op_type: str   # "write", "edit", "delete"
    uri: str
    candidate: dict[str, Any] | None = None  # 用于 write/edit


class MemoryUpdater:
    """记忆更新器 - 执行写入、编辑、删除操作。"""

    def __init__(
        self,
        store: MemoryStore,
        registry: Any | None = None,
    ):
        """
        初始化更新器。

        Args:
            store: 记忆存储实例
            registry: 记忆类型注册表（可选）
        """
        self._store = store
        self._registry = registry

    def apply_operations(self, operations: list[MemoryOperation]) -> list[str]:
        """
        执行一组写入/编辑/删除操作。

        Args:
            operations: 操作列表

        Returns:
            被修改的记忆 URI 列表
        """
        modified: list[str] = []

        for op in operations:
            if op.op_type == "write":
                self._apply_write(op.uri, op.candidate or {})
                modified.append(op.uri)
            elif op.op_type == "edit":
                self._apply_edit(op.uri, op.candidate or {})
                modified.append(op.uri)
            elif op.op_type == "delete":
                self._apply_delete(op.uri)
                modified.append(op.uri)

        return modified

    def _apply_write(self, uri: str, candidate: dict[str, Any]) -> None:
        """写入新记忆文件。"""
        memory_type = candidate.get("type", "unknown")
        space = candidate.get("space", "user")

        # 获取 schema
        schema = None
        if self._registry and hasattr(self._registry, "get_schema"):
            schema = self._registry.get_schema(memory_type)

        now = _now_iso()
        memory_data = MemoryData(
            uri=uri,
            type=memory_type,
            space=space,
            created_at=now,
            updated_at=now,
            active_count=1,
            source_sessions=[candidate.get("source_session", "")],
            content=candidate.get("content", ""),
            fields=candidate.get("fields", {}),
        )

        content = serialize_memory(memory_data, schema)
        self._store.write(uri, content)

    def _apply_edit(self, uri: str, candidate: dict[str, Any]) -> None:
        """编辑已有记忆文件（按 merge_op 合并）。"""
        existing_content = self._store.read(uri)
        if not existing_content:
            # 文件不存在，降级为写入
            self._apply_write(uri, candidate)
            return

        # 解析已有记忆
        existing = deserialize_memory(existing_content)

        memory_type = candidate.get("type", existing.type)
        space = candidate.get("space", existing.space)

        # 获取 schema
        schema = None
        if self._registry and hasattr(self._registry, "get_schema"):
            schema = self._registry.get_schema(memory_type)

        # 合并字段
        if schema:
            merged_fields = merge_memory(
                existing_fields=existing.fields,
                candidate_fields=candidate.get("fields", {}),
                schema=schema,
            )
        else:
            # 无 schema，简单覆盖
            merged_fields = {**existing.fields, **candidate.get("fields", {})}

        # 合并内容
        new_content = candidate.get("content", "")
        if new_content and existing.content:
            merged_content = existing.content.rstrip() + "\n\n" + new_content.strip() + "\n"
        else:
            merged_content = new_content or existing.content

        # 更新 source_sessions
        source_session = candidate.get("source_session", "")
        source_sessions = list(existing.source_sessions)
        if source_session and source_session not in source_sessions:
            source_sessions.append(source_session)

        memory_data = MemoryData(
            uri=uri,
            type=memory_type,
            space=space,
            created_at=existing.created_at,
            updated_at=_now_iso(),
            active_count=existing.active_count + 1,
            source_sessions=source_sessions,
            content=merged_content,
            fields=merged_fields,
        )

        content = serialize_memory(memory_data, schema)
        self._store.write(uri, content)

    def _apply_delete(self, uri: str) -> None:
        """删除记忆文件。"""
        self._store.delete(uri)

    def generate_uri(self, memory_type: str, name: str, space: str = "user") -> str:
        """
        根据类型和名称生成记忆 URI。

        Args:
            memory_type: 记忆类型，如 "profile", "preferences"
            name: 记忆名称/标识符
            space: 空间，"user" 或 "agent"

        Returns:
            记忆 URI，如 "user/preferences/naming-style.md"
        """
        safe_name = _sanitize_filename(name)
        return f"{space}/{memory_type}/{safe_name}.md"


def _now_iso() -> str:
    """获取当前 ISO 格式时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _sanitize_filename(name: str) -> str:
    """清理文件名，去除不安全字符。"""
    safe = name.replace(" ", "-").replace("/", "-").replace("\\", "-")
    safe = "".join(c for c in safe if c.isalnum() or c in "-_.")
    return safe.lower()[:64]
