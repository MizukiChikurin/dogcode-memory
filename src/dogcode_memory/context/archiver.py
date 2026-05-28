"""会话归档器 - 将旧消息从活跃列表移入归档存储。

归档存储结构：
    {archive_dir}/
        {session_id}/
            session.json              # 会话元数据
            active.jsonl              # 当前活跃消息（可为空）
            archives/
                {archive_id}/
                    summary.json      # 结构化摘要 + 元数据
                    messages.jsonl    # 原始消息（调试用）
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class ArchiveRecord:
    """归档记录模型。"""

    archive_id: str
    session_id: str
    summary: str
    structured_summary: dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    token_count: int = 0
    timestamp: str = ""
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "archive_id": self.archive_id,
            "session_id": self.session_id,
            "summary": self.summary,
            "structured_summary": self.structured_summary,
            "message_count": self.message_count,
            "token_count": self.token_count,
            "timestamp": self.timestamp or _now_iso(),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchiveRecord":
        """从字典反序列化。"""
        return cls(
            archive_id=data.get("archive_id", ""),
            session_id=data.get("session_id", ""),
            summary=data.get("summary", ""),
            structured_summary=data.get("structured_summary", {}),
            message_count=data.get("message_count", 0),
            token_count=data.get("token_count", 0),
            timestamp=data.get("timestamp", ""),
            fingerprint=data.get("fingerprint", ""),
        )


class ArchiveWriter:
    """归档文件写入器。"""

    @staticmethod
    def write_archive(
        directory: Path,
        record: ArchiveRecord,
        messages: list[dict[str, Any]],
    ) -> None:
        """
        将归档记录和原始消息写入指定目录。

        Args:
            directory: 归档目录路径，如 .../archives/archive_001/
            record: 归档记录
            messages: 原始消息列表
        """
        directory.mkdir(parents=True, exist_ok=True)

        # 写入 summary.json
        summary_path = directory / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)

        # 写入 messages.jsonl
        messages_path = directory / "messages.jsonl"
        with open(messages_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    @staticmethod
    def read_archive(directory: Path, archive_id: str) -> tuple[ArchiveRecord, list[dict[str, Any]]] | None:
        """
        读取归档文件。

        Args:
            directory: archives 父目录
            archive_id: 归档 ID

        Returns:
            (record, messages) 元组，不存在返回 None
        """
        archive_dir = directory / archive_id
        if not archive_dir.exists():
            return None

        summary_path = archive_dir / "summary.json"
        if not summary_path.exists():
            return None

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                record = ArchiveRecord.from_dict(json.load(f))
        except Exception:
            return None

        messages: list[dict[str, Any]] = []
        messages_path = archive_dir / "messages.jsonl"
        if messages_path.exists():
            try:
                with open(messages_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            messages.append(json.loads(line))
            except Exception:
                pass

        return record, messages


class SessionArchiver:
    """会话归档器，管理单个会话的归档生命周期。"""

    def __init__(
        self,
        archive_dir: str,
        summary_fn: Callable[[list[dict[str, Any]]], str] | None = None,
        structured_summary_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    ):
        """
        初始化归档器。

        Args:
            archive_dir: 归档根目录，对应架构中的 ~/.dogcode/sessions
            summary_fn: 摘要生成函数，接受消息列表返回摘要文本
            structured_summary_fn: 结构化摘要生成函数，返回结构化字典
        """
        self._archive_dir = Path(archive_dir)
        self._summary_fn = summary_fn
        self._structured_summary_fn = structured_summary_fn

    @property
    def archive_dir(self) -> Path:
        """获取归档根目录。"""
        return self._archive_dir

    def archive(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ArchiveRecord:
        """
        执行归档：生成摘要 → 写入归档文件 → 返回归档记录。

        Args:
            session_id: 会话 ID
            messages: 待归档的消息列表
            metadata: 可选的会话元数据

        Returns:
            归档记录
        """
        # 生成摘要
        summary = ""
        if self._summary_fn:
            try:
                summary = self._summary_fn(messages)
            except Exception:
                summary = ""

        # 生成结构化摘要
        structured_summary: dict[str, Any] = {}
        if self._structured_summary_fn:
            try:
                structured_summary = self._structured_summary_fn(messages)
            except Exception:
                structured_summary = {}

        # 估算 token 数
        token_count = _estimate_tokens(messages)

        archive_id = _generate_archive_id()
        record = ArchiveRecord(
            archive_id=archive_id,
            session_id=session_id,
            summary=summary,
            structured_summary=structured_summary,
            message_count=len(messages),
            token_count=token_count,
            timestamp=_now_iso(),
            fingerprint=metadata.get("fingerprint", "") if metadata else "",
        )

        # 写入归档
        session_dir = self._archive_dir / session_id / "archives"
        archive_dir = session_dir / archive_id
        ArchiveWriter.write_archive(archive_dir, record, messages)

        # 更新会话元数据
        self._update_session_metadata(session_id, metadata or {})

        return record

    def load_archives(self, session_id: str) -> list[ArchiveRecord]:
        """
        加载会话的所有归档摘要。

        Args:
            session_id: 会话 ID

        Returns:
            按时间排序的归档记录列表（最新的在前）
        """
        archives_dir = self._archive_dir / session_id / "archives"
        if not archives_dir.exists():
            return []

        records: list[ArchiveRecord] = []
        for archive_dir in archives_dir.iterdir():
            if not archive_dir.is_dir():
                continue
            summary_path = archive_dir / "summary.json"
            if not summary_path.exists():
                continue
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    record = ArchiveRecord.from_dict(json.load(f))
                records.append(record)
            except Exception:
                continue

        # 按时间戳倒序
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    def build_restore_context(
        self,
        session_id: str,
        token_budget: int = 4000,
    ) -> tuple[str, list[ArchiveRecord]]:
        """
        在 token 预算内组装归档摘要，用于会话恢复。

        Args:
            session_id: 会话 ID
            token_budget: Token 预算上限

        Returns:
            (拼接的摘要文本, 使用的归档记录列表)
        """
        archives = self.load_archives(session_id)
        if not archives:
            return "", []

        parts: list[str] = []
        used: list[ArchiveRecord] = []
        used_tokens = 0
        chars_per_token = 3.0

        for record in archives:
            text = record.summary
            # 若 summary 为空，尝试从 structured_summary 拼接 fallback
            if not text:
                text = _structured_summary_to_text(record.structured_summary)
            # 若仍为空，使用极简 fallback
            if not text:
                text = f"(Archived: {record.message_count} messages from {record.timestamp})"

            estimated_tokens = int(len(text) / chars_per_token)
            if used_tokens + estimated_tokens > token_budget and used:
                break

            parts.append(text)
            used.append(record)
            used_tokens += estimated_tokens

        return "\n\n".join(parts), used

    def list_session_ids(self) -> list[str]:
        """列出所有有归档的会话 ID。"""
        if not self._archive_dir.exists():
            return []
        return sorted([
            p.name for p in self._archive_dir.iterdir()
            if p.is_dir() and (p / "archives").exists()
        ])

    def delete_archive(self, session_id: str, archive_id: str) -> bool:
        """删除指定归档。"""
        archive_dir = self._archive_dir / session_id / "archives" / archive_id
        if not archive_dir.exists():
            return False
        import shutil
        shutil.rmtree(archive_dir)
        return True

    def _update_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """更新会话元数据文件。"""
        session_dir = self._archive_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        meta_path = session_dir / "session.json"
        existing: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        existing.update(metadata)
        existing["last_archived_at"] = _now_iso()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    """获取当前 ISO 格式时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _generate_archive_id() -> str:
    """生成归档 ID。"""
    return f"archive_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:6]}"


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """粗略估算消息列表的 token 数。"""
    total_chars = 0
    for m in messages:
        content = m.get("content", "") or ""
        total_chars += len(content)
        tool_calls = m.get("tool_calls")
        if tool_calls:
            total_chars += len(json.dumps(tool_calls))
    return int(total_chars / 3.0)


def _structured_summary_to_text(structured: dict[str, Any]) -> str:
    """将结构化摘要字典拼接为文本。"""
    if not structured:
        return ""
    parts: list[str] = []
    if structured.get("primary_request"):
        parts.append(f"Request: {structured['primary_request']}")
    files = structured.get("files_and_code", [])
    if files:
        parts.append(f"Files: {', '.join(str(f) for f in files[:10])}")
    decisions = structured.get("decisions", [])
    if decisions:
        parts.append(f"Decisions: {', '.join(str(d) for d in decisions[:5])}")
    errors = structured.get("errors", [])
    if errors:
        parts.append(f"Errors: {', '.join(str(e) for e in errors[:3])}")
    return "\n".join(parts)
