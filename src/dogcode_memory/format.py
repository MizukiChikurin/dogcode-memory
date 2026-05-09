"""序列化与反序列化 - 记忆文件的 Markdown + YAML 元数据头格式。"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import yaml

from dogcode_memory.schema import MemoryTypeSchema

MEMORY_FILE_HEADER = """---
{metadata}
---

"""


class MemoryData:
    """解析后的记忆数据对象。"""

    def __init__(
        self,
        uri: str = "",
        type: str = "",
        space: str = "",
        created_at: str = "",
        updated_at: str = "",
        active_count: int = 0,
        source_sessions: list[str] | None = None,
        content: str = "",
        fields: dict[str, Any] | None = None,
    ):
        self.uri = uri
        self.type = type
        self.space = space
        self.created_at = created_at
        self.updated_at = updated_at
        self.active_count = active_count
        self.source_sessions = source_sessions or []
        self.content = content
        self.fields = fields or {}


def serialize_memory(
    memory_data: MemoryData,
    schema: MemoryTypeSchema | None = None,
) -> str:
    """
    将记忆对象序列化为 Markdown + YAML 元数据头格式。

    Args:
        memory_data: 记忆数据对象
        schema: 可选的 Schema，用于验证和格式化字段

    Returns:
        序列化后的 Markdown 字符串
    """
    metadata = {
        "uri": memory_data.uri,
        "type": memory_data.type,
        "space": memory_data.space,
        "created_at": memory_data.created_at or _now_iso(),
        "updated_at": _now_iso(),
        "active_count": memory_data.active_count,
    }
    if memory_data.source_sessions:
        metadata["source_sessions"] = memory_data.source_sessions

    # 将自定义字段加入元数据
    for key, value in memory_data.fields.items():
        if key not in metadata:
            metadata[key] = value

    header = MEMORY_FILE_HEADER.format(metadata=yaml.dump(metadata, allow_unicode=True, sort_keys=False))

    # 内容部分：如果 content 非空则直接追加
    # 否则根据 schema 的字段生成结构化内容
    if memory_data.content.strip():
        return header + memory_data.content.strip() + "\n"

    # 根据 schema 生成结构化内容
    parts = []
    if schema:
        for field in schema.fields:
            value = memory_data.fields.get(field.name)
            if value is not None:
                parts.append(f"## {field.name}\n\n{value}\n")

    return header + "\n".join(parts) + "\n"


def deserialize_memory(content: str) -> MemoryData:
    """
    从 Markdown 内容反序列化为记忆对象。

    Args:
        content: Markdown 文件内容

    Returns:
        解析后的 MemoryData 对象
    """
    metadata, body = parse_metadata(content)

    return MemoryData(
        uri=metadata.get("uri", ""),
        type=metadata.get("type", ""),
        space=metadata.get("space", ""),
        created_at=metadata.get("created_at", ""),
        updated_at=metadata.get("updated_at", ""),
        active_count=metadata.get("active_count", 0),
        source_sessions=metadata.get("source_sessions", []),
        content=body,
        fields={k: v for k, v in metadata.items() if k not in {
            "uri", "type", "space", "created_at", "updated_at",
            "active_count", "source_sessions"
        }},
    )


def parse_metadata(raw_content: str) -> tuple[dict[str, Any], str]:
    """
    解析 YAML 元数据头和正文。

    Returns:
        (metadata_dict, body_string)
    """
    raw_content = raw_content.strip()
    if not raw_content.startswith("---"):
        return {}, raw_content

    # 查找第二个 ---
    match = re.search(r"^---\s*\n(.*?)\n---\s*\n", raw_content, re.DOTALL)
    if not match:
        return {}, raw_content

    yaml_str = match.group(1)
    body = raw_content[match.end():]

    try:
        metadata = yaml.safe_load(yaml_str) or {}
    except Exception:
        metadata = {}

    return metadata, body.strip()


def format_metadata(metadata: dict[str, Any]) -> str:
    """格式化 YAML 元数据头。"""
    return yaml.dump(metadata, allow_unicode=True, sort_keys=False)


def _now_iso() -> str:
    """获取当前 ISO 格式时间字符串。"""
    return datetime.now(timezone.utc).isoformat()
