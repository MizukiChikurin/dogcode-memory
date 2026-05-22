"""记忆 Schema 定义 - 数据模型与枚举。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MergeOp(Enum):
    """字段级合并策略枚举。"""

    PATCH = "patch"           # SEARCH/REPLACE 增量编辑
    SUM = "sum"               # 数值累加
    IMMUTABLE = "immutable"   # 首次写入后不可变
    APPEND = "append"         # 追加合并


class OperationMode(Enum):
    """记忆文件操作模式枚举。"""

    UPSERT = "upsert"     # 新增或更新，预取已有内容
    ADD_ONLY = "add_only" # 只新增，不读取已有文件


@dataclass
class MemoryField:
    """记忆类型中的单个字段定义。"""

    name: str
    field_type: str = "string"  # string, int, float, bool, list, dict
    merge_op: MergeOp = MergeOp.PATCH
    init_value: Any = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "name": self.name,
            "type": self.field_type,
            "merge_op": self.merge_op.value,
            "init_value": self.init_value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryField":
        """从字典反序列化。"""
        merge_op_str = data.get("merge_op", "patch")
        try:
            merge_op = MergeOp(merge_op_str)
        except ValueError:
            merge_op = MergeOp.PATCH
        return cls(
            name=data["name"],
            field_type=data.get("type", "string"),
            merge_op=merge_op,
            init_value=data.get("init_value"),
            description=data.get("description", ""),
        )


@dataclass
class MemoryTypeSchema:
    """记忆类型 Schema 定义，对应一个 YAML 模板。"""

    name: str
    space: str = "user"  # "user" 或 "agent"
    file_pattern: str = "{name}.md"  # 文件路径模式
    operation_mode: OperationMode = OperationMode.UPSERT
    fields: list[MemoryField] = field(default_factory=list)
    description: str = ""
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "name": self.name,
            "space": self.space,
            "file_pattern": self.file_pattern,
            "operation_mode": self.operation_mode.value,
            "fields": [f.to_dict() for f in self.fields],
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryTypeSchema":
        """从字典反序列化。"""
        mode_str = data.get("operation_mode", "upsert")
        try:
            mode = OperationMode(mode_str)
        except ValueError:
            mode = OperationMode.UPSERT
        return cls(
            name=data["name"],
            space=data.get("space", "user"),
            file_pattern=data.get("file_pattern", "{name}.md"),
            operation_mode=mode,
            fields=[MemoryField.from_dict(f) for f in data.get("fields", [])],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
        )

    def get_field(self, name: str) -> MemoryField | None:
        """按名称获取字段定义。"""
        for field in self.fields:
            if field.name == name:
                return field
        return None
