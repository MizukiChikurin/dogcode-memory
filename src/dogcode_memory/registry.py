"""YAML 模板注册表 - 加载和管理记忆类型定义。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from dogcode_memory.schema import MemoryTypeSchema


class MemoryTypeRegistry:
    """记忆类型注册表，负责加载内置和自定义 YAML 模板。"""

    def __init__(self, builtin_dir: str | None = None, custom_dir: str | None = None):
        """
        初始化注册表。

        Args:
            builtin_dir: 内置 YAML 模板目录，默认使用包内 schemas/ 目录
            custom_dir: 自定义 YAML 模板目录（可选）
        """
        self._types: dict[str, MemoryTypeSchema] = {}
        if builtin_dir is None:
            # 默认使用包内的 schemas/ 目录
            pkg_dir = Path(__file__).parent
            builtin_dir = str(pkg_dir / "schemas")
        self._builtin_dir = builtin_dir
        self._custom_dir = custom_dir
        self.load_schemas()

    def load_schemas(self) -> None:
        """加载内置和自定义 YAML 模板。"""
        self._types.clear()
        # 加载内置模板
        if self._builtin_dir and os.path.isdir(self._builtin_dir):
            for path in Path(self._builtin_dir).glob("*.yaml"):
                schema = self._load_yaml(str(path))
                if schema:
                    self._types[schema.name] = schema
        # 加载自定义模板（覆盖内置）
        if self._custom_dir and os.path.isdir(self._custom_dir):
            for path in Path(self._custom_dir).glob("*.yaml"):
                schema = self._load_yaml(str(path))
                if schema:
                    self._types[schema.name] = schema

    def get_schema(self, type_name: str) -> MemoryTypeSchema | None:
        """获取指定类型的 Schema。"""
        return self._types.get(type_name)

    def list_types(self) -> list[str]:
        """列出所有已注册的记忆类型名称。"""
        return list(self._types.keys())

    def list_schemas(self) -> list[MemoryTypeSchema]:
        """列出所有已注册的 Schema 对象。"""
        return list(self._types.values())

    def _load_yaml(self, path: str) -> MemoryTypeSchema | None:
        """解析单个 YAML 文件为 MemoryTypeSchema。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return None
            return MemoryTypeSchema.from_dict(data)
        except Exception:
            return None

    def register_schema(self, schema: MemoryTypeSchema) -> None:
        """动态注册一个 Schema（用于运行时扩展）。"""
        self._types[schema.name] = schema

    def to_dict(self) -> dict[str, Any]:
        """将所有 Schema 序列化为字典。"""
        return {name: schema.to_dict() for name, schema in self._types.items()}
