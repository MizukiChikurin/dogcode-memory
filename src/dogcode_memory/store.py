"""文件系统存储 - 基于 Markdown 文件的记忆持久化。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


class MemoryStore:
    """基于文件系统的记忆存储，每条记忆是一个 Markdown 文件。"""

    def __init__(self, base_dir: str):
        """
        初始化存储，绑定记忆根目录。

        Args:
            base_dir: 记忆根目录路径，对应架构中的 ~/.dogcode/memories
        """
        self._base_dir = Path(base_dir)
        self._ensure_dir(self._base_dir)

    @property
    def base_dir(self) -> Path:
        """获取存储根目录。"""
        return self._base_dir

    def read(self, uri: str) -> str:
        """
        读取记忆文件内容。

        Args:
            uri: 记忆 URI，如 "memories/user/profile.md" 或 "user/profile.md"

        Returns:
            文件内容字符串，文件不存在时返回空字符串
        """
        path = self._resolve_uri(uri)
        if not path.exists():
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def write(self, uri: str, content: str) -> None:
        """
        写入记忆文件。

        Args:
            uri: 记忆 URI
            content: 文件内容
        """
        path = self._resolve_uri(uri)
        self._ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def delete(self, uri: str) -> bool:
        """
        删除记忆文件。

        Returns:
            是否成功删除
        """
        path = self._resolve_uri(uri)
        if not path.exists():
            return False
        try:
            path.unlink()
            # 清理空目录
            self._cleanup_empty_dirs(path.parent)
            return True
        except Exception:
            return False

    def exists(self, uri: str) -> bool:
        """检查记忆文件是否存在。"""
        return self._resolve_uri(uri).exists()

    def list(self, type_name: str, space: str | None = None) -> list[str]:
        """
        列出指定类型下的所有记忆 URI。

        Args:
            type_name: 记忆类型名称，如 "profile", "preferences"
            space: 空间过滤，"user" 或 "agent"，None 表示在 user 和 agent 下都搜索

        Returns:
            URI 列表（相对 base_dir 的路径）
        """
        results: list[str] = []
        spaces = [space] if space else ["user", "agent"]

        for sp in spaces:
            search_dir = self._base_dir / sp / type_name
            if not search_dir.exists():
                continue
            for path in search_dir.rglob("*.md"):
                # 排除隐藏文件和归档文件
                if path.name.startswith(".") or path.name.startswith("_"):
                    continue
                # 计算相对 base_dir 的 URI
                rel = path.relative_to(self._base_dir)
                results.append(str(rel).replace("\\", "/"))
        return sorted(results)

    def move(self, source_uri: str, target_uri: str) -> bool:
        """
        移动记忆文件。

        Args:
            source_uri: 源 URI
            target_uri: 目标 URI

        Returns:
            是否成功移动
        """
        source_path = self._resolve_uri(source_uri)
        target_path = self._resolve_uri(target_uri)
        if not source_path.exists():
            return False
        self._ensure_dir(target_path.parent)
        try:
            shutil.move(str(source_path), str(target_path))
            self._cleanup_empty_dirs(source_path.parent)
            return True
        except Exception:
            return False

    def _resolve_uri(self, uri: str) -> Path:
        """
        将记忆 URI 解析为文件系统路径。

        URI 格式支持：
        - "user/profile.md" → {base_dir}/user/profile.md
        - "memories/user/profile.md" → {base_dir}/user/profile.md（兼容前缀）
        """
        # 去除 "memories/" 前缀（如果存在）
        clean = uri
        if clean.startswith("memories/"):
            clean = clean[len("memories/"):]
        return self._base_dir / clean.replace("/", os.sep)

    @staticmethod
    def _ensure_dir(directory: Path) -> None:
        """确保目录存在。"""
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cleanup_empty_dirs(directory: Path) -> None:
        """向上清理空目录。"""
        try:
            while directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
                directory = directory.parent
        except Exception:
            pass

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计信息。"""
        total_files = 0
        total_size = 0
        type_counts: dict[str, int] = {}

        if self._base_dir.exists():
            for path in self._base_dir.rglob("*.md"):
                if path.name.startswith(".") or path.name.startswith("_"):
                    continue
                total_files += 1
                total_size += path.stat().st_size
                # 尝试推断类型
                rel = path.relative_to(self._base_dir)
                parts = str(rel).split(os.sep)
                if len(parts) >= 2:
                    # 路径结构: user/profile.md 或 user/preferences/style.md
                    if parts[0] in ("user", "agent"):
                        type_key = parts[1].replace(".md", "")
                    else:
                        type_key = parts[0].replace(".md", "")
                    type_counts[type_key] = type_counts.get(type_key, 0) + 1

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "type_counts": type_counts,
            "base_dir": str(self._base_dir),
        }
