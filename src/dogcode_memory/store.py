"""文件系统存储 - 基于 Markdown 文件的记忆持久化。

并发安全说明：
- 所有写操作（write/delete/move）受 filelock 保护，确保同一存储目录下
  多进程/多线程不会同时修改文件。
- 读操作（read/exists/list）无锁，依赖文件系统的原子性读。
- filelock 使用 ~{base_dir}.lock 作为锁文件，进程退出后自动释放。
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from filelock import FileLock


class MemoryStore:
    """基于文件系统的记忆存储，每条记忆是一个 Markdown 文件。

    线程/进程安全：所有写操作通过文件锁互斥。
    """

    def __init__(self, base_dir: str):
        """
        初始化存储，绑定记忆根目录。

        Args:
            base_dir: 记忆根目录路径，对应架构中的 ~/.dogcode/memories
        """
        self._base_dir = Path(base_dir)
        self._ensure_dir(self._base_dir)
        # 锁文件放在存储目录同级，避免污染记忆数据
        self._lock = FileLock(str(self._base_dir) + ".store.lock")

    @property
    def base_dir(self) -> Path:
        """获取存储根目录。"""
        return self._base_dir

    @property
    def lock(self) -> FileLock:
        """获取文件锁对象（供外部组合锁使用）。"""
        return self._lock

    @contextmanager
    def _write_lock(self, timeout: float = 10.0) -> Generator[None, None, None]:
        """获取写操作锁的上下文管理器。

        Args:
            timeout: 等待锁的最长时间（秒），超时抛出 TimeoutError
        """
        self._lock.acquire(timeout=timeout)
        try:
            yield
        finally:
            self._lock.release()

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

    def write(self, uri: str, content: str, *, _lock: bool = True) -> None:
        """
        写入记忆文件。

        Args:
            uri: 记忆 URI
            content: 文件内容
            _lock: 是否获取写锁（内部批量操作时设为 False 由上层控制）
        """
        path = self._resolve_uri(uri)
        self._ensure_dir(path.parent)

        def _do_write() -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        if _lock:
            with self._write_lock():
                _do_write()
        else:
            _do_write()

    def delete(self, uri: str) -> bool:
        """
        删除记忆文件。

        Returns:
            是否成功删除
        """
        path = self._resolve_uri(uri)
        if not path.exists():
            return False
        with self._write_lock():
            try:
                path.unlink()
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
                if path.name.startswith(".") or path.name.startswith("_"):
                    continue
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
        with self._write_lock():
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
                rel = path.relative_to(self._base_dir)
                parts = str(rel).split(os.sep)
                if len(parts) >= 2:
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
