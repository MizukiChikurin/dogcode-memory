"""生命周期管理 - 热度评分与冷记忆归档。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from dogcode_memory.config import MemoryConfig
from dogcode_memory.store import MemoryStore


def hotness_score(
    active_count: int,
    updated_at: str,
    now: datetime | None = None,
    half_life_days: float = 7.0,
) -> float:
    """
    热度评分算法。

    公式：sigmoid(log1p(n)) × exp(-λ × age)
    其中 λ = ln(2) / half_life_days

    Args:
        active_count: 被访问/检索的次数
        updated_at: 最后更新时间（ISO 格式字符串）
        now: 当前时间，None 时使用系统当前时间
        half_life_days: 半衰期天数

    Returns:
        热度评分 [0.0, 1.0]
    """
    # 频率分量：sigmoid(log(1+n))
    freq = _sigmoid(math.log1p(active_count))

    # 时间衰减分量
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        age_days = (now - updated).total_seconds() / 86400.0
    except Exception:
        age_days = 0.0

    if half_life_days <= 0:
        recency = 0.0
    else:
        lambda_val = math.log(2) / half_life_days
        recency = math.exp(-lambda_val * max(0, age_days))

    return freq * recency


def _sigmoid(x: float) -> float:
    """Sigmoid 函数。"""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class MemoryArchiver:
    """冷记忆归档器 - 将不活跃的记忆移入归档目录。"""

    def __init__(
        self,
        store: MemoryStore,
        config: MemoryConfig | None = None,
    ):
        """
        初始化归档器。

        Args:
            store: 记忆存储
            config: 记忆配置
        """
        self._store = store
        self._config = config or MemoryConfig()

    def archive_cold_memories(self) -> list[str]:
        """
        扫描并归档冷记忆。

        将热度低于阈值且年龄超过最小天数的记忆移入 _archive/ 子目录。

        Returns:
            被归档的记忆 URI 列表
        """
        from dogcode_memory.format import deserialize_memory

        archived: list[str] = []
        all_uris = []

        # 收集所有记忆 URI
        # 遍历 user/ 和 agent/ 空间
        for space in ("user", "agent"):
            for subdir in ("profile", "preferences", "project", "tools", "patterns", "errors"):
                all_uris.extend(self._store.list(subdir, space=space))

        now = datetime.now(timezone.utc)

        for uri in all_uris:
            content = self._store.read(uri)
            if not content:
                continue

            memory = deserialize_memory(content)
            try:
                updated = datetime.fromisoformat(memory.updated_at.replace("Z", "+00:00"))
                age_days = (now - updated).total_seconds() / 86400.0
            except Exception:
                age_days = 0.0

            score = hotness_score(
                active_count=memory.active_count,
                updated_at=memory.updated_at,
                now=now,
                half_life_days=self._config.hotness_half_life_days,
            )

            if score < self._config.cold_threshold and age_days >= self._config.min_age_days_for_archive:
                archive_uri = self._to_archive_uri(uri)
                self._store.move(uri, archive_uri)
                archived.append(uri)

        return archived

    def restore_memory(self, archive_uri: str) -> bool:
        """
        从归档恢复记忆。

        Args:
            archive_uri: 归档中的 URI

        Returns:
            是否成功恢复
        """
        original_uri = self._from_archive_uri(archive_uri)
        if not original_uri:
            return False
        return self._store.move(archive_uri, original_uri)

    def list_archived(self) -> list[str]:
        """列出所有已归档的记忆 URI。"""
        # 归档文件存储在 user/_archive/ 和 agent/_archive/ 子目录下
        archived: list[str] = []
        base = self._store.base_dir
        if not base.exists():
            return archived

        for path in base.rglob("*.md"):
            # 只收集路径中包含 _archive 的文件
            rel = str(path.relative_to(base)).replace("\\", "/")
            if "_archive" in rel and not path.name.startswith("."):
                archived.append(rel)
        return sorted(archived)

    @staticmethod
    def _to_archive_uri(uri: str) -> str:
        """将普通 URI 转换为归档 URI。"""
        parts = uri.split("/")
        # 在 space 后插入 _archive
        if len(parts) >= 1:
            return "/".join([parts[0], "_archive"] + parts[1:])
        return f"_archive/{uri}"

    @staticmethod
    def _from_archive_uri(archive_uri: str) -> str | None:
        """将归档 URI 还原为普通 URI。"""
        parts = archive_uri.split("/")
        if "_archive" in parts:
            idx = parts.index("_archive")
            return "/".join(parts[:idx] + parts[idx + 1:])
        return None
