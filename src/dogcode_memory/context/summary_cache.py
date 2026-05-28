"""摘要缓存 - 避免对同一消息段重复调用 LLM 生成摘要。

设计原则：
1. 以消息列表的 hash 为 key，摘要文本为 value
2. 进程内内存缓存，不持久化（摘要可在需要时重新生成）
3. 提供线程安全的读写接口
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any


def compute_message_hash(messages: list[dict[str, Any]]) -> str:
    """
    计算消息列表的确定性哈希值，用于摘要缓存 key。

    算法：
    1. 将每条消息按 (role, content, tool_calls) 序列化为紧凑 JSON
    2. 拼接所有消息的 JSON 字符串
    3. 使用 SHA-256 计算哈希，取前 24 位作为 key

    Args:
        messages: 消息列表

    Returns:
        哈希字符串，如 "a1b2c3d4e5f6..."
    """
    parts: list[str] = []
    for m in messages:
        # 只取影响语义的核心字段，忽略临时 token 计数等
        key_data = {
            "role": m.get("role", ""),
            "content": m.get("content", "") or "",
        }
        tool_calls = m.get("tool_calls")
        if tool_calls:
            key_data["tool_calls"] = tool_calls
        # 使用排序键 + 无空格确保确定性
        parts.append(json.dumps(key_data, sort_keys=True, separators=(",", ":")))

    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:24]


class SummaryCache:
    """摘要缓存，以消息 hash 为 key，线程安全。"""

    def __init__(self, max_size: int = 256):
        """
        初始化缓存。

        Args:
            max_size: 最大缓存条目数，超出时按 LRU 淘汰
        """
        self._max_size = max_size
        self._cache: dict[str, str] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()

    def get(self, messages: list[dict[str, Any]]) -> str | None:
        """
        获取消息列表对应的缓存摘要。

        Args:
            messages: 消息列表

        Returns:
            缓存的摘要文本，未命中返回 None
        """
        key = compute_message_hash(messages)
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                # 更新访问顺序（移至末尾 = 最近使用）
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
            return value

    def put(self, messages: list[dict[str, Any]], summary: str) -> None:
        """
        将摘要写入缓存。

        Args:
            messages: 消息列表
            summary: 生成的摘要文本
        """
        key = compute_message_hash(messages)
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self._max_size:
                # LRU 淘汰最久未使用的
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)

            self._cache[key] = summary
            self._access_order.append(key)

    def invalidate(self, messages: list[dict[str, Any]]) -> bool:
        """
        使指定消息列表的缓存失效。

        Returns:
            是否成功移除
        """
        key = compute_message_hash(messages)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存。"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计信息。"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_keys": list(self._access_order),
            }
