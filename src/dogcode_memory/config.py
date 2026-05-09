"""配置模型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryConfig:
    """记忆管理配置。"""

    enabled: bool = True
    # 存储根目录，默认使用 ~/.dogcode/memories
    storage_dir: str = ""
    # Embedding 配置
    embedding_provider: str = ""  # 空字符串表示使用主 LLM provider
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    # 提取配置
    extraction_model: str = ""  # 空字符串表示使用主模型
    max_extraction_tokens: int = 4000
    # 去重配置
    dedup_similarity_threshold: float = 0.85
    # 生命周期配置
    hotness_half_life_days: float = 7.0
    cold_threshold: float = 0.1
    min_age_days_for_archive: int = 30
    # 注入配置
    injection_token_budget: int = 2000

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "enabled": self.enabled,
            "storage_dir": self.storage_dir,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "extraction_model": self.extraction_model,
            "max_extraction_tokens": self.max_extraction_tokens,
            "dedup_similarity_threshold": self.dedup_similarity_threshold,
            "hotness_half_life_days": self.hotness_half_life_days,
            "cold_threshold": self.cold_threshold,
            "min_age_days_for_archive": self.min_age_days_for_archive,
            "injection_token_budget": self.injection_token_budget,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryConfig":
        """从字典反序列化。"""
        return cls(
            enabled=data.get("enabled", True),
            storage_dir=data.get("storage_dir", ""),
            embedding_provider=data.get("embedding_provider", ""),
            embedding_model=data.get("embedding_model", "text-embedding-3-small"),
            embedding_dimension=data.get("embedding_dimension", 1536),
            extraction_model=data.get("extraction_model", ""),
            max_extraction_tokens=data.get("max_extraction_tokens", 4000),
            dedup_similarity_threshold=data.get("dedup_similarity_threshold", 0.85),
            hotness_half_life_days=data.get("hotness_half_life_days", 7.0),
            cold_threshold=data.get("cold_threshold", 0.1),
            min_age_days_for_archive=data.get("min_age_days_for_archive", 30),
            injection_token_budget=data.get("injection_token_budget", 2000),
        )


@dataclass
class ContextConfig:
    """上下文管理配置。"""

    # Snip 配置
    snip_keep_recent_tools: int = 5
    snip_threshold_chars: int = 1500
    snip_min_lines: int = 6
    # 摘要配置
    summarize_keep_recent_turns: int = 20
    # Token 估算 fudge 因子
    token_fudge_factor: float = 1.1
    # 新增：摘要缓存
    enable_summary_cache: bool = True
    # 新增：归档配置
    archive_on_compress: bool = True
    archive_dir: str = ""  # 默认 ~/.dogcode/sessions

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "snip_keep_recent_tools": self.snip_keep_recent_tools,
            "snip_threshold_chars": self.snip_threshold_chars,
            "snip_min_lines": self.snip_min_lines,
            "summarize_keep_recent_turns": self.summarize_keep_recent_turns,
            "token_fudge_factor": self.token_fudge_factor,
            "enable_summary_cache": self.enable_summary_cache,
            "archive_on_compress": self.archive_on_compress,
            "archive_dir": self.archive_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextConfig":
        """从字典反序列化。"""
        return cls(
            snip_keep_recent_tools=data.get("snip_keep_recent_tools", 5),
            snip_threshold_chars=data.get("snip_threshold_chars", 1500),
            snip_min_lines=data.get("snip_min_lines", 6),
            summarize_keep_recent_turns=data.get("summarize_keep_recent_turns", 20),
            token_fudge_factor=data.get("token_fudge_factor", 1.1),
            enable_summary_cache=data.get("enable_summary_cache", True),
            archive_on_compress=data.get("archive_on_compress", True),
            archive_dir=data.get("archive_dir", ""),
        )
