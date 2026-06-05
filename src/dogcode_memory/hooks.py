"""零侵入接入封装 - 让 dogcode-memory 在不修改服务端源码的情况下接入。

提供两种接入路径：
1. Hook 适配器：实现 ReuleauxCoder HookBase 接口，注册到 agent.hook_registry
2. Monkey-patch 工具：自动包装 agent.chat、loop._full_messages、store.save

使用方式：
    # 方案 A（服务端已初始化 HookRegistry）
    from dogcode_memory.hooks import MemoryLifecycleHook
    hook = MemoryLifecycleHook(pipeline)
    agent.hook_registry.register(HookPoint.SESSION_START, hook)
    agent.hook_registry.register(HookPoint.SESSION_SAVE, hook)

    # 方案 B（服务端完全不改代码）
    from dogcode_memory.monkey_patch import install_memory_on_agent
    install_memory_on_agent(agent, pipeline)
"""

from __future__ import annotations

from typing import Any, Protocol

from dogcode_memory.pipeline import MemoryPipeline


# ───────────────────────────────────────────────
# 方案 A：Hook 适配器
# ───────────────────────────────────────────────

try:
    # 尝试导入 ReuleauxCoder 的 Hook 基类
    from reuleauxcoder.domain.hooks.base import ObserverHook
    from reuleauxcoder.domain.hooks.types import (
        HookPoint,
        RunnerShutdownContext,
        SessionSaveContext,
        SessionStartContext,
    )

    _HAS_REULEAUX_HOOKS = True
except ImportError:
    _HAS_REULEAUX_HOOKS = False
    # 定义占位类，避免导入错误
    class ObserverHook:  # type: ignore[no-redef]
        """ReuleauxCoder ObserverHook 的占位符。"""

        name = ""
        priority = 0

        def run(self, context: Any) -> None:
            pass

    SessionStartContext = Any  # type: ignore[misc, no-redef]
    SessionSaveContext = Any  # type: ignore[misc, no-redef]
    RunnerShutdownContext = Any  # type: ignore[misc, no-redef]
    HookPoint = Any  # type: ignore[misc, no-redef]


class MemoryLifecycleHook(ObserverHook):
    """
    ReuleauxCoder ObserverHook 适配器。

    监听 SESSION_START 和 SESSION_SAVE 事件：
    - SESSION_START: 注入历史记忆到当前会话上下文
    - SESSION_SAVE: 异步提取长期记忆
    - RUNNER_SHUTDOWN: 关闭异步任务池
    """

    def __init__(
        self,
        pipeline: MemoryPipeline,
        name: str = "dogcode_memory_lifecycle",
        priority: int = 50,
    ):
        """
        初始化 Hook。

        Args:
            pipeline: 记忆管线实例
            name: Hook 名称
            priority: Hook 优先级（越高越先执行）
        """
        self.pipeline = pipeline
        self.name = name
        self.priority = priority
        self._injected_sessions: set[str] = set()

    def run(self, context: Any) -> None:
        """Hook 入口。"""
        ctx_type = type(context).__name__

        # 灵活的上下文匹配：按类型名或属性判断
        if ctx_type == "SessionStartContext" or (
            hasattr(context, "session_id") and not hasattr(context, "session_data")
        ):
            self._on_session_start(context)
        elif ctx_type == "SessionSaveContext" or hasattr(context, "session_data"):
            self._on_session_save(context)
        elif ctx_type == "RunnerShutdownContext" or ctx_type.endswith("ShutdownContext"):
            self._on_shutdown(context)

    def _on_session_start(self, context: SessionStartContext) -> None:
        """会话开始时注入记忆。"""
        session_id = getattr(context, "session_id", None) or ""
        if not session_id or session_id in self._injected_sessions:
            return

        self._injected_sessions.add(session_id)

        # 尝试从上下文中获取当前工作区信息作为 query
        metadata = getattr(context, "metadata", {}) or {}
        query = metadata.get("workspace_name", "")

        injection_result = self.pipeline.on_session_start(
            session_id=session_id or "",
            context=query,
        )

        if injection_result and injection_result.get("text"):
            # 将记忆注入到 metadata 中，供后续使用
            metadata["dogcode_memory_injected"] = injection_result["text"]
            metadata["dogcode_memory_uris"] = injection_result.get("uris", [])
            metadata["dogcode_memory_count"] = injection_result.get("count", 0)

    def _on_session_save(self, context: SessionSaveContext) -> None:
        """会话保存时异步提取记忆。"""
        session_id = getattr(context, "session_id", None) or ""
        if not session_id:
            return

        session_data = getattr(context, "session_data", {}) or {}
        messages = session_data.get("messages", [])

        if not messages:
            return

        # 异步提取，不阻塞保存
        self.pipeline.on_session_end_async(
            session_id=session_id,
            messages=messages,
        )

    def _on_shutdown(self, context: RunnerShutdownContext) -> None:
        """Runner 关闭时清理资源。"""
        self.pipeline.wait_all_tasks(timeout=30.0)
        self.pipeline.shutdown_async_pool()

    @classmethod
    def create_from_config(cls, config: Any) -> "MemoryLifecycleHook":
        """通过 Config 创建 Hook（用于 ReuleauxCoder 自动发现）。"""
        memory_config = getattr(config, "memory", None)
        if memory_config is None:
            from dogcode_memory.config import MemoryConfig
            memory_config = MemoryConfig()

        pipeline = MemoryPipeline.create(
            storage_dir=getattr(memory_config, "storage_dir", "~/.dogcode/memories"),
            llm=getattr(config, "llm", None),
            config=memory_config,
        )
        return cls(pipeline=pipeline)


# ───────────────────────────────────────────────
# 便捷注册函数
# ───────────────────────────────────────────────

def register_memory_hooks(
    agent: Any,
    pipeline: MemoryPipeline,
) -> MemoryLifecycleHook:
    """
    将记忆生命周期 Hook 注册到 Agent。

    Args:
        agent: 包含 hook_registry 的 Agent 实例
        pipeline: 记忆管线

    Returns:
        注册的 Hook 实例
    """
    hook = MemoryLifecycleHook(pipeline)
    registry = getattr(agent, "hook_registry", None)
    if registry is None:
        raise AttributeError("Agent has no hook_registry attribute")

    if _HAS_REULEAUX_HOOKS:
        registry.register(HookPoint.SESSION_START, hook)
        registry.register(HookPoint.SESSION_SAVE, hook)
        registry.register(HookPoint.RUNNER_SHUTDOWN, hook)
    else:
        raise RuntimeError(
            "ReuleauxCoder hooks not available. "
            "Use monkey_patch.install_memory_on_agent instead."
        )

    return hook
