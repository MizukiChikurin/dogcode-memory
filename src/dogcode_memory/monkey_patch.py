"""Monkey-patch 工具 - 零侵入自动接入 dogcode-memory。

在不修改 ReuleauxCoder-ezcode 源码的前提下，通过运行时方法替换实现记忆系统集成。

核心思路：
1. 包装消息构建方法，在 system prompt 中注入历史记忆
2. 包装会话保存方法，在保存后异步触发记忆提取
3. 包装会话恢复方法，在恢复时加载归档摘要和相关记忆

使用方式：
    from dogcode_memory.pipeline import MemoryPipeline
    from dogcode_memory.monkey_patch import install_memory

    pipeline = MemoryPipeline.create(storage_dir="~/.dogcode/memories", llm=llm)
    install_memory(agent, pipeline)
    # agent.chat / agent.loop._full_messages / store.save 已自动包装
"""

from __future__ import annotations

import threading
import time
import weakref
from typing import Any, Callable

from dogcode_memory.pipeline import MemoryPipeline


# 全局弱引用字典，避免循环引用导致内存泄漏
_patch_registry: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()


def _get_original(obj: Any, method_name: str) -> Callable[..., Any] | None:
    """获取对象的原始方法。"""
    patches = _patch_registry.get(obj)
    if patches and method_name in patches:
        return patches[method_name]
    return getattr(obj, method_name, None)


def _record_patch(obj: Any, method_name: str, original: Callable[..., Any]) -> None:
    """记录原始方法，便于后续卸载。"""
    if obj not in _patch_registry:
        _patch_registry[obj] = {}
    _patch_registry[obj][method_name] = original


def _has_been_patched(obj: Any, method_name: str) -> bool:
    """检查对象方法是否已被 patch。"""
    patches = _patch_registry.get(obj)
    return patches is not None and method_name in patches


# ───────────────────────────────────────────────
# 核心 Patch 函数
# ───────────────────────────────────────────────

def patch_message_builder(
    loop: Any,
    pipeline: MemoryPipeline,
    system_prompt_key: str = "system",
) -> bool:
    """
    包装消息构建方法，在 system prompt 中注入历史记忆。

    识别以下方法名（按优先级）：
    1. _full_messages
    2. build_messages
    3. _build_messages

    Args:
        loop: AgentLoop 或类似对象
        pipeline: 记忆管线
        system_prompt_key: system prompt 在消息列表中的 role 标识

    Returns:
        是否成功 patch
    """
    method_names = ["_full_messages", "build_messages", "_build_messages"]
    target_method = None
    target_name = None

    for name in method_names:
        if hasattr(loop, name) and callable(getattr(loop, name)):
            target_method = getattr(loop, name)
            target_name = name
            break

    if target_method is None:
        return False

    if _has_been_patched(loop, target_name):
        return True

    _record_patch(loop, target_name, target_method)

    # 注入状态跟踪
    injected_attr = "_dogcode_memory_injected"
    if not hasattr(loop, injected_attr):
        setattr(loop, injected_attr, False)

    def _wrapped(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        messages = target_method(*args, **kwargs)

        if not getattr(loop, injected_attr, False):
            # 获取会话上下文作为 query
            session_id = _get_session_id_from_loop(loop)
            query = _get_query_from_messages(messages)

            injection_result = pipeline.on_session_start(
                session_id=session_id or "",
                context=query,
            )

            if injection_result and injection_result.get("text"):
                # 将记忆合并到现有 system prompt，不插入额外消息
                messages = _merge_memory_into_system_prompt(
                    messages, injection_result, system_prompt_key
                )

            setattr(loop, injected_attr, True)

        return messages

    setattr(loop, target_name, _wrapped)
    return True


def patch_session_save(
    store: Any,
    pipeline: MemoryPipeline,
) -> bool:
    """
    包装会话保存方法，在保存后异步触发记忆提取。

    识别以下方法名：
    1. save
    2. persist
    3. write

    Args:
        store: SessionStore 或类似对象
        pipeline: 记忆管线

    Returns:
        是否成功 patch
    """
    method_names = ["save", "persist", "write"]
    target_method = None
    target_name = None

    for name in method_names:
        if hasattr(store, name) and callable(getattr(store, name)):
            target_method = getattr(store, name)
            target_name = name
            break

    if target_method is None:
        return False

    if _has_been_patched(store, target_name):
        return True

    _record_patch(store, target_name, target_method)

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        result = target_method(*args, **kwargs)

        # 尝试从参数中提取 session_id 和 messages
        session_id, messages = _extract_session_data(args, kwargs, store)

        if session_id and messages:
            pipeline.on_session_end_async(session_id, messages)

        return result

    setattr(store, target_name, _wrapped)
    return True


def patch_chat_method(
    agent: Any,
    pipeline: MemoryPipeline,
) -> bool:
    """
    包装 Agent.chat 方法，在首次聊天时注入记忆。

    Args:
        agent: Agent 实例
        pipeline: 记忆管线

    Returns:
        是否成功 patch
    """
    if not hasattr(agent, "chat") or not callable(getattr(agent, "chat")):
        return False

    target_name = "chat"
    if _has_been_patched(agent, target_name):
        return True

    original_chat = agent.chat
    _record_patch(agent, target_name, original_chat)

    injected_attr = "_dogcode_memory_chat_injected"
    if not hasattr(agent, injected_attr):
        setattr(agent, injected_attr, False)

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # 首次 chat 时注入记忆
        if not getattr(agent, injected_attr, False):
            session_id = _get_session_id_from_agent(agent)
            # 尝试从 args 中获取用户输入作为 query
            query = ""
            if args and isinstance(args[0], str):
                query = args[0]

            injection_result = pipeline.on_session_start(
                session_id=session_id or "",
                context=query,
            )

            # 如果有 loop，将记忆注入 system prompt
            loop = getattr(agent, "loop", None)
            if loop and injection_result and injection_result.get("text"):
                # 标记 loop 已注入，避免重复
                loop_injected = "_dogcode_memory_injected"
                if hasattr(loop, loop_injected):
                    setattr(loop, loop_injected, True)

            setattr(agent, injected_attr, True)

        return original_chat(*args, **kwargs)

    setattr(agent, target_name, _wrapped)
    return True


def patch_reset_method(
    agent: Any,
    pipeline: MemoryPipeline,
) -> bool:
    """
    包装 Agent.reset 方法，在重置时触发记忆提取。

    Args:
        agent: Agent 实例
        pipeline: 记忆管线

    Returns:
        是否成功 patch
    """
    if not hasattr(agent, "reset") or not callable(getattr(agent, "reset")):
        return False

    target_name = "reset"
    if _has_been_patched(agent, target_name):
        return True

    original_reset = agent.reset
    _record_patch(agent, target_name, original_reset)

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        # 重置前提取当前会话记忆
        session_id = _get_session_id_from_agent(agent)
        messages = _get_messages_from_agent(agent)

        if session_id and messages:
            pipeline.on_session_end_async(session_id, messages)

        # 清除注入标记
        loop = getattr(agent, "loop", None)
        if loop:
            setattr(loop, "_dogcode_memory_injected", False)
        setattr(agent, "_dogcode_memory_chat_injected", False)

        return original_reset(*args, **kwargs)

    setattr(agent, target_name, _wrapped)
    return True


# ───────────────────────────────────────────────
# 一键安装
# ───────────────────────────────────────────────

def install_memory(
    agent: Any,
    pipeline: MemoryPipeline,
    loop: Any | None = None,
    store: Any | None = None,
) -> dict[str, bool]:
    """
    一键安装记忆系统到 Agent。

    自动检测并包装以下组件：
    - agent.chat: 首次调用时注入记忆
    - agent.reset: 重置时触发记忆提取
    - loop._full_messages: 构建消息时注入 system prompt
    - store.save: 保存后异步提取记忆

    Args:
        agent: Agent 实例
        pipeline: 记忆管线
        loop: 可选的 AgentLoop 实例，默认从 agent.loop 获取
        store: 可选的 SessionStore 实例，默认从 agent.session_store 获取

    Returns:
        各组件 patch 结果字典
    """
    if loop is None:
        loop = getattr(agent, "loop", None)
    if store is None:
        store = getattr(agent, "session_store", None)

    results = {
        "chat_patched": patch_chat_method(agent, pipeline),
        "reset_patched": patch_reset_method(agent, pipeline),
        "message_builder_patched": False,
        "session_save_patched": False,
    }

    if loop is not None:
        results["message_builder_patched"] = patch_message_builder(loop, pipeline)

    if store is not None:
        results["session_save_patched"] = patch_session_save(store, pipeline)

    return results


def uninstall_memory(agent: Any, loop: Any | None = None, store: Any | None = None) -> dict[str, bool]:
    """
    卸载记忆系统，恢复原始方法。

    Args:
        agent: Agent 实例
        loop: 可选的 AgentLoop 实例
        store: 可选的 SessionStore 实例

    Returns:
        各组件卸载结果字典
    """
    results = {"chat_uninstalled": False, "reset_uninstalled": False}

    patches = _patch_registry.pop(agent, {})
    for method_name, original in patches.items():
        setattr(agent, method_name, original)
        if method_name == "chat":
            results["chat_uninstalled"] = True
        elif method_name == "reset":
            results["reset_uninstalled"] = True

    if loop is not None:
        loop_patches = _patch_registry.pop(loop, {})
        for method_name, original in loop_patches.items():
            setattr(loop, method_name, original)
            results["message_builder_uninstalled"] = True

    if store is not None:
        store_patches = _patch_registry.pop(store, {})
        for method_name, original in store_patches.items():
            setattr(store, method_name, original)
            results["session_save_uninstalled"] = True

    return results


# ───────────────────────────────────────────────
# 辅助函数
# ───────────────────────────────────────────────

def _get_session_id_from_agent(agent: Any) -> str | None:
    """从 Agent 获取会话 ID。"""
    # 尝试多种可能的属性名
    for attr in ["current_session_id", "session_id", "_session_id"]:
        val = getattr(agent, attr, None)
        if val:
            return str(val)
    return None


def _get_session_id_from_loop(loop: Any) -> str | None:
    """从 Loop 获取会话 ID。"""
    agent = getattr(loop, "agent", None)
    if agent:
        return _get_session_id_from_agent(agent)
    for attr in ["session_id", "_session_id"]:
        val = getattr(loop, attr, None)
        if val:
            return str(val)
    return None


def _get_messages_from_agent(agent: Any) -> list[dict[str, Any]]:
    """从 Agent 获取当前消息列表。"""
    context = getattr(agent, "context", None)
    if context:
        messages = getattr(context, "messages", None)
        if messages:
            return messages
    loop = getattr(agent, "loop", None)
    if loop:
        messages = getattr(loop, "messages", None)
        if messages:
            return messages
    return []


def _get_query_from_messages(messages: list[dict[str, Any]]) -> str:
    """从消息列表中提取查询文本。"""
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"])[:200]
    return ""


def _merge_memory_into_system_prompt(
    messages: list[dict[str, Any]],
    injection_result: dict[str, Any],
    system_prompt_key: str = "system",
) -> list[dict[str, Any]]:
    """将记忆内容合并到现有 system prompt，不插入额外消息。

    在 system prompt 末尾添加 [memory_meta: count=N] 标记，
    前端可通过解析该标记感知记忆注入状态。
    """
    memory_text = injection_result.get("text", "")
    if not memory_text:
        return messages

    count = injection_result.get("count", 0)
    meta_marker = f"\n\n[memory_meta: count={count}]"

    # 查找 system prompt 位置
    system_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == system_prompt_key:
            system_idx = i
            break

    new_messages = list(messages)
    if system_idx >= 0:
        # 修改现有 system prompt，将记忆内容合并到末尾
        original = new_messages[system_idx].get("content", "") or ""
        new_content = (
            f"{original}\n\n"
            f"--- Cross-session context ---\n"
            f"{memory_text}"
            f"{meta_marker}"
        )
        new_messages[system_idx] = {
            **new_messages[system_idx],
            "role": system_prompt_key,
            "content": new_content,
        }
    else:
        # 没有 system prompt，创建一条包含记忆的新 system 消息
        new_messages.insert(0, {
            "role": system_prompt_key,
            "content": f"--- Cross-session context ---\n{memory_text}{meta_marker}",
        })

    return new_messages


def _extract_session_data(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    store: Any,
) -> tuple[str | None, list[dict[str, Any]]]:
    """从 save 方法参数中提取 session_id 和 messages。"""
    session_id = None
    messages = []

    # 尝试从 args 中提取
    if len(args) >= 1 and isinstance(args[0], str):
        session_id = args[0]
    if len(args) >= 2 and isinstance(args[1], list):
        messages = args[1]

    # 尝试从 kwargs 中提取
    if not session_id:
        session_id = kwargs.get("session_id") or kwargs.get("id")
    if not messages:
        messages = kwargs.get("messages", [])
        if not messages:
            messages = kwargs.get("session_data", {}).get("messages", [])

    # 尝试从 store 的当前状态推断
    if not session_id:
        session_id = getattr(store, "current_session_id", None)
        session_id = session_id or getattr(store, "_current_session_id", None)

    return (str(session_id) if session_id else None, messages)
