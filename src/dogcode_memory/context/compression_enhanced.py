"""归档压缩策略 - Layer 2.5 压缩增强。

当 Layer 2（LLM 摘要）缓存命中率低时，触发归档压缩：
1. 将旧消息段移入 SessionArchiver
2. 保留结构化摘要 + 最近消息
3. 触发异步记忆提取（由上层 Pipeline 完成）

与 Layer 3 HardCollapse 的区别：
- HardCollapse：只保留极简摘要，丢弃决策细节
- ArchiveCompress：保留结构化摘要（文件路径、决策、错误等）
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from dogcode_memory.context.summary_cache import SummaryCache


class LLMClient(Protocol):
    """LLM 客户端协议。"""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """发送聊天请求，返回包含 content 属性的对象。"""
        ...


STRUCTURED_SUMMARY_PROMPT = """\
请对以下对话生成结构化摘要，保留关键技术细节以便后续恢复上下文。

输出格式要求（JSON）：
{
  "primary_request": "用户的主要请求和意图",
  "key_concepts": ["技术概念1", "技术概念2"],
  "files_and_code": [
    {"path": "文件路径", "importance": "为什么重要", "changes": "修改摘要"}
  ],
  "decisions": ["关键决策1", "关键决策2"],
  "errors": ["遇到的错误及状态"],
  "pending_tasks": ["待办任务1"],
  "current_work": "当前正在做的工作"
}

注意：
- 文件路径必须完整保留
- 代码片段保留关键签名
- 只输出 JSON，不要其他内容
"""


def generate_structured_summary(
    messages: list[dict[str, Any]],
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """
    生成结构化摘要：按 Turn 分段，保留文件路径、决策、错误。

    Args:
        messages: 消息列表
        llm: 可选的 LLM 客户端，提供时生成高质量结构化摘要

    Returns:
        结构化摘要字典
    """
    # 尝试使用 LLM 生成
    if llm:
        try:
            flat = _flatten_messages(messages)
            resp = llm.chat(
                messages=[
                    {"role": "system", "content": STRUCTURED_SUMMARY_PROMPT},
                    {"role": "user", "content": flat[:15000]},
                ],
            )
            content = resp.content if hasattr(resp, "content") else str(resp)
            parsed = _parse_json_from_text(content)
            if parsed and isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Fallback：基于规则提取
    return _extract_structured_fallback(messages)


class ArchiveCompressStrategy:
    """
    Layer 2.5 归档压缩策略。

    触发条件：
    - 上下文超过 75% 预算
    - Layer 2（CachedSummarize）缓存命中率低或已耗尽

    行为：
    1. 识别可归档的旧消息段
    2. 调用 SessionArchiver 归档
    3. 用结构化摘要替换已归档消息
    4. 保留最近 N 轮用户消息不变
    """

    def __init__(
        self,
        archiver: Any | None = None,
        summary_cache: SummaryCache | None = None,
        keep_recent_turns: int = 20,
    ):
        """
        初始化策略。

        Args:
            archiver: SessionArchiver 实例
            summary_cache: 摘要缓存实例
            keep_recent_turns: 保留的最近用户轮数
        """
        self._archiver = archiver
        self._summary_cache = summary_cache or SummaryCache()
        self._keep_recent_turns = keep_recent_turns
        self._hit_threshold = 0.3  # 缓存命中率低于此值认为"命中率低"

    def should_trigger(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int,
        max_tokens: int,
    ) -> bool:
        """
        判断是否触发归档压缩。

        触发条件：
        1. 当前 token 数超过 75% 预算
        2. 且（缓存未命中 或 缓存命中率低）

        Args:
            messages: 当前活跃消息
            current_tokens: 当前估算 token 数
            max_tokens: 最大 token 预算

        Returns:
            是否触发
        """
        threshold = int(max_tokens * 0.75)
        if current_tokens <= threshold:
            return False

        # 检查缓存是否命中
        cached = self._summary_cache.get(messages)
        if cached is None:
            return True

        # 缓存命中但内容过短，说明质量不够
        if len(cached) < 200:
            return True

        return False

    def compress(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        llm: LLMClient | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        执行归档压缩。

        Args:
            messages: 活跃消息列表（会被原地修改）
            session_id: 会话 ID
            llm: 可选的 LLM 客户端

        Returns:
            (是否执行了压缩, 压缩信息字典)
        """
        split_index = self._find_split_index(messages)
        if split_index <= 0 or split_index >= len(messages):
            return False, {"reason": "no_archivable_messages"}

        old_messages = messages[:split_index]
        tail = messages[split_index:]

        # 生成结构化摘要
        structured = generate_structured_summary(old_messages, llm)
        text_summary = structured.get("primary_request", "") or _fallback_summary(old_messages)

        # 写入缓存
        self._summary_cache.put(old_messages, text_summary)

        # 归档（如果提供了 archiver）
        archive_record = None
        if self._archiver is not None:
            try:
                archive_record = self._archiver.archive(
                    session_id=session_id,
                    messages=old_messages,
                    metadata={"compression_layer": "archive_compress"},
                )
            except Exception:
                archive_record = None

        # 替换活跃消息
        summary_msg = {
            "role": "user",
            "content": (
                "[Archived context - see structured summary below]\n"
                f"{text_summary}\n\n"
                f"Files: {', '.join(structured.get('files_and_code', [])[:10] or [])}\n"
                f"Decisions: {', '.join(structured.get('decisions', [])[:5] or [])}"
            ),
        }
        assistant_ack = {
            "role": "assistant",
            "content": "Context archived. I have the summary from earlier conversation.",
        }

        messages.clear()
        messages.append(summary_msg)
        messages.append(assistant_ack)
        messages.extend(tail)

        info = {
            "reason": "archive_compress",
            "archived_message_count": len(old_messages),
            "kept_message_count": len(tail),
            "has_structured_summary": bool(structured),
            "archive_record": archive_record.to_dict() if archive_record else None,
        }
        return True, info

    def _find_split_index(self, messages: list[dict[str, Any]]) -> int:
        """返回分割索引，保留最近 N 轮用户消息。"""
        if self._keep_recent_turns <= 0:
            return len(messages)

        user_turn_starts = [
            i for i, msg in enumerate(messages)
            if msg.get("role") == "user"
        ]
        if len(user_turn_starts) <= self._keep_recent_turns:
            return 0
        return user_turn_starts[-self._keep_recent_turns]


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """将消息列表扁平化为字符串。"""
    parts = []
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        text = m.get("content", "") or ""
        if text:
            parts.append(f"[{i}][{role}] {text[:600]}")
        tool_calls = m.get("tool_calls")
        if tool_calls:
            parts.append(f"[{i}][tool_calls] {str(tool_calls)[:400]}")
    return "\n".join(parts)


def _parse_json_from_text(text: str) -> Any:
    """从文本中提取 JSON 对象。"""
    import json

    text = text.strip()
    # 尝试直接解析
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # 从代码块中提取
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        inner = match.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            pass

    # 查找最外层花括号
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return None


def _extract_structured_fallback(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """基于规则提取结构化摘要（无 LLM fallback）。"""
    files_seen: set[str] = set()
    errors: list[str] = []
    decisions: list[str] = []
    pending: list[str] = []
    concepts: set[str] = set()

    file_pattern = re.compile(r"[\w./\\\-]+\.\w{1,10}")
    error_pattern = re.compile(r"(?:error|exception|failed|traceback)[\s:]*([^\n]{10,200})", re.IGNORECASE)

    for m in messages:
        text = m.get("content", "") or ""
        # 提取文件路径
        for match in file_pattern.finditer(text):
            path = match.group()
            if "/" in path or "\\" in path or "." in path:
                files_seen.add(path)

        # 提取错误
        for match in error_pattern.finditer(text):
            errors.append(match.group(1).strip()[:150])

        # 提取决策（简单启发式）
        lower = text.lower()
        if any(k in lower for k in ("decided", "decision", "chosen", "selected", "will use")):
            line = text.strip()[:200]
            if line:
                decisions.append(line)

        # 提取待办
        if any(k in lower for k in ("pending", "todo", "next step", "need to")):
            line = text.strip()[:200]
            if line:
                pending.append(line)

        # 技术概念（简单关键词）
        for kw in ("python", "typescript", "react", "api", "database", "docker", "kubernetes"):
            if kw in lower:
                concepts.add(kw)

    # 提取主要请求（第一条用户消息）
    primary_request = ""
    for m in messages:
        if m.get("role") == "user":
            primary_request = (m.get("content", "") or "")[:300]
            break

    return {
        "primary_request": primary_request,
        "key_concepts": sorted(concepts),
        "files_and_code": sorted(files_seen)[:20],
        "decisions": decisions[:5],
        "errors": errors[:5],
        "pending_tasks": pending[:5],
        "current_work": messages[-1].get("content", "")[:200] if messages else "",
    }


def _fallback_summary(messages: list[dict[str, Any]]) -> str:
    """极简 fallback 摘要。"""
    structured = _extract_structured_fallback(messages)
    parts = []
    if structured["primary_request"]:
        parts.append(f"Request: {structured['primary_request']}")
    if structured["files_and_code"]:
        parts.append(f"Files: {', '.join(structured['files_and_code'][:10])}")
    if structured["decisions"]:
        parts.append(f"Decisions: {', '.join(structured['decisions'][:3])}")
    return "\n".join(parts) or "(no extractable context)"
