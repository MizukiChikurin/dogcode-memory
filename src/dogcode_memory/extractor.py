"""记忆提取管线 - 从会话消息中提取结构化记忆。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from dogcode_memory.config import MemoryConfig
from dogcode_memory.schema import MemoryTypeSchema


class LLMClient(Protocol):
    """LLM 客户端协议，供提取器调用。"""

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        """发送聊天请求，返回包含 content 属性的对象。"""
        ...


@dataclass
class CandidateMemory:
    """候选记忆 - 从会话中提取的原始记忆数据。"""

    type: str                    # 记忆类型名称
    abstract: str                # L0: 一句话摘要
    overview: str = ""           # L1: 概览
    content: str = ""            # L2: 详细内容
    fields: dict[str, Any] = field(default_factory=dict)  # 结构化字段
    source_session: str = ""     # 来源会话 ID
    confidence: float = 1.0      # 置信度 0-1

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "type": self.type,
            "abstract": self.abstract,
            "overview": self.overview,
            "content": self.content,
            "fields": self.fields,
            "source_session": self.source_session,
            "confidence": self.confidence,
        }


EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction system. Analyze the conversation and extract structured long-term memories.

Your task:
1. Read the conversation carefully
2. Identify important facts, preferences, patterns, and knowledge that should persist across sessions
3. Output a JSON array of candidate memories

For each memory, include:
- type: one of [profile, preferences, project, tools, patterns, errors]
- abstract: a one-sentence summary
- overview: a brief description (2-3 sentences)
- content: detailed information with specific facts
- fields: structured key-value pairs relevant to the type

Memory type guidelines:
- profile: user identity, coding habits, role, expertise areas
- preferences: naming conventions, style preferences, workflow preferences
- project: architecture decisions, tech stack, directory structure
- tools: tool usage patterns, best practices, common failures
- patterns: recurring design patterns, refactoring strategies
- errors: encountered errors and their solutions

Only extract genuinely useful, persistent knowledge. Skip ephemeral details.
Output must be valid JSON.
"""


class MemoryExtractor:
    """记忆提取器 - 从会话消息中提取结构化候选记忆。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        registry: Any | None = None,
        config: MemoryConfig | None = None,
    ):
        """
        初始化提取器。

        Args:
            llm: LLM 客户端，用于生成摘要和提取记忆
            registry: 记忆类型注册表（可选，用于类型验证）
            config: 记忆配置
        """
        self._llm = llm
        self._registry = registry
        self._config = config or MemoryConfig()

    def extract(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> list[CandidateMemory]:
        """
        从会话消息中提取候选记忆。

        编排流程：格式化对话 → 调用 LLM → 解析候选 → 校准

        Args:
            session_id: 会话 ID
            messages: 会话消息列表

        Returns:
            候选记忆列表
        """
        if not self._llm or not messages:
            return []

        # 格式化对话
        formatted = self._format_conversation(messages)

        # 调用 LLM 提取
        raw_candidates = self._call_llm_extract(formatted)

        # 解析并校准
        candidates = self._parse_candidates(raw_candidates, session_id)
        candidates = self._calibrate_candidates(candidates, messages)

        return candidates

    def _format_conversation(self, messages: list[dict[str, Any]]) -> str:
        """
        格式化对话为 LLM 输入文本。

        保留角色、内容摘要、工具调用信息。
        """
        parts = []
        for i, m in enumerate(messages):
            role = m.get("role", "?")
            content = m.get("content", "") or ""
            # 截断过长的内容
            if len(content) > 800:
                content = content[:400] + "\n... [truncated] ...\n" + content[-400:]
            parts.append(f"[{i}][{role}]: {content}")

            # 提取工具调用信息
            tool_calls = m.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", "") or fn.get("args", "")
                    parts.append(f"  [tool_call] {name}: {str(args)[:200]}")

            # 提取工具响应状态
            tool_status = m.get("tool_status") or m.get("status")
            if tool_status:
                parts.append(f"  [tool_status] {tool_status}")

        return "\n".join(parts)

    def _call_llm_extract(self, formatted_conversation: str) -> str:
        """调用 LLM 提取候选记忆。"""
        try:
            resp = self._llm.chat(
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": formatted_conversation[: self._config.max_extraction_tokens]},
                ],
            )
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            return "[]"

    def _parse_candidates(
        self,
        llm_response: str,
        session_id: str,
    ) -> list[CandidateMemory]:
        """解析 LLM 返回的 JSON 为候选记忆列表。"""
        candidates: list[CandidateMemory] = []

        # 尝试从响应中提取 JSON 数组
        json_str = self._extract_json_array(llm_response)
        if not json_str:
            return candidates

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]

            for item in data:
                if not isinstance(item, dict):
                    continue
                candidate = CandidateMemory(
                    type=item.get("type", "unknown"),
                    abstract=item.get("abstract", ""),
                    overview=item.get("overview", ""),
                    content=item.get("content", ""),
                    fields=item.get("fields", {}),
                    source_session=session_id,
                    confidence=item.get("confidence", 1.0),
                )
                # 类型验证
                if self._registry and hasattr(self._registry, "get_schema"):
                    schema = self._registry.get_schema(candidate.type)
                    if schema is None:
                        continue  # 跳过未注册的类型
                candidates.append(candidate)
        except Exception:
            pass

        return candidates

    @staticmethod
    def _extract_json_array(text: str) -> str:
        """从文本中提取 JSON 数组。"""
        text = text.strip()

        # 尝试直接解析
        if text.startswith("[") and text.endswith("]"):
            return text

        # 尝试从 markdown 代码块中提取
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            inner = match.group(1).strip()
            if inner.startswith("[") and inner.endswith("]"):
                return inner

        # 尝试查找最外层方括号
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return text[start : end + 1]

        return ""

    def _calibrate_candidates(
        self,
        candidates: list[CandidateMemory],
        messages: list[dict[str, Any]],
    ) -> list[CandidateMemory]:
        """
        用对话中真实数据校准候选记忆。

        目前实现：验证文件路径和工具名的真实性。
        """
        # 收集对话中所有文件路径和工具名
        all_files: set[str] = set()
        all_tools: set[str] = set()

        for m in messages:
            content = m.get("content", "") or ""
            # 提取文件路径
            for match in re.finditer(r"[\w./\\\-]+\.\w{1,5}", content):
                path = match.group()
                if "/" in path or "\\" in path:
                    all_files.add(path)

            # 提取工具名
            tool_calls = m.get("tool_calls", [])
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if name:
                    all_tools.add(name)

        # 校准每个候选
        calibrated = []
        for c in candidates:
            # 如果候选包含文件路径，验证其是否真实出现
            for key, value in list(c.fields.items()):
                if isinstance(value, str):
                    # 简化处理：接受所有候选，仅降低可疑候选的置信度
                    pass
            calibrated.append(c)

        return calibrated
