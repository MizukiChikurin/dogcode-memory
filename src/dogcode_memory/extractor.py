"""记忆提取管线 - 从会话消息中提取结构化记忆。"""

from __future__ import annotations

import json
import os
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

        校准策略：
        1. 收集对话中真实出现的工具名、文件路径、代码片段等证据。
        2. tools 类型：验证 tool_name 是否在真实调用列表中；若不在，大幅降低置信度（防止 LLM 编造工具名）。
        3. errors 类型：验证错误消息是否在对话中有近似出现；若无，降低置信度。
        4. project 类型：验证引用的文件路径是否真实出现；每个未验证路径降低置信度。
        5. 所有类型：扫描 content 和 fields 中引用的文件路径，未在对话中出现则降低置信度。
        6. 置信度低于 0.3 的候选直接过滤。
        """
        all_files, all_tools, all_error_keywords = self._collect_evidence(messages)

        calibrated: list[CandidateMemory] = []
        for c in candidates:
            c.confidence = self._apply_calibration(c, all_files, all_tools, all_error_keywords)
            if c.confidence >= 0.3:
                calibrated.append(c)
        return calibrated

    @staticmethod
    def _collect_evidence(
        messages: list[dict[str, Any]],
    ) -> tuple[set[str], set[str], set[str]]:
        """
        从对话中收集可用于校准的真实证据。

        返回 (all_files, all_tools, all_error_keywords) 三个集合。
        """
        all_files: set[str] = set()
        all_tools: set[str] = set()
        all_error_keywords: set[str] = set()

        for m in messages:
            content = m.get("content", "") or ""

            # 从内容中提取文件路径（使用与候选校准相同的提取逻辑）
            content_paths = MemoryExtractor._extract_paths_from_text(content)
            for path in content_paths:
                all_files.add(path)
                if "/" in path or "\\" in path:
                    all_files.add(os.path.basename(path))

            # 从 tool_calls 中提取工具名和参数中的文件路径
            tool_calls = m.get("tool_calls", [])
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                if name:
                    all_tools.add(name)
                # 从工具参数中提取路径
                args = fn.get("arguments", "") or fn.get("args", "")
                args_str = json.dumps(args) if not isinstance(args, str) else args
                arg_paths = MemoryExtractor._extract_paths_from_text(args_str)
                for path in arg_paths:
                    all_files.add(path)
                    if "/" in path or "\\" in path:
                        all_files.add(os.path.basename(path))

            # 提取错误类关键词（Exception/Error/Traceback 等附近文本）
            error_patterns = [
                r"(?:Exception|Error|Traceback)[\s\w]*?:\s*([^\n\r]+)",
                r"(?:failed|failure|error|exception)[\s:]+([^\n\r]{10,200})",
            ]
            for pattern in error_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    snippet = match.group(1).strip()
                    if snippet:
                        all_error_keywords.add(snippet.lower())

            # 从 tool_status / status 中提取状态信息
            tool_status = m.get("tool_status") or m.get("status")
            if tool_status and isinstance(tool_status, str):
                all_error_keywords.add(tool_status.lower())

        return all_files, all_tools, all_error_keywords

    @staticmethod
    def _apply_calibration(
        candidate: CandidateMemory,
        all_files: set[str],
        all_tools: set[str],
        all_error_keywords: set[str],
    ) -> float:
        """
        对单个候选应用校准规则，返回校准后的置信度。
        """
        confidence = candidate.confidence
        text_to_check = " ".join([
            candidate.abstract,
            candidate.overview,
            candidate.content,
            *[str(v) for v in candidate.fields.values() if isinstance(v, str)],
        ]).lower()

        # 规则 1：tools 类型 —— 验证 tool_name
        if candidate.type == "tools":
            tool_name = str(candidate.fields.get("tool_name", "")).strip()
            if tool_name and tool_name not in all_tools:
                # LLM 编造了工具名，置信度减半
                confidence *= 0.5

        # 规则 2：errors 类型 —— 验证错误依据
        if candidate.type == "errors":
            error_msg = str(candidate.fields.get("error_message", "")).strip().lower()
            if error_msg and error_msg not in all_error_keywords:
                # 检查是否有部分匹配
                partial_match = any(error_msg in kw or kw in error_msg for kw in all_error_keywords)
                if not partial_match:
                    confidence *= 0.6

        # 规则 3：所有类型 —— 验证引用的文件路径真实性
        # 从候选文本中提取路径，与 all_files 对比
        paths_in_candidate = MemoryExtractor._extract_paths_from_text(text_to_check)
        for path in paths_in_candidate:
            # 检查完整路径或文件名是否在证据集中
            basename = os.path.basename(path)
            if path not in all_files and basename not in all_files:
                confidence *= 0.85
                if confidence < 0.3:
                    break

        return round(confidence, 3)

    @staticmethod
    def _extract_paths_from_text(text: str) -> set[str]:
        """
        从文本中提取可能的文件路径，自动去重 basename 避免重复惩罚。
        """
        paths: set[str] = set()
        basenames_from_full: set[str] = set()

        # 匹配带斜杠的路径模式（优先）
        for match in re.finditer(r"[\w\-]+(?:[/\\][\w\-]+)+(?:\.\w{1,10})?", text):
            path = match.group()
            if "/" in path or "\\" in path:
                paths.add(path)
                basenames_from_full.add(os.path.basename(path))

        # 匹配单独的文件名（带后缀），排除已被完整路径覆盖的 basename
        # 使用 \b 避免匹配 "index.ts" 当它是 "index.tsx" 的一部分
        for match in re.finditer(r"\b[\w\-]+\.(?:py|js|ts|java|go|rs|json|yaml|yml|md|txt|sql|sh)\b", text):
            filename = match.group()
            if filename not in basenames_from_full:
                paths.add(filename)

        return paths
