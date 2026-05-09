"""合并策略 - 实现字段级的多种合并操作。"""

from __future__ import annotations

import re
from typing import Any

from dogcode_memory.schema import MemoryField, MemoryTypeSchema, MergeOp


class SearchReplaceBlock:
    """SEARCH/REPLACE 块。"""

    def __init__(self, search: str, replace: str):
        self.search = search
        self.replace = replace


def patch_merge(current: str | None, patch_blocks: list[SearchReplaceBlock]) -> str:
    """
    SEARCH/REPLACE 增量合并。

    每个 patch_block 包含 search 和 replace 文本。
    按顺序在 current 中搜索并替换。

    Args:
        current: 当前字段值
        patch_blocks: SEARCH/REPLACE 块列表

    Returns:
        合并后的字符串
    """
    if current is None:
        current = ""

    result = current
    for block in patch_blocks:
        if not block.search.strip():
            # 空 search 表示追加
            result = result.rstrip() + "\n\n" + block.replace.strip() + "\n"
            continue

        # 尝试精确匹配
        if block.search in result:
            result = result.replace(block.search, block.replace, 1)
            continue

        # 尝试模糊匹配（忽略首尾空白差异）
        normalized_search = block.search.strip()
        normalized_result = result.strip()
        if normalized_search in normalized_result:
            # 找到位置并替换
            idx = normalized_result.find(normalized_search)
            result = result[:idx] + block.replace + result[idx + len(normalized_search):]
            continue

        # 如果都未匹配，默认追加
        result = result.rstrip() + "\n\n" + block.replace.strip() + "\n"

    return result


def sum_merge(current: int | float | None, increment: int | float | None) -> int | float:
    """
    数值累加合并。

    Args:
        current: 当前值
        increment: 增量值

    Returns:
        累加后的值
    """
    current = current or 0
    increment = increment or 0
    if isinstance(current, float) or isinstance(increment, float):
        return float(current) + float(increment)
    return int(current) + int(increment)


def immutable_merge(current: Any, new_value: Any) -> Any:
    """
    不可变合并 - 首次写入后保持不变。

    Args:
        current: 当前值
        new_value: 新值

    Returns:
        如果 current 已存在则返回 current，否则返回 new_value
    """
    return current if current is not None else new_value


def append_merge(current: str | list | None, new_content: str | list) -> str | list:
    """
    追加合并。

    Args:
        current: 当前值（字符串或列表）
        new_content: 新内容

    Returns:
        追加后的值
    """
    if current is None:
        return new_content

    if isinstance(current, list) and isinstance(new_content, list):
        return current + new_content

    # 统一按字符串处理
    current_str = str(current).rstrip()
    new_str = str(new_content).strip()
    if not current_str:
        return new_str
    if not new_str:
        return current_str
    return current_str + "\n\n" + new_str + "\n"


def merge_memory(
    existing_fields: dict[str, Any],
    candidate_fields: dict[str, Any],
    schema: MemoryTypeSchema,
) -> dict[str, Any]:
    """
    按 Schema 定义的字段策略合并整条记忆。

    Args:
        existing_fields: 已有记忆的字段值字典
        candidate_fields: 候选记忆的字段值字典
        schema: 记忆类型 Schema

    Returns:
        合并后的字段值字典
    """
    result = dict(existing_fields)

    for field_name, new_value in candidate_fields.items():
        field_def = schema.get_field(field_name)
        if field_def is None:
            # 未定义的字段，默认使用 append 策略
            result[field_name] = append_merge(result.get(field_name), new_value)
            continue

        current_value = result.get(field_name)

        if field_def.merge_op == MergeOp.PATCH:
            # 将新值解析为 SearchReplaceBlock 列表或直接替换
            if isinstance(new_value, list) and new_value:
                blocks = []
                for item in new_value:
                    if isinstance(item, dict) and "search" in item and "replace" in item:
                        blocks.append(SearchReplaceBlock(item["search"], item["replace"]))
                    elif isinstance(item, SearchReplaceBlock):
                        blocks.append(item)
                if blocks:
                    result[field_name] = patch_merge(current_value, blocks)
                else:
                    result[field_name] = new_value if current_value is None else append_merge(current_value, new_value)
            else:
                result[field_name] = new_value if current_value is None else append_merge(current_value, new_value)

        elif field_def.merge_op == MergeOp.SUM:
            result[field_name] = sum_merge(current_value, new_value)

        elif field_def.merge_op == MergeOp.IMMUTABLE:
            result[field_name] = immutable_merge(current_value, new_value)

        elif field_def.merge_op == MergeOp.APPEND:
            result[field_name] = append_merge(current_value, new_value)

    return result
