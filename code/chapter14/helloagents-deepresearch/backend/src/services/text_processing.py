"""Utility helpers for normalizing agent generated text."""

from __future__ import annotations

import json
import re

# 原有 HelloAgents 文本工具调用协议：[TOOL_CALL:tool_name:{参数}]
_TOOL_CALL_PATTERN = re.compile(r"\[TOOL_CALL:[^\]]+\]", re.IGNORECASE)

# DeepSeek DSML 标记兼容半角/全角竖线，1~2 个均可（|、||、｜、｜｜）
_DSML_BARS = r"[|｜]{1,2}"

# DSML invoke 开标签（完整标签文本，属性值内不含 >）
_DSML_INVOKE_OPEN_PATTERN = re.compile(
    rf"<{_DSML_BARS}DSML{_DSML_BARS}\s*invoke\b[^>]*>",
    re.IGNORECASE,
)

# 任意 DSML 开/闭标签（calls / invoke / parameter / 无名闭标签等；结尾 > 可能被模型漏写，设为可选）
_DSML_TAG_PATTERN = re.compile(
    rf"</?{_DSML_BARS}DSML{_DSML_BARS}[^>]*>?",
    re.IGNORECASE,
)

# DSML 闭标签（含 </parameter>、</invoke>、</dsml> 等；结尾 > 可选）
_DSML_CLOSE_PATTERN = re.compile(
    rf"</{_DSML_BARS}DSML{_DSML_BARS}[^>]*>?",
    re.IGNORECASE,
)

# name 属性定位
_NAME_ATTR_PATTERN = re.compile(r"name\s*=\s*(['\"])", re.IGNORECASE)


def _extract_note_content(raw: str):
    """从 note:{json} 参数中提取 content 正文。

    容忍两种模型常见残缺输出：
    - 末尾多余符号（如 ]，或属性闭合引号丢失后截断到 JSON 内部引号）；
    - content 为最后一个字段时被截断（缺失结尾的引号与大括号）。
    """
    candidate = raw.strip()
    if candidate.endswith("]"):
        candidate = candidate[:-1].rstrip()
    candidates = [candidate]
    if not candidate.endswith('"'):
        candidates.append(candidate + '"')
        candidates.append(candidate + '"}')
    else:
        candidates.append(candidate + '}')
    # 兜底：若属性值内出现真实换行/制表符（非法 JSON 控制字符），转义后重试
    sanitized = (
        candidate.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    )
    if sanitized != candidate:
        candidates.extend(
            [sanitized, sanitized + '"', sanitized + '"}']
            if not sanitized.endswith('"')
            else [sanitized, sanitized + '}']
        )
    for item in candidates:
        try:
            payload = json.loads(item)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _extract_invoke_note_content(tag: str):
    """解析单个 DSML invoke 开标签；若 name 指向 note:{json}，返回其 content 正文。"""
    match_name = _NAME_ATTR_PATTERN.search(tag)
    if not match_name:
        return None
    quote = match_name.group(1)
    value_start = match_name.end()
    # 属性值的闭合引号必是该标签内的最后一个引号
    value_end = tag.rfind(quote)
    if value_end <= value_start:
        return None
    raw = tag[value_start:value_end]
    if not raw.startswith("note:"):
        return None
    return _extract_note_content(raw[len("note:"):])


def _strip_dsml_invoke_spans(text: str) -> str:
    """移除（非 note 的）DSML invoke 块：从 invoke 开标签到其后最近的 DSML 闭标签。"""
    while True:
        match_open = _DSML_INVOKE_OPEN_PATTERN.search(text)
        if not match_open:
            return text
        match_close = _DSML_CLOSE_PATTERN.search(text, match_open.end())
        if not match_close:
            # 无闭标签时，仅删除开标签起的所有残留
            return text[: match_open.start()] + text[match_open.end():]
        text = text[: match_open.start()] + text[match_close.end():]


def strip_tool_calls(text: str) -> str:
    """移除工具调用标记；优先提取 DSML invoke 中 note:{json} 的 content 正文。"""

    if not text:
        return text

    changed = False
    extracted = []

    # 1) 清理原有 [TOOL_CALL:...] 协议标记
    cleaned = _TOOL_CALL_PATTERN.sub("", text)
    if cleaned != text:
        changed = True
    text = cleaned

    # 2) DSML invoke：note:{json} 的 content 正文优先提取，替换对应开标签
    if _DSML_INVOKE_OPEN_PATTERN.search(text):
        parts = []
        position = 0
        for match in _DSML_INVOKE_OPEN_PATTERN.finditer(text):
            parts.append(text[position:match.start()])
            content = _extract_invoke_note_content(match.group(0))
            if content is not None:
                changed = True
                extracted.append(content)
                parts.append(content)
            else:
                # 非 note 的 invoke 原样保留，交由整块移除逻辑处理
                parts.append(match.group(0))
            position = match.end()
        parts.append(text[position:])
        text = "".join(parts)

    # 3) 移除其余（非 note 的）DSML invoke 整块结构
    cleaned = _strip_dsml_invoke_spans(text)
    if cleaned != text:
        changed = True
    text = cleaned

    # 4) 清除所有残留 DSML 标签（calls / parameter / 闭合标签等）
    cleaned = _DSML_TAG_PATTERN.sub("", text)
    if cleaned != text:
        changed = True
    text = cleaned

    if not changed:
        return text

    text = text.strip()
    if not text and extracted:
        return "\n\n".join(part.strip() for part in extracted if part.strip())

    return text
