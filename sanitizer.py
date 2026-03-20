"""Sanitize LLM responses — normalize tool calls for downstream consumers."""

import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)

# Pattern 1: <tool_call>{"name": "func", "arguments": {...}}</tool_call> — JSON in tags
_JSON_TOOL_CALL = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)

# Pattern 2: <tool_code>func(arg1="val1", arg2="val2")</tool_code> — function-call syntax
_FUNC_TOOL_CALL = re.compile(
    r"<tool_code>\s*(\w+)\((.*?)\)\s*</tool_code>", re.DOTALL
)

# Pattern 3: <tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>
_CHATML_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_CHATML_PARAM = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL
)

# Quick check for any XML tool call
_HAS_TOOL_XML = re.compile(r"<tool_call>|<tool_code>")


def sanitize_response(response: dict) -> dict:
    """Normalize an OpenAI-compatible chat completion response.

    Two passes:
    1. Convert XML tool calls in content/reasoning_content to proper tool_calls
    2. Strip whitespace padding from tool_call argument values
    """
    response = _convert_xml_tool_calls(response)
    response = _strip_tool_call_whitespace(response)
    return response


def _convert_xml_tool_calls(response: dict) -> dict:
    """Convert XML tool calls in content fields to proper tool_calls entries."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return response

    tool_calls: list[dict] = []

    for field in ("content", "reasoning_content"):
        text = message.get(field)
        if not text or not _HAS_TOOL_XML.search(text):
            continue

        parsed, cleaned = _extract_tool_calls(text)
        if parsed:
            tool_calls.extend(parsed)
            message[field] = cleaned.strip() or None
            logger.info("Sanitized %d XML tool call(s) from %s", len(parsed), field)

    if not tool_calls:
        return response

    existing = message.get("tool_calls") or []
    message["tool_calls"] = existing + tool_calls
    response["choices"][0]["finish_reason"] = "tool_calls"

    return response


def _strip_tool_call_whitespace(response: dict) -> dict:
    """Strip whitespace padding from tool_call argument values.

    Some models (qwen3.5) wrap argument values in newlines:
      {"key": "\n\nvalue\n\n"}
    This causes downstream validation failures.
    """
    try:
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return response
    except (KeyError, IndexError, TypeError):
        return response

    changed = False
    for tc in tool_calls:
        try:
            func = tc.get("function", {})
            args_str = func.get("arguments", "")
            if not args_str or not isinstance(args_str, str):
                continue
            args = json.loads(args_str)
            if not isinstance(args, dict):
                continue
            stripped = {}
            for k, v in args.items():
                k_clean = k.strip() if isinstance(k, str) else k
                if isinstance(v, str):
                    v_clean = v.strip()
                    if v_clean != v or k_clean != k:
                        changed = True
                    stripped[k_clean] = v_clean
                else:
                    if k_clean != k:
                        changed = True
                    stripped[k_clean] = v
            if stripped != args:
                func["arguments"] = json.dumps(stripped)
        except (json.JSONDecodeError, AttributeError):
            continue

    if changed:
        logger.info("Stripped whitespace padding from tool_call arguments")

    return response


def _extract_tool_calls(text: str) -> tuple[list[dict], str]:
    """Extract tool calls from text, return (parsed_calls, cleaned_text)."""
    calls: list[dict] = []
    cleaned = text

    # Pattern 1: JSON in <tool_call> tags
    for match in _JSON_TOOL_CALL.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            calls.append(_make_tool_call(name, arguments))
            cleaned = cleaned.replace(match.group(0), "")
        except (json.JSONDecodeError, AttributeError):
            logger.debug("Failed to parse JSON tool call: %s", match.group(1)[:100])

    # Pattern 3: ChatML parameter style (before pattern 2 since both use <tool_call>)
    for match in _CHATML_TOOL_CALL.finditer(text):
        full_match = match.group(0)
        if full_match not in cleaned:
            continue
        func_name = match.group(1)
        params_text = match.group(2)
        params = {}
        for param_match in _CHATML_PARAM.finditer(params_text):
            params[param_match.group(1)] = param_match.group(2)
        calls.append(_make_tool_call(func_name, json.dumps(params)))
        cleaned = cleaned.replace(full_match, "")

    # Pattern 2: function-call syntax in <tool_code> tags
    for match in _FUNC_TOOL_CALL.finditer(text):
        func_name = match.group(1)
        args_str = match.group(2).strip()
        params = _parse_func_args(args_str)
        calls.append(_make_tool_call(func_name, json.dumps(params)))
        cleaned = cleaned.replace(match.group(0), "")

    return calls, cleaned


def _make_tool_call(name: str, arguments: str) -> dict:
    """Build an OpenAI-format tool_calls entry."""
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _parse_func_args(args_str: str) -> dict:
    """Parse function-call style arguments: key="value", key2=123."""
    if not args_str:
        return {}
    params = {}
    for m in re.finditer(r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|(\S+))', args_str):
        key = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
        params[key] = value
    return params
