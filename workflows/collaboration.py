"""History summaries and execution diagnostics for agent collaboration."""

import hashlib
import json


def _bounded_collaboration_text(
    value,
    head=1000,
    tail=2000,
):
    text = str(value or "")
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    suffix = text[-tail:] if tail else ""
    return text[:head] + f"\n...[omitted {omitted} chars]...\n" + suffix


def _collaboration_code_summary(code):
    if not isinstance(code, str) or not code:
        return None
    preview_chars = 500
    if len(code) <= preview_chars:
        preview = code
    else:
        head = preview_chars // 2
        tail = preview_chars - head
        preview = (
            code[:head]
            + f"\n...[omitted {len(code) - preview_chars} code chars]...\n"
            + code[-tail:]
        )
    return {
        "code_chars": len(code),
        "code_lines": len(code.splitlines()),
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "code_preview": preview,
    }


def _compact_dataset_results(results):
    compacted = []
    for item in list(results or []):
        if not isinstance(item, dict):
            compacted.append(_bounded_collaboration_text(item, head=500, tail=0))
            continue
        compact = {}
        for key in (
            "id",
            "configs",
            "splits",
            "revision",
            "gated",
            "private",
            "train_element_size",
            "test_element_size",
        ):
            if item.get(key) is not None:
                compact[key] = item[key]
        if item.get("description"):
            compact["description"] = _bounded_collaboration_text(
                item["description"],
                head=500,
                tail=0,
            )
        compacted.append(compact)
    return compacted


# 压缩后的协作记录结果
def _compact_collaboration_result(sender, tool_name, value, round_index):
    value = value if isinstance(value, dict) else {"value": value}
    compact = {
        "round": round_index,
        "sender": sender,
        "tool": tool_name,
        "success": value.get("success"),
    }
    for key in (
        "error_type",
        "exit_code",
        "selection",
        "selected_dataset",
        "prepared_data_manifest",
        "validation_warnings",
        "next_action_hint",
    ):
        if value.get(key) is not None:
            compact[key] = value[key]

    diagnostic_values = set()
    if value.get("error"):
        error_text = str(value["error"])
        compact["error"] = _bounded_collaboration_text(error_text)
        diagnostic_values.add(error_text)

    code_summary = _collaboration_code_summary(value.get("code"))
    if code_summary is not None:
        compact.update(code_summary)

    for key in ("traceback", "output"):
        if value.get(key):
            diagnostic = str(value[key])
            if diagnostic not in diagnostic_values:
                compact[key] = _bounded_collaboration_text(diagnostic)
                diagnostic_values.add(diagnostic)

    if tool_name == "send_dialogue" and value.get("message") is not None:
        compact["message"] = _bounded_collaboration_text(
            value["message"],
            head=1500,
            tail=500,
        )
    if tool_name == "search_huggingface_datasets":
        compact["results"] = _compact_dataset_results(value.get("results"))
    return compact


_EXECUTION_TOOL_NAMES = frozenset({"write_and_run_dataset_code"})


def _execution_failure_context(tool_name, value):
    if tool_name not in _EXECUTION_TOOL_NAMES:
        return None
    if not isinstance(value, dict) or value.get("success") is not False:
        return None
    return {"tool": tool_name, **value}


def _render_collaboration_history(conversation, omit_latest_senders=(), limit=6, latest_execution_failure=None,):
    omitted_indexes = set()
    for sender in omit_latest_senders:  # 对每个指定的发送者，从列表末尾向前找。
        for index in range(len(conversation) - 1, -1, -1):
            if (
                index not in omitted_indexes
                and conversation[index].get("sender") == sender
            ):
                omitted_indexes.add(index)
                break
    visible = [
        item
        for index, item in enumerate(conversation)
        if index not in omitted_indexes
    ]
    if latest_execution_failure:
        # The latest failure is supplied separately as the authoritative diagnostic.
        # Do not let older failure traces or repair advice compete for model attention.
        latest_failure_index = max(
            (
                index
                for index, item in enumerate(visible)
                if item.get("tool") in _EXECUTION_TOOL_NAMES
                and item.get("success") is False
            ),
            default=None,
        )
        if latest_failure_index is not None:
            visible = visible[latest_failure_index + 1:]
    limit = max(0, int(limit))
    visible = visible[-limit:] if limit else []
    return "\n".join(
        json.dumps(item, ensure_ascii=True, sort_keys=True, default=str)
        for item in visible
    )
