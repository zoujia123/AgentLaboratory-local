"""Shared helpers for agent tool schemas, feedback, execution diagnostics, and papers."""

import hashlib
import json
import re

from execution_diagnostics import classify_runtime_data_failure, runtime_data_repair_hint
from run_logging import current_prompt_recorder, current_run_logger


def _object_schema(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _string(description):
    return {"type": "string", "description": description}


def _code_hash(code):
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()


# 为了避免把同一份 inspection 结果在 prompt 里发送两遍。
def _inspection_feedback_for_prompt(feedback, inspected_dataset):
    """Reference an identical inspection already included separately in this prompt."""
    if (
        not isinstance(feedback, dict)
        or feedback.get("tool") != "inspect_selected_dataset"
        or feedback.get("success") is not True
        or not isinstance(inspected_dataset, dict)
        or inspected_dataset.get("success", True) is not True
        or not inspected_dataset.get("dataset_schema")
    ):
        return feedback
    identity_fields = ("dataset_id", "revision", "config_name", "split")
    if any(not inspected_dataset.get(name) for name in identity_fields):
        return feedback
    feedback_details = {key: value for key, value in feedback.items() if key not in {"tool", "success"}}
    inspected_details = {key: value for key, value in inspected_dataset.items() if key not in {"tool", "success"}}
    if feedback_details != inspected_details:
        return feedback
    return {
        "tool": "inspect_selected_dataset",
        "success": True,
        "message": "Inspection succeeded; see Authoritative inspected dataset below for the complete result.",
    }


def _authoritative_execution_failure(value):
    if not isinstance(value, dict) or value.get("success") is not False:
        return None
    return dict(value)


def _stale_diagnostic_marker():
    return {
        "message": (
            "A newer execution failure is supplied below as the authoritative "
            "diagnostic. Do not rely on older error analysis or repair advice."
        )
    }


def _preparation_feedback_for_prompt(feedback, inspected_dataset, latest_execution_failure):
    """Keep the last rejection visible without repeating the retained failed program."""
    if not _authoritative_execution_failure(latest_execution_failure):
        return _inspection_feedback_for_prompt(feedback, inspected_dataset)
    if (
        isinstance(feedback, dict)
        and feedback.get("success") is False
        and feedback.get("error_type") == "unchanged_failed_dataset_code"
    ):
        return {
            "tool": "write_and_run_dataset_code",
            "success": False,
            "error_type": "unchanged_failed_dataset_code",
            "code_hash": feedback.get("code_hash"),
            "error": "The last submission repeated a failed program and was not executed again.",
            "repair_hint": (
                "Use the authoritative execution failure below to correct the failing "
                "operation, then call write_and_run_dataset_code with the complete corrected "
                "program. Do not resubmit the rejected code hash."
            ),
        }
    return _stale_diagnostic_marker()


# 根据调用栈结构，省略部分第三方库栈帧
# 它能缩短 traceback，是因为它先识别出“哪些文字属于第三方库调用”，再把连续的多层库调用替换成一句省略说明。
def _compact_solver_traceback(traceback_text):
    """Keep user frames and exception chains; shorten recognized library frame runs."""
    if len(traceback_text) <= 2500:
        return traceback_text
    lines = traceback_text.splitlines(keepends=True)
    frame = re.compile(r'^\s*File "([^"]+)", line \d+(?:, in .*)?\s*$')
    chunks = []
    index = 0
    while index < len(lines):
        match = frame.match(lines[index].rstrip("\r\n"))
        if match is None:
            chunks.append((False, lines[index]))
            index += 1
            continue
        start = index
        index += 1
        while index < len(lines):
            line = lines[index]
            if frame.match(line.rstrip("\r\n")) or not line.strip() or not line[0].isspace():
                break
            index += 1
        path = match.group(1).replace("\\", "/").lower()
        library = "/site-packages/" in path or "/dist-packages/" in path
        chunks.append((library, "".join(lines[start:index])))
    rendered = []
    index = 0
    omitted = 0
    while index < len(chunks):
        if not chunks[index][0]:
            rendered.append(chunks[index][1])
            index += 1
            continue
        start = index
        while index < len(chunks) and chunks[index][0]:
            index += 1
        # Retain the innermost library frame of each contiguous run.
        count = index - start - 1
        if count:
            omitted += count
            rendered.append(f"  ... {count} library frames omitted ...\n")
        rendered.append(chunks[index - 1][1])
    if not omitted:
        return traceback_text
    compact = "".join(rendered) + "\n[Full execution traceback recorded in workflow.log and prompts.jsonl.]"
    return compact if len(compact) < len(traceback_text) else traceback_text


# 错误信息太长时，只保留开头和结尾，把中间省略掉。 它只是裁剪文字，不会判断错误原因，也不会修复代码。
def _bounded_solver_diagnostic(value, limit=4000):
    text = str(value or "")
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.75))
    tail = max(1, limit - head)
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def _latest_experiment_failure_messages(
    system_prompt,
    base_user_prompt,
    current_code,
    diagnostic,
    action_rejection=None,
):
    rejection_text = ""
    if action_rejection:
        rejection = {
            "error_type": action_rejection.get("error_type"),
            "error": action_rejection.get("error"),
            "repair_hint": action_rejection.get("repair_hint"),
            "original_error_type": action_rejection.get("original_error_type"),
            "original_error": action_rejection.get("original_error"),
            "original_failure_category": action_rejection.get(
                "original_failure_category"
            ),
            "original_repair_hint": action_rejection.get("original_repair_hint"),
            "code_hash": action_rejection.get("code_hash"),
        }
        if action_rejection.get("original_execution_diagnostic"):
            rejection["original_execution_diagnostic"] = action_rejection[
                "original_execution_diagnostic"
            ]
        rejection_text = (
            "\n\nLatest code action rejection:\n"
            + json.dumps(rejection, ensure_ascii=True, default=str)
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_user_prompt},
        {
            "role": "user",
            "content": (
                "Authoritative experiment recovery state:\n"
                "Earlier execution failures are superseded by this diagnostic. "
                "Do not rely on earlier errors.\n\n"
                "Current experiment code:\n```python\n"
                + str(current_code)
                + "\n```\n\nLatest execution diagnostic:\n"
                + json.dumps(diagnostic, ensure_ascii=True, default=str)
                + rejection_text
            ),
        },
    ]


def _coalesce_experiment_edit_batches(argument_batches):
    """Safely flatten repeated atomic edit calls into one edit request."""
    edits = []
    seen_old_text = set()
    for arguments in argument_batches:
        if not isinstance(arguments, dict) or set(arguments) != {"edits"}:
            return None
        batch = arguments.get("edits")
        if not isinstance(batch, list) or not batch:
            return None
        for edit in batch:
            if not isinstance(edit, dict) or set(edit) != {"old_text", "new_text"}:
                return None
            old_text = edit.get("old_text")
            new_text = edit.get("new_text")
            if (
                not isinstance(old_text, str)
                or not old_text
                or not isinstance(new_text, str)
                or old_text in seen_old_text
            ):
                return None
            seen_old_text.add(old_text)
            edits.append(edit)
    if not 1 <= len(edits) <= 6:
        return None
    return {"edits": edits}


def _process_execution_result(current, fingerprint, execution, phase):
    """
    Update execution state once and return the model-facing feedback.
    execution	执行器返回的原始结果，是这个函数的输入
    current	    实验过程的状态记录，会被这个函数修改
    feedback	准备返回给模型的工具结果
    diagnostic	失败时生成的精简诊断，保存在 current 中
    """
    success = execution["success"]  # true or false
    current.update( output=execution["output"], success=success, code_hash=fingerprint,)    # 把本次执行的输出、成败和代码版本记下来。
    feedback = {
        "success": success,
        "error_type": execution["error_type"],
        "error": execution["error"],
        "traceback": execution["traceback"],
        "exit_code": execution["exit_code"],
        "output": execution["output"],
        "code_hash": fingerprint,
        "failure_category": None,
        "repair_hint": None,
        "same_category_count": 0,
        "full_replacement_required": False,
    }   # 先建立一份基础反馈

    if success:
        current.update(last_failure_category=None,consecutive_failure_category_count=0,force_replace=False,)   # 如果执行成功，就清理当前失败状态
        if "latest_execution_failure" in current:
            current["latest_execution_failure"] = None  # 主实验使用这个字段保存最新错误；补充实验目前没有使用它。如果有这个字段，就清空它，表示当前执行已经成功。
        return feedback

    # Classify the original diagnostic before shortening it for the model.
    category = classify_runtime_data_failure(
        execution["error"], execution["traceback"],
        error_type=execution["error_type"],
    ) or "code_execution_error"
    repair_hint = runtime_data_repair_hint(category)    # 根据类别取修复建议
    # 执行环境故障不代表代码本身失败，仍允许原样重试。
    infrastructure_error = category == "execution_infrastructure_error"
    # 计算连续同类失败次数
    if category == current.get("last_failure_category"):
        count = current.get("consecutive_failure_category_count", 0) + 1
    else:
        count = 1
    # 判断是否要求完整重写
    force_replace = count >= 2 and category not in {
        "code_execution_error", "execution_infrastructure_error",
    }
    current.update( last_failure_category=category,consecutive_failure_category_count=count,force_replace=force_replace,)
    feedback.update(failure_category=category,repair_hint=repair_hint,same_category_count=count,full_replacement_required=force_replace,)

    original_traceback = execution["traceback"] or ""   # 取出原始 traceback。
    compact_traceback = _compact_solver_traceback(original_traceback)   # 压缩 traceback，便于日志记录。
    # 生成供缓存使用的诊断摘要
    diagnostic = {
        "success": False,
        "error_type": execution["error_type"],
        "error": _bounded_solver_diagnostic(execution["error"]),        # 按调用栈结构折叠部分库栈帧。
        "traceback": _bounded_solver_diagnostic(compact_traceback, limit=6000), # 再限制缓存文字的长度。
        "failure_category": category,
        "repair_hint": repair_hint,
        "same_category_count": count,
        "full_replacement_required": force_replace,
        "code_hash": fingerprint,
    }
    # Historical failures and the latest failure reuse this read-only snapshot.
    if not infrastructure_error:
        current["failed_hashes"].add(fingerprint)
        current.setdefault("failed_failures", {})[fingerprint] = diagnostic
    if "latest_execution_failure" in current:
        current["latest_execution_failure"] = diagnostic

    if compact_traceback != original_traceback:
        current_run_logger().debug(f"[MLESolverAgent][{phase}] FULL EXECUTION DIAGNOSTIC\n"+ json.dumps(feedback, ensure_ascii=False, indent=2, default=str))
        current_prompt_recorder().record({"event": "solver_execution_diagnostic","agent": "MLESolverAgent","phase": phase,"code_hash": fingerprint,"result": feedback,})
        return {**feedback, "traceback": compact_traceback}

    return feedback


def _latex_escape_text(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _citation_catalog(state):
    catalog = []
    for index, paper in enumerate(
        getattr(state, "literature_papers", []) or [],
        start=1,
    ):
        arxiv_id = str((paper or {}).get("arxiv_id") or "").strip()
        if arxiv_id:
            catalog.append({"key": f"paper_{index}", "arxiv_id": arxiv_id})
    return catalog
