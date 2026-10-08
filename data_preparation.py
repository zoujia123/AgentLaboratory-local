"""Check data-preparation code scope and suggest local repairs."""

import ast
import copy

from execution_diagnostics import (
    classify_runtime_data_failure,
    python_syntax_error,
    runtime_data_repair_hint,
)
from tools import execute_code

_DATA_PREPARATION_FORBIDDEN_MODULE_PREFIXES = (
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.tree",
    "sklearn.svm",
    "sklearn.neighbors",
    "sklearn.neural_network",
    "sklearn.cluster",
    "sklearn.naive_bayes",
    "sklearn.discriminant_analysis",
    "sklearn.gaussian_process",
    "sklearn.metrics",
    "xgboost",
    "lightgbm",
    "catboost",
    "tensorflow",
    "keras",
    "torch.nn",
    "torch.optim",
)

_DATA_PREPARATION_FORBIDDEN_METHODS = {
    "predict",
    "predict_proba",
    "predict_log_proba",
    "fit_predict",
    "score",
    "evaluate",
    "backward",
}

_DATA_PREPARATION_FORBIDDEN_CALLS = {
    "GridSearchCV",
    "RandomizedSearchCV",
    "cross_val_score",
    "cross_validate",
    "learning_curve",
    "validation_curve",
}

_DATA_PREPARATION_FORBIDDEN_METRIC_CALLS = {
    "accuracy_score",
    "balanced_accuracy_score",
    "classification_report",
    "confusion_matrix",
    "f1_score",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
}

_DATA_PREPARATION_FORBIDDEN_NAMESPACE_IMPORTS = {
    "sklearn": {
        "cluster",
        "discriminant_analysis",
        "ensemble",
        "gaussian_process",
        "linear_model",
        "metrics",
        "naive_bayes",
        "neighbors",
        "neural_network",
        "svm",
        "tree",
    },
    "torch": {"nn", "optim"},
}


def _dotted_ast_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_data_preparation_forbidden_module(module_name):
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in _DATA_PREPARATION_FORBIDDEN_MODULE_PREFIXES
    )


def _data_preparation_scope_violations(code):
    try:
        tree = ast.parse(code)  # 把 Python 源码转换成 AST，即抽象语法树。
    except SyntaxError:
        # The execution layer already returns a precise syntax diagnostic.
        return []

    violations = set()  # 创建集合，保存发现的违规项。
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):    # 处理普通导入：
            for alias in node.names:
                if _is_data_preparation_forbidden_module(alias.name):
                    violations.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):    # 处理 from ... import ... 形式的导入
            module_name = node.module or ""
            if _is_data_preparation_forbidden_module(module_name):
                violations.add(f"import from {module_name}")
            elif module_name in _DATA_PREPARATION_FORBIDDEN_NAMESPACE_IMPORTS:
                for alias in node.names:
                    if alias.name in _DATA_PREPARATION_FORBIDDEN_NAMESPACE_IMPORTS[module_name]:
                        violations.add(f"import {module_name}.{alias.name}")
        elif isinstance(node, ast.Call):    # 处理函数与方法调用：
            dotted_name = _dotted_ast_name(node.func)
            call_name = dotted_name.rsplit(".", 1)[-1]
            if _is_data_preparation_forbidden_module(dotted_name):
                violations.add(f"call {dotted_name}()")
            if call_name in _DATA_PREPARATION_FORBIDDEN_METHODS:
                violations.add(f"call .{call_name}()")
            if call_name in _DATA_PREPARATION_FORBIDDEN_CALLS:
                violations.add(f"call {call_name}()")
            if call_name in _DATA_PREPARATION_FORBIDDEN_METRIC_CALLS:
                violations.add(f"call {call_name}()")
            if call_name.endswith(("Classifier", "Regressor")):
                violations.add(f"construct {call_name}()")
    return sorted(violations)


def _run_preparation_code(code, workdir, inspected_dataset, code_execution_timeout_seconds):
    # 检查代码是否为空，如果为空直接返回失败
    if not code.strip():
        return {"success": False, "code": code, "output": "", "error": "Provide Python code"}
    
    # 先检查检查代码有没有超出“数据准备”的职责
    violations = _data_preparation_scope_violations(code)
    if violations:
        return {
            "success": False,
            "error_type": "data_preparation_scope_violation",
            "error": ("Data preparation code may only prepare reusable data. Move model training, prediction, and final evaluation to MLESolver: "+ ", ".join(violations)),
            "traceback": "",    # 没有运行代码，所以没有执行异常的调用栈
            "exit_code": None,  # 没有启动执行子进程，因此没有退出码
            "code": code,   # 保留被拒绝的代码，便于修改
            "output": "",   # 没有程序输出
            "selection": {},        # 没有成功完成准备，不返回最终数据集选择
            "artifacts": [],    # 没有成功执行后收集到的产物信息
            "prepared_data_manifest": None, # 没有准备结果清单
        }
  
    # 调用执行器运行代码
    details = dict(execute_code(
        code, timeout=code_execution_timeout_seconds, workdir=workdir, return_details=True,
        expected_hf_dataset=inspected_dataset,
    ))
    if details["success"] and not details.get("executed_dataset"):
        details.update(
            success=False,
            error_type="dataset_binding_error",
            error="Preparation did not provide evidence of loading the selected dataset",
        )

    output = details["output"]  # 取出程序输出，暂存到 output。
    result = {
        "success": details["success"],
        "error_type": details["error_type"],
        "error": details["error"],
        "traceback": details["traceback"],
        "exit_code": details["exit_code"],
        "code": code,
        "output": output,
        "selection": copy.deepcopy(inspected_dataset) if details["success"] else {},
        "executed_dataset": details.get("executed_dataset"),
        "artifacts": list(details.get("artifacts", [])),
        "prepared_data_manifest": details.get("prepared_data_manifest"),
    }
    # 最后整理结果，并在失败时补充修复建议。
    if result["success"] is False:
        if details.get("failure_context"):
            result["failure_context"] = copy.deepcopy(details["failure_context"])
        category = classify_runtime_data_failure(       # 共用执行状态和错误文字的分类规则
            result["error"], result["traceback"],
            error_type=result["error_type"],
        )
        if category is not None:   
            result["failure_category"] = category
            repair_hint = runtime_data_repair_hint(category)    # 根据类别查出预先写好的修复建议
        else:   
            repair_hint = python_syntax_error(code)    # 检查代码是否存在语法错误。

        if repair_hint:
            result["repair_hint"] = repair_hint
    return result

