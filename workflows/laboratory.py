"""Coordinate agents through the laboratory research workflow."""

import copy
import json
from pathlib import Path

from agents import (
    MLEngineerAgent,
    MLESolverAgent,
    PaperSolverAgent,
    PhDStudentAgent,
    PostdocAgent,
    ProfessorAgent,
    SWEngineerAgent,
)
from run_logging import current_run_logger, phase_log

from .collaboration import (
    _compact_collaboration_result,
    _execution_failure_context,
    _render_collaboration_history,
)
from .supplements import (
    _find_duplicate_supplement,
    _supplement_request_fingerprint,
)


class LaboratoryWorkflow:
    def __init__(self, state, model, api_key, workdir, collaboration_rounds, compile_latex=False, *, code_execution_timeout_seconds, paper_writing):
        self.state = state
        self.model = model
        self.api_key = api_key
        self.workdir = Path(workdir).resolve()
        self.compile_latex = bool(compile_latex)
        self.collaboration_rounds = collaboration_rounds
        self.code_execution_timeout_seconds = code_execution_timeout_seconds
        self.phd = PhDStudentAgent(model, api_key)
        self.postdoc = PostdocAgent(model, api_key)
        self.ml_engineer = MLEngineerAgent(model, api_key)
        self.sw_engineer = SWEngineerAgent(model, api_key)
        self.ml_solver = MLESolverAgent(
            model, api_key, code_execution_timeout_seconds=code_execution_timeout_seconds,
        )
        self.paper_solver = PaperSolverAgent(model, api_key, paper_writing=paper_writing)
        self.professor = ProfessorAgent(model, api_key)

    # _save_text()：保存指定的文本文件。把生成的数据加载代码保存到当前运行目录下的 src/load_data.py
    def _save_text(self, relative_path, content):
        path = self.workdir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")


    # save_state()：把当前研究状态保存到 state.json。self.state 保存这次研究的主题、文献综述、实验计划、代码、执行结果等信息。
    def save_state(self):
        path = self.workdir / "state.json"
        path.write_text(
            json.dumps(self.state.to_dict(), ensure_ascii=True, indent=2, default=str),
            encoding="utf-8",
        )

    def _formulate_plan(self):
        feedback = None
        previous_plan = ""
        history = self.state.planning_history = []
        budget = self.collaboration_rounds["plan_max_rounds"]
        for round_number in range(1, budget + 1):
            postdoc_result = self.postdoc.plan_turn(
                self.state.topic, self.state.literature_review, feedback,
                approved=False, previous_plan=previous_plan,
                planning_history=copy.deepcopy(history), seeds=self.state.seeds,
                round_number=round_number, remaining_reviews=budget - round_number + 1,
            )
            previous_plan = postdoc_result.value["message"]
            history.append({"round": round_number, "plan": previous_plan})
            self.save_state()
            feedback = self.phd.plan_feedback(
                self.state.topic, self.state.literature_review, previous_plan,
                planning_history=copy.deepcopy(history), seeds=self.state.seeds,
            )
            history[-1]["feedback"] = feedback
            self.save_state()
            if feedback["approved"] is True:
                submission = self.postdoc.plan_turn(
                    self.state.topic, self.state.literature_review, feedback,
                    approved=True, previous_plan=previous_plan,
                    planning_history=copy.deepcopy(history), seeds=self.state.seeds,
                    round_number=round_number, remaining_reviews=budget - round_number,
                )
                return {
                    "plan": submission.value["plan"],
                    "data_requirements": submission.value["data_requirements"],
                }
        raise RuntimeError("Plan formulation exceeded its dialogue budget")

    def _interpret_results(self):
        latest_postdoc_feedback = None
        stage = "review"
        workflow_notice = ""
        history = self.state.interpretation_history
        round_index = 0

        while True:
            current_run_logger().info(f"[results interpretation][round={round_index}] CALL PhDStudentAgent")
            
            phd_result = self.phd.interpretation_turn(
                self.state.topic,
                self.state.plan,
                self.state.experiment_code,
                self.state.experiment_output,
                latest_postdoc_feedback,
                literature_review=self.state.literature_review,
                supplementary_experiments=self.state.supplementary_experiments,
                # The latest review is supplied separately, including its supplement decision.
                interpretation_history=(
                    history[:-1]
                    if latest_postdoc_feedback is not None and history[-1]["type"] == "postdoc_feedback"
                    else history[:]
                ),
                stage=stage,
                workflow_notice=workflow_notice,
            )

            workflow_notice = ""

            if phd_result.tool_name == "submit_interpretation":
                interpretation = phd_result.value["interpretation"]
                history.append({
                    'round': round_index,
                    'sender': 'PhDStudentAgent',
                    'type': 'final_interpretation',
                    'interpretation': interpretation,
                })
                self.save_state()
                return interpretation

            if phd_result.tool_name == "send_dialogue":
                phd_message = phd_result.value["message"]
                history.append({
                    'round': round_index,
                    'sender': 'PhDStudentAgent',
                    'type': 'interpretation_question',
                    'message': phd_message,
                })
                self.save_state()
                current_run_logger().info(f"[results interpretation][round={round_index}] HANDOFF PhDStudentAgent -> PostdocAgent")

                # 记录流程从 PhD 转交给 Postdoc。
                postdoc_feedback = self.postdoc.interpretation_feedback(
                    self.state.topic,
                    self.state.plan,
                    self.state.experiment_output,
                    phd_message,
                    experiment_code=self.state.experiment_code,
                    literature_review=self.state.literature_review,
                    supplementary_experiments=self.state.supplementary_experiments,
                    interpretation_history=history[:-1],
                )

                latest_postdoc_feedback = postdoc_feedback
                history.append({
                    'round': round_index,
                    'sender': 'PostdocAgent',
                    'type': 'postdoc_feedback',
                    'message': postdoc_feedback['message'],
                    'supplement_required': postdoc_feedback['supplement_required'],
                })
                self.save_state()

                stage = {
                    True: "supplement",
                    False: "submit",
                }[postdoc_feedback["supplement_required"]]
                round_index += 1
                continue

            if phd_result.tool_name == "request_supplementary_experiment":  # PhD 申请执行补充实验。
                supplement_request = phd_result.value["request"]
                request_fingerprint = _supplement_request_fingerprint(supplement_request)   # 计算request_fingerprint，用于判断相同补充实验是否已经做过
                duplicate = _find_duplicate_supplement(self.state.supplementary_experiments, request_fingerprint)
                if duplicate is not None:   # 如果确实找到了重复记录，就进入拒绝分支。
                    duplicate_id = duplicate.get("supplement_id", "existing supplement")
                    workflow_notice = (
                        f"Duplicate supplementary request rejected: {duplicate_id}. "
                        "The existing results are available; this request was not executed again."
                    )
                    history.append({
                        'round': round_index,
                        'sender': 'LaboratoryWorkflow',
                        'type': 'duplicate_supplement_rejected',
                        'supplement_id': duplicate_id,
                        'request_fingerprint': request_fingerprint,
                        'request': supplement_request,
                    })
                    self.save_state()
                    round_index += 1
                    continue
                history.append({
                    "round": round_index,
                    "sender": "PhDStudentAgent",
                    "type": "supplement_request",
                    "request_fingerprint": request_fingerprint,
                    "request": supplement_request,
                })

                self.save_state()
                supplement_index = len(self.state.supplementary_experiments) + 1
                supplement_id = f"supplement_{supplement_index:02d}"
                current_run_logger().info(f"[results interpretation][round={round_index}] HANDOFF PhDStudentAgent -> MLESolverAgent supplement={supplement_id}")

                with phase_log(f"supplementary experiment {supplement_id}"):
                    supplement_workdir = self.workdir / "supplements" / supplement_id
                    supplement_workdir.mkdir(parents=True, exist_ok=True)
                    experiment = self.ml_solver.solve(
                        self.state, str(supplement_workdir),
                        supplement_request=supplement_request,
                    )
                    supplement = {
                        **experiment,
                        "supplement_id": supplement_id,
                        "request": supplement_request,
                    }

                supplement["request_fingerprint"] = request_fingerprint
                self.state.supplementary_experiments.append(supplement)
                self._save_text(f"src/{supplement_id}.py", supplement["code"])
                self._save_text(f"src/{supplement_id}_output.log", supplement["output"])
                history.append({
                    'round': round_index,
                    'sender': 'MLESolverAgent',
                    'type': 'supplement_result',
                    'supplement_id': supplement['supplement_id'],
                    'request_fingerprint': request_fingerprint,
                    'request': supplement_request,
                })
                self.save_state()
                latest_postdoc_feedback = None
                stage = "review"
                round_index += 1
                continue

            raise RuntimeError(
                "PhDStudentAgent must send dialogue, request a supplementary "
                "experiment, or submit the interpretation"
            )

    def _prepare_dataset_collaboratively(self):
        logger = current_run_logger()
        conversation = []   # 保存多轮协作消息和工具结果的摘要历史，后面通过 append() 逐条追加。
        sw_feedback = None
        ml_feedback = None
        inspected_dataset = None    # 已检查的数据集信息，例如有哪些字段、字段类型、数据划分、实际资源等。保存的是检查结果，不是完整数据集。
        selected_dataset = None # 当前选中了哪个数据集，主要保存 dataset_id、revision、config_name、split 等身份信息。
        latest_execution_failure = None # 最近一次有效的代码执行失败信息，供 Agent 修复代码，例如错误类型、报错内容和修复提示。
        failed_trial_code_failures = {} # 记录哪些代码已经运行失败，避免 Agent 原封不动地重复执行同一段失败代码。

        for round_index in range(self.collaboration_rounds["data_preparation_max_rounds"]):
            logger.info(f"[data preparation collaboration][round={round_index}] CALL SWEngineerAgent")

            sw_result = self.sw_engineer.data_preparation_turn(
                self.state.topic, self.state.data_requirements,
                conversation_history=_render_collaboration_history(conversation, omit_latest_senders={"SWEngineerAgent", "MLEngineerAgent"}, latest_execution_failure=latest_execution_failure,),
                feedback=sw_feedback,
                ml_result=ml_feedback,
                inspected_dataset=inspected_dataset,
                latest_execution_failure=latest_execution_failure,
            )
            conversation.append(_compact_collaboration_result("SWEngineerAgent", sw_result.tool_name, sw_result.value, round_index,))

            logger.info(f"[data preparation collaboration][round={round_index}] HANDOFF SWEngineerAgent -> MLEngineerAgent")
            
            sw_message = sw_result.value["message"]
            ml_result = self.ml_engineer.data_preparation_turn(
                self.state.topic,
                self.state.data_requirements,
                sw_message,
                code_execution_timeout_seconds=self.code_execution_timeout_seconds,
                conversation_history=_render_collaboration_history(conversation, omit_latest_senders={"SWEngineerAgent", "MLEngineerAgent"}, latest_execution_failure=latest_execution_failure,),
                workdir=self.workdir,
                feedback=ml_feedback,
                inspected_dataset=inspected_dataset,
                selected_dataset=selected_dataset,
                latest_execution_failure=latest_execution_failure,
                failed_trial_code_failures=failed_trial_code_failures,
            )
            tool_name = ml_result.tool_name
            ml_value = ml_result.value
            if tool_name == "search_huggingface_datasets":
                if ml_value.get("success") is True and ml_value.get("selected_dataset"):
                    selected_dataset = ml_value["selected_dataset"]
            elif tool_name == "inspect_selected_dataset":
                if ml_value.get("success") is True:
                    inspected_dataset = ml_value
            elif tool_name == "write_and_run_dataset_code":
                if ml_value.get("success") is True:
                    ml_value["selection"] = copy.deepcopy(inspected_dataset)
                    logger.info("[data preparation collaboration] FRAMEWORK SUBMITTED SUCCESSFUL MLE TRIAL")
                    return copy.deepcopy(inspected_dataset), ml_value

                repeated_failure = ml_value.get("error_type") == "unchanged_failed_dataset_code"
                if not repeated_failure or latest_execution_failure is None:
                    latest_execution_failure = _execution_failure_context(tool_name, ml_value)
            ml_feedback = {"tool": tool_name, **ml_value}
            conversation.append( _compact_collaboration_result("MLEngineerAgent",tool_name,ml_value,round_index,)
            )
            logger.info(f"[data preparation collaboration][round={round_index}] HANDOFF MLEngineerAgent -> SWEngineerAgent")
        raise RuntimeError("Data preparation exceeded its collaboration budget")

    def perform_research(self):
        with phase_log("literature review"):
            literature = self.phd.conduct_literature_review(self.state.topic)
        self.state.literature_papers = literature["papers"]
        self.state.literature_review = literature["review"]
        self.save_state()

        with phase_log("plan formulation"):
            planning = self._formulate_plan()
        self.state.plan = planning["plan"]
        self.state.data_requirements = planning["data_requirements"]
        self.save_state()

        with phase_log("data preparation collaboration"):
            self.state.dataset_selection, dataset = (
                self._prepare_dataset_collaboratively()
            )
        self.state.dataset_code = dataset["code"]   # 保存数据准备代码及其执行输出
        self.state.dataset_output = dataset["output"]
        self.state.dataset_manifest = dataset.get("prepared_data_manifest") or {}   # 保存数据产物清单，告诉后续实验阶段哪些变量是特征、目标和分析数据
        self.state.dataset_validation_warnings = list(dataset.get("validation_warnings") or []) # 保存不阻止运行、但需要关注的数据验证警告。
        self._save_text("src/load_data.py", self.state.dataset_code)    # 把最终数据加载代码写入 src/load_data.py
        self.save_state()

        with phase_log("running experiments"):
            experiment = self.ml_solver.solve(self.state, self.workdir)
        self.state.experiment_code = experiment["code"]
        self.state.experiment_output = experiment["output"]
        self._save_text("src/run_experiments.py", self.state.experiment_code)
        self._save_text("src/experiment_output.log", self.state.experiment_output)
        self.save_state()

        # 解释实验结果
        with phase_log("results interpretation"):
            self.state.interpretation = self._interpret_results()
        self.save_state()

        with phase_log("report writing"):
            paper = self.paper_solver.write(self.state,self.workdir,compile_pdf=self.compile_latex,)    # 默认只生成 LaTeX 源码，不执行 PDF 编译
        self.state.paper = paper["latex"]
        self._save_text("report.tex", self.state.paper)

        with phase_log("readme writing"):
            self.state.readme = self.professor.generate_readme(self.state)
        self._save_text("readme.md", self.state.readme)
        self.save_state()

        return self.state
