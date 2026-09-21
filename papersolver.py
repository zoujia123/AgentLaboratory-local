import json
import re
from pathlib import Path

from agent_runtime import AgentRuntime, ToolRegistry
from prompts import (
    paper_blueprint_prompt,
    paper_reward_prompt,
    paper_section_prompt,
    reviewer_prompt,
)
from utils import compile_latex


def _schema(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _string(description):
    return {"type": "string", "description": description}


PAPER_SECTIONS = (
    "Abstract",
    "Introduction",
    "Background",
    "Related Work",
    "Methods",
    "Experimental Setup",
    "Results",
    "Discussion",
)
PAPER_GENERATION_ORDER = PAPER_SECTIONS[1:] + (PAPER_SECTIONS[0],)
SECTION_WORD_TARGETS = {
    "Abstract": 250,
    "Introduction": 600,
    "Background": 450,
    "Related Work": 500,
    "Methods": 700,
    "Experimental Setup": 650,
    "Results": 600,
    "Discussion": 650,
}
SECTION_MAX_CHARS = {
    "Abstract": 2500,
    "Introduction": 6500,
    "Background": 5500,
    "Related Work": 5500,
    "Methods": 7000,
    "Experimental Setup": 6500,
    "Results": 6500,
    "Discussion": 6500,
}


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


class PaperSolverAgent:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key

    def write(self, state, workdir, compile_pdf=False):
        workdir = Path(workdir).resolve()
        available_figures = [
            name
            for name in ("Figure_1.png", "Figure_2.png")
            if Path(workdir, name).is_file()
        ]
        citations = _citation_catalog(state)
        supplements = getattr(state, "supplementary_experiments", []) or []
        authoritative_evidence = {
            "topic": state.topic,
            "literature_review": state.literature_review,
            "plan": state.plan,
            "dataset_selection": state.dataset_selection,
            "fixed_numerical_seeds": getattr(state, "seeds", {}),
            "experiment_output": state.experiment_output,
            "supplementary_experiments": [
                {key: value for key, value in item.items() if key != "code"}
                for item in supplements
            ],
            "experiment_reward": state.experiment_score,
            "interpretation": state.interpretation,
            "citation_catalog": citations,
            "available_figures": available_figures,
        }

        def submit_paper_blueprint(title, central_claim, section_plans):
            if set(section_plans) != set(PAPER_SECTIONS):
                return {
                    "success": False,
                    "error": "The blueprint must plan every fixed paper section exactly once",
                }
            return {
                "success": True,
                "title": title.strip(),
                "central_claim": central_claim.strip(),
                "section_plans": {
                    section: str(section_plans[section]).strip()
                    for section in PAPER_SECTIONS
                },
            }

        blueprint_registry = ToolRegistry().register(
            "submit_paper_blueprint",
            "Submit the short evidence-grounded blueprint for all fixed paper sections.",
            _schema(
                {
                    "title": {
                        "type": "string",
                        "description": "Restrained research paper title",
                        "maxLength": 180,
                    },
                    "central_claim": {
                        "type": "string",
                        "description": "One claim supported by the observed evidence",
                        "maxLength": 600,
                    },
                    "section_plans": {
                        "type": "object",
                        "properties": {
                            section: {
                                "type": "string",
                                "description": f"Concise writing plan for {section}",
                                "maxLength": 500,
                            }
                            for section in PAPER_SECTIONS
                        },
                        "required": list(PAPER_SECTIONS),
                        "additionalProperties": False,
                    },
                },
                ["title", "central_claim", "section_plans"],
            ),
            submit_paper_blueprint,
        )
        blueprint = AgentRuntime(
            model=self.model,
            api_key=self.api_key,
            agent_name="PaperSolverAgent",
            phase="report writing blueprint",
            max_tool_steps=2,
            max_tokens=3000,
        ).run_one(
            system_prompt=paper_blueprint_prompt(PAPER_SECTIONS, available_figures),
            user_prompt=(
                "Authoritative evidence:\n"
                + json.dumps(
                    {
                        **authoritative_evidence,
                        "experiment_code": state.experiment_code,
                    },
                    ensure_ascii=True,
                    default=str,
                )
            ),
            registry=blueprint_registry,
        ).value

        tex_dir = workdir / "tex"
        sections_dir = tex_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        (tex_dir / "paper_blueprint.json").write_text(
            json.dumps(blueprint, ensure_ascii=True, indent=2, default=str),
            encoding="utf-8",
        )

        section_bodies = {}
        allowed_citation_keys = {item["key"] for item in citations}
        forbidden_section_fragments = (
            "\\documentclass",
            "\\begin{document}",
            "\\end{document}",
            "\\title{",
            "\\maketitle",
            "\\begin{abstract}",
            "\\begin{thebibliography}",
            "\\begin{figure}",
            "\\includegraphics",
        )

        for section in PAPER_GENERATION_ORDER:
            def submit_paper_section(section_name, latex_body, expected=section):
                if section_name != expected:
                    return {
                        "success": False,
                        "error": f"Submit only the requested section {expected!r}",
                    }
                body = str(latex_body).strip()
                if not body:
                    return {"success": False, "error": "Section body cannot be empty"}
                forbidden = [
                    fragment
                    for fragment in forbidden_section_fragments
                    if fragment in body
                ]
                if re.search(r"\\section\*?\s*\{", body):
                    forbidden.append("main section command")
                if forbidden:
                    return {
                        "success": False,
                        "error": (
                            "The framework owns document structure and figures; remove: "
                            + ", ".join(forbidden)
                        ),
                    }
                used_citations = set()
                for group in re.findall(r"\\cite(?:t|p)?\{([^}]*)\}", body):
                    used_citations.update(
                        key.strip()
                        for key in group.split(",")
                        if key.strip()
                    )
                unknown_citations = sorted(used_citations - allowed_citation_keys)
                if unknown_citations:
                    return {
                        "success": False,
                        "error": "Unknown citation keys: " + ", ".join(unknown_citations),
                    }
                return {
                    "success": True,
                    "section_name": expected,
                    "latex_body": body,
                }

            section_registry = ToolRegistry().register(
                "submit_paper_section",
                f"Submit only the LaTeX body for the fixed {section} section.",
                _schema(
                    {
                        "section_name": {
                            "type": "string",
                            "enum": [section],
                            "description": "The current fixed section name",
                        },
                        "latex_body": {
                            "type": "string",
                            "description": f"LaTeX body for {section}, without its heading",
                            "maxLength": SECTION_MAX_CHARS[section],
                        },
                    },
                    ["section_name", "latex_body"],
                ),
                submit_paper_section,
            )
            section_evidence = dict(authoritative_evidence)
            if section in {"Methods", "Experimental Setup"}:
                section_evidence["experiment_code"] = state.experiment_code
                section_evidence["supplementary_experiments"] = [
                    dict(item) for item in supplements
                ]
            section_result = AgentRuntime(
                model=self.model,
                api_key=self.api_key,
                agent_name="PaperSolverAgent",
                phase=f"report writing section {section}",
                max_tool_steps=2,
                max_tokens=4096,
            ).run_one(
                system_prompt=paper_section_prompt(
                    section,
                    SECTION_WORD_TARGETS[section],
                    available_figures,
                ),
                user_prompt=(
                    "Approved paper blueprint:\n"
                    + json.dumps(blueprint, ensure_ascii=True, default=str)
                    + "\nCurrent section plan:\n"
                    + blueprint["section_plans"][section]
                    + "\nAuthoritative evidence:\n"
                    + json.dumps(
                        section_evidence,
                        ensure_ascii=True,
                        default=str,
                    )
                ),
                registry=section_registry,
            ).value
            section_bodies[section] = section_result["latex_body"]
            section_index = PAPER_SECTIONS.index(section) + 1
            section_filename = (
                f"{section_index:02d}_"
                + section.lower().replace(" ", "_")
                + ".tex"
            )
            (sections_dir / section_filename).write_text(
                section_bodies[section] + "\n",
                encoding="utf-8",
            )

        document = [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb,booktabs,graphicx}",
            r"\usepackage[hidelinks]{hyperref}",
            r"\title{" + _latex_escape_text(blueprint["title"]) + "}",
            r"\author{Agent Laboratory}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
        ]
        for section in PAPER_SECTIONS:
            document.extend(["", rf"\section{{{section}}}", section_bodies[section]])
            if section == "Results":
                for figure_index, figure in enumerate(available_figures, start=1):
                    document.extend([
                        "",
                        r"\begin{figure}[htbp]",
                        r"\centering",
                        rf"\includegraphics[width=0.82\linewidth]{{{figure}}}",
                        (
                            rf"\caption{{Observed experiment result visualization "
                            rf"{figure_index}.}}"
                        ),
                        r"\end{figure}",
                    ])
        if citations:
            document.extend(["", r"\begin{thebibliography}{99}"])
            for citation in citations:
                document.append(
                    rf"\bibitem{{{citation['key']}}} Reviewed work, "
                    rf"\texttt{{arXiv:{_latex_escape_text(citation['arxiv_id'])}}}."
                )
            document.append(r"\end{thebibliography}")
        document.extend(["", r"\end{document}", ""])
        latex = "\n".join(document)
        compile_result = compile_latex(
            latex,
            str(workdir),
            compile=bool(compile_pdf),
        )
        if "[CODE EXECUTION ERROR]" in compile_result:
            raise RuntimeError(compile_result)
        return {
            "success": True,
            "latex": latex,
            "compile_result": compile_result,
            "blueprint": blueprint,
            "sections": section_bodies,
        }


class ReviewerAgent:
    def __init__(self, model, api_key, reviewer_index, profile):
        self.model = model
        self.api_key = api_key
        self.reviewer_index = reviewer_index
        self.profile = profile

    def review(self, state):
        def submit_review(
            summary,
            strengths,
            weaknesses,
            originality,
            quality,
            clarity,
            significance,
            questions,
            limitations,
            ethical_concerns,
            soundness,
            presentation,
            contribution,
            overall_score,
            confidence,
            decision,
        ):
            ratings = {
                "originality": int(originality),
                "quality": int(quality),
                "clarity": int(clarity),
                "significance": int(significance),
                "soundness": int(soundness),
                "presentation": int(presentation),
                "contribution": int(contribution),
            }
            invalid = [
                name for name, value in ratings.items() if not 1 <= value <= 4
            ]
            if invalid:
                return {
                    "success": False,
                    "error": f"Ratings must be 1 to 4: {invalid}",
                }
            overall_score = int(overall_score)
            confidence = int(confidence)
            if not 1 <= overall_score <= 10:
                return {"success": False, "error": "overall_score must be 1 to 10"}
            if not 1 <= confidence <= 5:
                return {"success": False, "error": "confidence must be 1 to 5"}
            if decision not in {"Accept", "Reject"}:
                return {"success": False, "error": "decision must be Accept or Reject"}
            return {
                "success": True,
                "reviewer_index": self.reviewer_index,
                "summary": summary,
                "strengths": strengths,
                "weaknesses": weaknesses,
                **ratings,
                "questions": questions,
                "limitations": limitations,
                "ethical_concerns": bool(ethical_concerns),
                "overall_score": overall_score,
                "confidence": confidence,
                "decision": decision,
            }

        registry = ToolRegistry().register(
            "submit_review",
            "Submit an independent paper review. This does not revise the paper.",
            _schema(
                {
                    "summary": _string("Paper summary"),
                    "strengths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific strengths",
                    },
                    "weaknesses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific weaknesses",
                    },
                    "originality": {"type": "integer", "minimum": 1, "maximum": 4},
                    "quality": {"type": "integer", "minimum": 1, "maximum": 4},
                    "clarity": {"type": "integer", "minimum": 1, "maximum": 4},
                    "significance": {"type": "integer", "minimum": 1, "maximum": 4},
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Clarifying questions for the authors",
                    },
                    "limitations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Limitations and potential negative impacts",
                    },
                    "ethical_concerns": {
                        "type": "boolean",
                        "description": "Whether an ethics review is warranted",
                    },
                    "soundness": {"type": "integer", "minimum": 1, "maximum": 4},
                    "presentation": {"type": "integer", "minimum": 1, "maximum": 4},
                    "contribution": {"type": "integer", "minimum": 1, "maximum": 4},
                    "overall_score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                    "decision": {"type": "string", "enum": ["Accept", "Reject"]},
                },
                [
                    "summary",
                    "strengths",
                    "weaknesses",
                    "originality",
                    "quality",
                    "clarity",
                    "significance",
                    "questions",
                    "limitations",
                    "ethical_concerns",
                    "soundness",
                    "presentation",
                    "contribution",
                    "overall_score",
                    "confidence",
                    "decision",
                ],
            ),
            submit_review,
        )
        runtime = AgentRuntime(
            model=self.model,
            api_key=self.api_key,
            agent_name=f"ReviewerAgent#{self.reviewer_index}",
            phase="paper review",
            max_tool_steps=1,
        )
        return runtime.run_one(
            system_prompt=reviewer_prompt(self.reviewer_index, self.profile),
            user_prompt=(
                f"Research plan: {state.plan}\nObserved output: {state.experiment_output}\n"
                f"Supplementary experiments: "
                f"{json.dumps(getattr(state, 'supplementary_experiments', []), ensure_ascii=True)}\n"
                f"Fixed numerical seeds: "
                f"{json.dumps(getattr(state, 'seeds', {}), sort_keys=True)}\n"
                f"Paper: {state.paper}"
            ),
            registry=registry,
        ).value


class PaperRewardAgent:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key

    def score(self, state):
        def submit_paper_score(score, rationale):
            score = float(score)
            if not 0.0 <= score <= 10.0:
                return {"success": False, "error": "score must be between 0 and 10"}
            return {"success": True, "score": score, "rationale": rationale}

        registry = ToolRegistry().register(
            "submit_paper_score",
            "Submit the final paper score without revising any artifact.",
            _schema(
                {
                    "score": {"type": "number", "minimum": 0, "maximum": 10},
                    "rationale": _string("Rationale integrating all three reviews"),
                },
                ["score", "rationale"],
            ),
            submit_paper_score,
        )
        runtime = AgentRuntime(
            model=self.model,
            api_key=self.api_key,
            agent_name="PaperRewardAgent",
            phase="paper scoring",
            max_tool_steps=1,
        )
        return runtime.run_one(
            system_prompt=paper_reward_prompt(),
            user_prompt=(
                f"Fixed numerical seeds: "
                f"{json.dumps(getattr(state, 'seeds', {}), sort_keys=True)}\n"
                f"Paper: {state.paper}\nReviews: {state.reviews}"
            ),
            registry=registry,
        ).value
