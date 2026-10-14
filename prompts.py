DATA_PREPARATION_OUTPUT_GUIDANCE = (
    "Keep explicit data-preparation prints to a short diagnostic summary, aiming "
    "for under 2000 characters in total. Prefer row counts, shapes, dtypes, and "
    "aggregate missing/non-finite counts. If examples are necessary, show only a "
    "few scalar values or short text excerpts. Bound both the number of displayed "
    "items and the length of each value; an outer [:5] slice or head() does not "
    "bound nested dictionaries, arrays, or long text inside those items. For nested "
    "containers, report their type, shape, or length and only bounded scalar "
    "previews. Do not print complete datasets, tables, columns, dictionaries, "
    "arrays, tensors, images, or binary contents. Do not convert entire objects "
    "with .tolist(), .to_dict(), or serialization merely to print them; these "
    "conversions can bypass abbreviated displays. Avoid per-row or per-batch "
    "printing. If detailed diagnostic data must be retained, save it to a file "
    "in the working directory and print only its path and a brief description. "
    "Preview limits apply only to display: preserve all data, validation checks, "
    "splits, and sample sizes required by the research task. Preparation code is "
    "rerun before experiments, so its prints repeat and share the captured "
    "experiment output; leave room for the experiment's actual results."
)


# Selection guidance for the literature review.
LITERATURE_RELEVANCE_RULE = (
    "Add the paper if it has a clear, useful connection to the broader research "
    "background; an exact experimental match is unnecessary."
)

LITERATURE_RETRY_RELEVANCE_RULE = (
    "You did not choose a paper in the first round. "
    "This is your last chance to choose. "
    "Choose the paper closest to the research topic, even if it is not a close match. "
    "Set add=true and explain the connection. Then submit the literature review."
)


def literature_review_prompt():
    return (
        "You are a computer science PhD student. Read the available papers and choose at least one for the literature review. "
        "The paper does not need to study exactly the same question. "
        "Explain how it helps with the topic and what it cannot tell us."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def phd_plan_feedback_prompt():
    return (
        "You are a computer science PhD student. Review the proposed experiment plan and suggest only changes necessary to answer the research question correctly."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def postdoc_interpretation_feedback_prompt():
    return (
        "You are a computer science postdoctoral researcher. Review the PhD student's "
        "interpretation of observed results and assess the need for additional evidence."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def postdoc_plan_prompt():
    return (
        "You are a computer science postdoctoral researcher. Develop a minimal, feasible experiment plan that answers the research question, using the literature and PhD feedback."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def phd_interpretation_prompt():
    return (
        "You are a computer science PhD student. Develop an evidence-based "
        "interpretation of experimental results using the research context and Postdoc feedback."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def ml_dataset_selection_prompt():
    return (
        "You are a machine learning engineer. Prepare reusable data for the research task from the provided local datasets, following the data-preparation plan."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def sw_data_preparation_prompt():
    return (
        "You are a software engineer. Review data preparation and suggest necessary fixes, without adding model training or evaluation."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def professor_readme_prompt():
    return (
        "You are a computer science professor. Write an accurate English research "
        "README grounded in the supplied evidence."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def experiment_solver_prompt():
    return (
        "You are a machine learning engineer. Implement the current experiment task using "
        "the prepared data and produce reproducible, observed results."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def paper_blueprint_prompt():
    return (
        "You are a research paper writer. Develop a concise blueprint for an English "
        "research paper grounded in the supplied evidence."
        "\nAt each turn, choose exactly one available tool and call it once."
    )


def paper_section_prompt():
    return (
        "You are a research paper writer. Write the requested section of an English "
        "research paper using the supplied blueprint and evidence."
        "\nAt each turn, choose exactly one available tool and call it once."
    )
