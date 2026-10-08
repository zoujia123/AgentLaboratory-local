from dataclasses import asdict, dataclass, field


@dataclass
class ResearchState:
    query_id: str
    recording_id: str
    topic: str
    seeds: dict = field(default_factory=dict)
    literature_papers: list = field(default_factory=list)
    literature_review: str = ""
    plan: str = ""
    data_requirements: str = ""
    dataset_selection: dict = field(default_factory=dict)
    dataset_code: str = ""
    dataset_output: str = ""
    dataset_manifest: dict = field(default_factory=dict)
    dataset_validation_warnings: list = field(default_factory=list)
    experiment_code: str = ""
    experiment_output: str = ""
    supplementary_experiments: list = field(default_factory=list)
    interpretation_history: list = field(default_factory=list)
    interpretation: str = ""
    paper: str = ""
    readme: str = ""

    def to_dict(self):
        return asdict(self)
