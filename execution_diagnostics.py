"""Classify recurring runtime data errors into actionable repair categories."""


_RUNTIME_DATA_FAILURE_PATTERNS = (
    (
        "image_object_api_error",
        (
            "pil.image",
            "'pngimagefile' object is not subscriptable",
            "'jpegimagefile' object is not subscriptable",
            "'image' object has no attribute 'astype'",
            "'image' object has no attribute 'reshape'",
            "'image' object has no attribute 'shape'",
        ),
    ),
    (
        "dataset_split_seed_api_error",
        (
            "dataset.train_test_split() got an unexpected keyword argument 'random_state'",
            "dataset.train_test_split() got an unexpected keyword argument \"random_state\"",
        ),
    ),
    (
        "sklearn_train_test_split_seed_policy_error",
        (
            "train_test_split() is missing random_state=sklearn_seed",
            "train_test_split() must use random_state=sklearn_seed",
        ),
    ),
    (
        "local_replay_split_expression_error",
        (
            "split expressions are not replayable",
        ),
    ),
    (
        "list_of_records_access_error",
        (
            "list indices must be integers or slices, not str",
        ),
    ),
    (
        "non_finite_numeric_input",
        (
            "input x contains infinity",
            "input contains infinity",
            "input x contains nan",
            "input contains nan",
            "value too large for dtype",
        ),
    ),
    (
        "ragged_array_input",
        (
            "expected sequence of length",
            "setting an array element with a sequence",
            "ragged nested sequences",
        ),
    ),
    (
        "dataset_container_api_error",
        (
            "not supported between instances of 'int' and 'ellipsis'",
            "'list' object has no attribute 'astype'",
        ),
    ),
)


_DATASET_MAP_TRACE_MARKERS = (
    "datasets.arrow_dataset",
    "datasets\\arrow_dataset",
    "datasets/arrow_dataset",
    "dataset.map",
)


_BATCHED_IMAGE_LIST_FAILURES = (
    "'list' object has no attribute 'transpose'",
)


_RUNTIME_DATA_REPAIR_HINTS = {
    "image_object_api_error": (
        "The runtime value is a PIL image, not a mapping. Do not use "
        "image['bytes'] or image['path']. Use the observed image object's API: "
        "call .convert('RGB') only when the planned representation requires RGB, "
        "then use .resize(...) or np.asarray(image) as appropriate. Preserve the "
        "original mode when the experiment requires grayscale or alpha. If the value "
        "may be either a mapping or a PIL image, branch on its runtime type before "
        "accessing fields."
    ),
    "batched_dataset_map_error": (
        "For datasets.Dataset.map(batched=True), each batch field is a list of "
        "examples, not one example. Do not pass batch['image'] directly to a PIL "
        "operation; transform each image with a list comprehension and return a "
        "same-length image list, or use batched=False to receive one image per call. "
        "Return a mapping whose output fields have equal lengths. If intentionally "
        "changing cardinality, remove incompatible original columns and keep every "
        "returned field aligned."
    ),
    "dataset_split_seed_api_error": (
        "Use datasets.Dataset.train_test_split(seed=...) for a Hugging Face Dataset. "
        "Use random_state=... only with sklearn splitters operating on arrays or "
        "DataFrames; do not pass sklearn-only arguments to Dataset methods."
    ),
    "sklearn_train_test_split_seed_policy_error": (
        "This code uses sklearn.model_selection.train_test_split. The execution "
        "policy requires random_state=SKLEARN_SEED for every sklearn split. If the "
        "task requires chronological, temporal, or order-preserving evaluation, first "
        "preserve the required order and prefer an explicit deterministic boundary with "
        "positional slicing instead of train_test_split. Otherwise add "
        "random_state=SKLEARN_SEED to the sklearn split. Do not use random_state with "
        "datasets.Dataset.train_test_split; that API uses seed=SKLEARN_SEED."
    ),
    "local_replay_split_expression_error": (
        "The local material replay accepts one exact split string, not a list, "
        "slice, or split expression. Prefer separate load_dataset calls with one "
        "exact split per call for only the splits needed. If the code needs a "
        "DatasetDict, call load_dataset without split (or with split=None) and then "
        "select exact split names from it. Use only names listed in the latest "
        "inspection result; keep the selected dataset fixed and do not search again."
    ),
    "list_of_records_access_error": (
        "A list represents rows, not named columns. Iterate records and extract a "
        "field, convert the records to a DataFrame, or use a datasets.Dataset column "
        "API. Do not use list['column']."
    ),
    "non_finite_numeric_input": (
        "Rebuild the numerical feature pipeline: convert +/-inf to NaN immediately "
        "after feature computation, then check percentage changes, zero denominators, "
        "and source missingness. Inspect the traceback to identify whether the "
        "non-finite value is in X, y/y_true, or y_pred. If it is in y, y_true, or "
        "a target column, convert None, NaN, and +/-inf to missing values, build a "
        "finite-target row mask before splitting, and drop or correctly construct "
        "those target rows. Do not impute targets by default unless the research "
        "design explicitly justifies it. Apply the same row mask to every aligned "
        "feature and label variant. If it is in X, clean or impute features using "
        "statistics fitted on training data only. If it is in y_pred, repair the "
        "upstream computation instead of filling predictions blindly. Verify finite "
        "counts before every model fit and metric. Do not reverse the train/test "
        "split to hide missing values. Make a substantive code change and rerun; "
        "never resubmit the failed code hash."
    ),
    "ragged_array_input": (
        "Rebuild the input representation before batching or fitting. Normalize every "
        "sample to one shape, mode, and dtype; validate the collection with np.stack "
        "or an equivalent fixed-shape operation before creating tensors. Do not patch "
        "only downstream augmentation or model layers."
    ),
    "dataset_container_api_error": (
        "Keep datasets.Dataset, arrays, and list-of-records distinct. Use "
        "Dataset.train_test_split(seed=...) or split integer row indices; do not pass "
        "a Dataset directly to sklearn's splitter. When a split returns lists, access "
        "fields by iterating records rather than list['column']."
    ),
}


def classify_runtime_data_failure(error=None, traceback_text=""):
    text = "\n".join(str(value or "") for value in (error, traceback_text)).casefold()
    in_dataset_map = any(marker in text for marker in _DATASET_MAP_TRACE_MARKERS)
    if (
        (in_dataset_map or "batched=true" in text or "batched = true" in text)
        and ("expected length" in text or "got length" in text or "arrowinvalid" in text)
    ):
        return "batched_dataset_map_error"
    if in_dataset_map and any(
        fragment in text for fragment in _BATCHED_IMAGE_LIST_FAILURES
    ):
        return "batched_dataset_map_error"
    for category, fragments in _RUNTIME_DATA_FAILURE_PATTERNS:
        if any(fragment in text for fragment in fragments):
            return category
    return None


def runtime_data_repair_hint(category):
    return _RUNTIME_DATA_REPAIR_HINTS.get(category)


def runtime_object_guidance():
    """Short, query-independent guidance for common local container APIs."""
    return (
        "Runtime object discipline: use the observed runtime type and its API; do not "
        "assume every image is an ndarray, every dataset is a DataFrame, or every "
        "collection supports named-column indexing. For PIL.Image.Image, use image "
        "methods such as .convert(...) and .resize(...) before array conversion when "
        "appropriate. In Dataset.map(batched=True), returned fields must remain "
        "length-aligned. Dataset.train_test_split uses seed=..., whereas sklearn "
        "splitters use random_state=.... For a list of records, iterate records or "
        "convert it before selecting a named field."
    )
