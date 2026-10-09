"""Shared failure classification and repair hints for preparation and experiments."""

import ast


_DATASET_MAP_TRACE_MARKERS = (
    "datasets.arrow_dataset",
    "datasets\\arrow_dataset",
    "datasets/arrow_dataset",
    "dataset.map",
)


_BATCHED_IMAGE_LIST_FAILURES = (
    "'list' object has no attribute 'transpose'",
)


_DATASET_MAP_CALLBACK_MARKERS = (
    "dataset.map",
    "in apply_function_on_filtered_inputs",
    "in _map_single",
)


# Decoded Image features can use format-specific PIL classes without a PIL frame
# in the traceback. Match their object errors, not an arbitrary mention of shape.
_PIL_IMAGE_OBJECT_FAILURES = tuple(
    f"'{image_type}' object {failure}"
    for image_type in (
        "image", "imagefile", "pngimagefile", "jpegimagefile", "tiffimagefile",
        "gifimagefile", "bmpimagefile", "webpimagefile", "ppmimagefile", "icoimagefile",
    )
    for failure in (
        "is not subscriptable",
        "has no attribute 'shape'",
        "has no attribute 'dtype'",
        "has no attribute 'astype'",
        "has no attribute 'reshape'",
        "has no attribute 'flatten'",
    )
)


# 每条规则为 (category, groups)：组内任意片段命中，组间全部满足。
# 从上往下匹配，首个命中的类别立即返回；同一类别可有多条规则。
# 例如 (("nameerror: name ",), ("is not defined",)) 要求两段文字同时出现。
_RUNTIME_DATA_FAILURE_PATTERNS = (
    (
        "offline_resource_access_error",
        (("network and subprocess access are disabled",),),
    ),
    (
        "nltk_resource_missing",
        (
            ("resource",),
            ("not found",),
            ("nltk", "corpora/stopwords", "tokenizers/punkt"),
        ),
    ),
    (
        "undefined_variable_error",
        (
            ("nameerror: name ",),
            ("is not defined",),
        ),
    ),
    (
        "cross_validation_scoring_api_error",
        (
            ("for evaluating multiple scores",),
            ("cross_validate",),
        ),
    ),
    (
        "cross_validation_scoring_api_error",
        (
            (
                "'scoring' parameter of cross_val_score",
                '"scoring" parameter of cross_val_score',
            ),
            ("got {", "got [", "got ("),
        ),
    ),
    (
        "local_replay_split_expression_error",
        (
            ("selected dataset resource mismatch",),
            ("split [", "split=[", "split 'train+test'", 'split "train+test"'),
        ),
    ),
    (
        "batched_dataset_map_error",
        (
            _DATASET_MAP_TRACE_MARKERS + ("batched=true", "batched = true"),
            ("expected length", "got length", "arrowinvalid"),
        ),
    ),
    (
        "batched_dataset_map_error",
        (
            _DATASET_MAP_TRACE_MARKERS,
            _BATCHED_IMAGE_LIST_FAILURES,
        ),
    ),
    (
        "dataset_map_argument_error",
        (
            _DATASET_MAP_TRACE_MARKERS,
            _DATASET_MAP_CALLBACK_MARKERS,
            ("got multiple values for argument",),
        ),
    ),
    (
        "batched_dataset_text_map_error",
        (
            _DATASET_MAP_TRACE_MARKERS,
            (
                "expected string or bytes-like object, got 'list'",
                "'list' object has no attribute 'split'",
            ),
            _DATASET_MAP_CALLBACK_MARKERS,
        ),
    ),
    (
        "hf_dataset_column_indexing_error",
        (
            ("invalid literal for int()",),
            ("_check_valid_index_key",),
            ("datasets",),
        ),
    ),
    (
        "image_array_interface_error",
        (
            ("has no attribute '__array_interface__'",),
            ("fromarray",),
            ("pil.image", "pil\\image.py", "pil/image.py"),
        ),
    ),
    (
        "list_shape_access_error",
        (("'list' object has no attribute 'shape'",),),
    ),
    (
        "image_object_api_error",
        (("pil.image",) + _PIL_IMAGE_OBJECT_FAILURES,),
    ),
    (
        "dataset_split_seed_api_error",
        (
            (
                "dataset.train_test_split() got an unexpected keyword argument 'random_state'",
                "dataset.train_test_split() got an unexpected keyword argument \"random_state\"",
            ),
        ),
    ),
    (
        "sklearn_train_test_split_seed_policy_error",
        (
            (
                "train_test_split() is missing random_state=sklearn_seed",
                "train_test_split() must use random_state=sklearn_seed",
            ),
        ),
    ),
    (
        "local_replay_split_expression_error",
        (("split expressions are not replayable",),),
    ),
    (
        "mapping_numeric_input",
        (("float() argument must be a string or a real number, not 'dict'",),),
    ),
    (
        "non_numeric_model_input",
        (("could not convert string to float",),),
    ),
    (
        "list_of_records_access_error",
        (("list indices must be integers or slices, not str",),),
    ),
    (
        "string_indexing_error",
        (("string indices must be integers",),),
    ),
    (
        "non_finite_numeric_input",
        (
            (
                "input x contains infinity",
                "input contains infinity",
                "input x contains nan",
                "input contains nan",
                "value too large for dtype",
            ),
        ),
    ),
    (
        "ragged_array_input",
        (
            (
                "all input arrays must have the same shape",
                "expected sequence of length",
                "setting an array element with a sequence",
                "ragged nested sequences",
            ),
        ),
    ),
    (
        "tensor_device_transfer_error",
        (
            (
                "'list' object has no attribute 'to'",
                "'numpy.ndarray' object has no attribute 'to'",
            ),
        ),
    ),
    (
        "dataset_container_api_error",
        (
            (
                "not supported between instances of 'int' and 'ellipsis'",
                "'list' object has no attribute 'astype'",
                "'datasetdict' object has no attribute 'train_test_split'",
            ),
        ),
    ),
    (
        "missing_key_or_column",
        (("keyerror:",),),
    ),
    (
        "tensor_shape_mismatch",
        (("shapes cannot be multiplied", "size mismatch"),),
    ),
    (
        "resource_exhaustion",
        (("not enough memory", "out of memory"),),
    ),
    (
        "data_representation_error",
        (("cannot handle this data type",),),
    ),
    (
        "evaluation_subset_alignment",
        ((
            "inconsistent numbers of samples",
            "found input variables with inconsistent numbers of samples",
        ),),
    ),
    (
        "unavailable_model_asset",
        (("from_pretrained", "can't load tokenizer"),),
    ),
    (
        "unavailable_torchvision",
        ((
            "no module named 'torchvision'",
            'no module named "torchvision"',
            "no module named 'torchvision.",
            'no module named "torchvision.',
        ),),
    ),
    (
        "unavailable_dependency",
        (("no module named", "cannot import name"),),
    ),
    (
        "seed_policy",
        (("seed policy violations",),),
    ),
    (
        "pandas_positional_indexing_error",
        ((
            "positional indexers are out-of-bounds",
            "single positional indexer is out-of-bounds",
        ),),
    ),
    (
        "indexing_error",
        (("index out of range", "not found in axis"),),
    ),
)


_RUNTIME_DATA_REPAIR_HINTS = {
    "list_shape_access_error": (
        "The object accessed with .shape is a Python list. Use the traceback to "
        "identify that exact variable; an error on labels does not mean the image "
        "arrays need reshaping. If only the sample count is needed, use len(values) "
        "instead of values.shape. Hugging Face label columns may be Python lists, "
        "and sklearn train_test_split preserves list inputs as lists, so y_train "
        "and y_val can remain lists even when X_train is an ndarray. If array "
        "operations are required, convert the labels before splitting, for example "
        "train_labels = np.asarray(train_labels), and also convert test_labels = "
        "np.asarray(test_labels) before accessing its shape. Use the verified target "
        "dtype: np.int64 is appropriate for integer class labels, but do not blindly "
        "cast regression targets or string labels to integers. Check every affected "
        "label variable, including y_train, y_val, and test_labels; scalar targets "
        "should be one-dimensional and match the corresponding feature row counts. "
        "Preserve row order, split proportions, and fixed seeds. For lists of images "
        "or records, inspect their elements and build the intended representation "
        "instead of blindly coercing the whole list into a numeric array."
    ),
    "offline_resource_access_error": (
        "The execution environment deliberately blocks network and subprocess access. "
        "Remove runtime downloads, network requests, and installation commands; "
        "retrying the same request cannot obtain the resource. Use the inspected "
        "local datasets and resources already available in installed packages. For "
        "NLTK stopwords or tokenizer data, replace the resource dependency with a "
        "fixed embedded word list or sklearn.feature_extraction.text.ENGLISH_STOP_WORDS "
        "and resource-free tokenization such as Python re. Remove dependent calls "
        "to nltk.corpus.stopwords.words or nltk.word_tokenize if their data is missing, "
        "as well as nltk.download. For a negation comparison, use the same tokenizer "
        "and base stopword set for both variants, explicitly include the target "
        "negation words in the removal set, and subtract them for the retained variant. "
        "Generate both variants independently from the same original text."
    ),
    "nltk_resource_missing": (
        "NLTK is installed, but the requested corpus or tokenizer data is not local. "
        "Package availability does not imply availability of stopwords, punkt, or "
        "punkt_tab. This environment is offline: do not call nltk.download or install "
        "resources. For stopwords, use a fixed embedded set or the installed sklearn "
        "ENGLISH_STOP_WORDS constant. Replace resource-dependent tokenization with "
        "one consistent tokenizer using Python re, or NLTK wordpunct_tokenize, which "
        "does not require downloaded data. For a comparison that removes versus "
        "retains not, no, and never, define negation_words = {'not', 'no', 'never'}, "
        "base_stopwords = set(ENGLISH_STOP_WORDS) | negation_words, and "
        "retained_stopwords = base_stopwords - negation_words; do not assume every "
        "default list contains all target words. Apply each set independently to "
        "the same original tokens, preserving sample order, labels, and train/test "
        "splits. Keep the chosen word list and all other preprocessing fixed."
    ),
    "dataset_map_argument_error": (
        "Check the Dataset.map callback signature against its positional arguments "
        "and fn_kwargs. By default map passes a row or batch as the first argument; "
        "fn_kwargs={'text': 'text'} passes the literal string 'text' again and does "
        "not select the text column. Bind each parameter only once. Either accept "
        "a row/batch and access its ['text'] field, or use input_columns='text' with "
        "a callback designed for that column. With batched=True the column is a "
        "list of strings: process each string and return a dictionary of equally "
        "long output lists. With batched=False process one string and return a "
        "dictionary of output fields. For paired stopword variants, return both "
        "derived text columns from the same original input and retain both outputs. "
        "Keep existing supervised train/test splits when supplied; a DatasetDict is a mapping "
        "of splits and has no train_test_split method."
    ),
    "undefined_variable_error": (
        "Locate the undefined name and distinguish a missing module import or alias "
        "from a missing computed variable. Check spelling, imports, scope, and "
        "assignments before the failing line. For example, random.seed(...) requires "
        "import random; random is a Python standard-library module and needs no "
        "installation. Check the supplied environment before adding an import: "
        "an undefined transforms name does not establish that torchvision is "
        "available. If a dependency is unavailable, replace its dependent calls "
        "using installed modules instead of restoring the unavailable import. "
        "For exact-text edits, old_text must be an existing unique fragment copied "
        "from the current code, and new_text must differ from old_text. To insert "
        "import random, an example is old_text=\"import numpy as np\" and "
        "new_text=\"import numpy as np\\nimport random\", only if that anchor exists "
        "exactly once. Do not use the missing import itself as old_text. If matched "
        "0 is reported, select a real unique anchor or submit a corrected complete "
        "replacement; do not repeat the rejected edit. "
        "Each code execution starts in a fresh worker. "
        "For a supplementary experiment, only the validated dataset code is "
        "automatically prepended; the original experiment code is read-only "
        "reference and its variables or fitted models are not automatically "
        "available. Recreate the minimal required preprocessing or model from the "
        "prepared variables before using it. For example, if X_train_np was defined "
        "only in the main experiment, construct it from the prepared X_train and "
        "the same verified ordered numeric feature names using np.column_stack "
        "over numeric columns. Preserve labels, row alignment, the requested split, "
        "model settings, and fixed seeds. Check prerequisites on a small batch "
        "before expensive evaluation. Do not invent placeholder data, change only "
        "later printing/statistics, or resubmit code with the missing assignment."
    ),
    "cross_validation_scoring_api_error": (
        "cross_val_score supports one scoring metric and returns one array of fold "
        "scores. For multiple metrics supplied as a dictionary or list, use "
        "sklearn.model_selection.cross_validate with the same estimator, training "
        "data, requested folds, and scoring specification. Its result is a mapping "
        "with keys such as test_accuracy, test_precision, test_recall, and test_f1; "
        "compute each mean and standard deviation from results['test_' + metric], "
        "not results[metric]. Include the correct import and keep metric names and "
        "scorers valid. Preserve the requested evaluation scope and fixed seeds; "
        "training-only cross-validation must not include the held-out test set. "
        "Fit any learned preprocessing within each training fold using a Pipeline. "
        "If only one metric is required, retain cross_val_score with a single "
        "valid scorer and summarize its returned array directly."
    ),
    "image_object_api_error": (
        "Use the traceback to identify the exact failing value. When the runtime "
        "value is a PIL image, including a format-specific ImageFile subclass, "
        "use .size for (width, height), or np.asarray(image) before accessing "
        ".shape, .dtype, .astype(), .reshape(), or .flatten(). Array shape is "
        "(height, width) with an optional channel dimension, not PIL's size order. "
        "Hugging Face Image features may decode arrays written by Dataset.map "
        "back into PIL images, possibly with a different format-specific class. "
        "Convert after reading the image from the dataset, or use an explicitly "
        "numeric feature column; conversion inside map alone does not establish "
        "the type on a later read. Inspect the actual decoded type rather than "
        "assuming the initially inspected type is unchanged. Do not use "
        "image['bytes'] or image['path'] on a PIL image; branch on the actual type "
        "if a mapping is also possible. Use .resize(...) as required, and "
        ".convert('RGB') only when the planned representation requires RGB. "
        "Preserve required grayscale or alpha modes. Verify one decoded sample's "
        "type, array shape, dtype, and pixel range before rerunning the full work."
    ),
    "image_array_interface_error": (
        "PIL.Image.fromarray received an object without the required array "
        "interface. Inspect the actual value passed to it. Values written as "
        "NumPy arrays by Dataset.map may be read back as nested Python lists. "
        "Use the map callback's batched setting to distinguish a batch of images "
        "from one image's nested pixel list; isinstance(value, list) alone cannot "
        "distinguish them. For each individual image, use np.asarray to obtain "
        "an array, then verify its shape, channels, dtype, and pixel range before "
        "calling Image.fromarray. Choose a compatible dtype and image mode from "
        "those values; use uint8 only when the data represents 8-bit pixels, and "
        "do not blindly cast normalized floating-point pixels or reshape them. "
        "Convert lists to arrays before array methods such as flatten as well. "
        "If the value is already a PIL image, use its image methods directly. "
        "Alternatively, derive the required representations from the original "
        "PIL image in one pass to avoid the intermediate storage round trip. "
        "Check a small batch before rerunning the full experiment. Renaming "
        "variables, moving convert calls, or only toggling batched does not "
        "correct the object passed to Image.fromarray."
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
    "batched_dataset_text_map_error": (
        "Inspect the failing Dataset.map callback and its batched argument. With "
        "batched=True, batch['text'] is a list of strings for the batch; this does "
        "not mean the raw dataset stores lists in its text column. A scalar text "
        "function such as re.sub or text.split must receive one string at a time. Apply it to "
        "each string, for example {'text_no_urls': [remove_urls(text) for text in "
        "batch['text']]}, and return a same-length list. Alternatively, use "
        "batched=False and pass example['text'] to the scalar function. Repair "
        "the callback using the inspected schema; do not stringify the entire "
        "list or request another inspection solely because this error mentions "
        "a list. For stopword preprocessing, return a dictionary such as "
        "{'text_no_stopwords': [remove_all_stopwords(text) for text in batch['text']], "
        "'text_negation': [retain_negation_words(text) for text in batch['text']]}. "
        "Both variants must be derived from the original text; deleted negation "
        "words cannot be recovered by another cleaning pass. Assign the map result "
        "and preserve row alignment and labels. If load_dataset returned a "
        "DatasetDict, use its existing supervised train/test entries rather than "
        "calling train_test_split on the container."
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
        "inspection result; keep the selected dataset fixed and do not search again. "
        "In particular, split='train+test' is not a recorded split name. When the "
        "inspected 'train' and 'test' splits exist, load them separately and preserve "
        "their held-out roles; otherwise use the actual inspected split names. "
        "Exclude unlabeled 'unsupervised' data from supervised training and "
        "evaluation unless the research task explicitly requires its use."
    ),
    "mapping_numeric_input": (
        "The estimator received mapping/dict rows instead of scalar numeric features. "
        "Rebuild the data-to-model boundary as a dense 2-D numeric matrix from "
        "explicitly selected scalar feature fields. Do not pass np.array(dataset), "
        "np.asarray(dataset.select_columns(...)), or np.array(list_of_dicts) directly "
        "to sklearn; these can produce a 1-D object array of dicts. Iterating as "
        "[row for row in dataset] still returns dictionaries. list(row.values()) "
        "removes the dictionary wrapper but does not make string fields numeric. "
        "For numerical-only features, select scalar numeric columns from the "
        "inspected schema, excluding the target and target-derived fields. Use an "
        "explicit ordered feature list, for example X = np.column_stack([np.asarray("
        "dataset[name], dtype=np.float64) for name in numeric_features]). For records, "
        "extract row[name] in that same order. Extract y from the target column as "
        "a 1-D scalar array rather than an array of one-field mappings. Before a full "
        "rerun, check a small batch: X.ndim == 2, numeric dtype, no dict/string cells, "
        "y.ndim == 1, and matching row counts. A successful shape print does not "
        "validate the representation; X.shape == (N,) is not a feature matrix."
    ),
    "non_numeric_model_input": (
        "A string reached numeric conversion. Rebuild one consistent numeric feature "
        "representation for every required split; do not patch only one model call or "
        "repair a hand-written feature list one column at a time. Derive the feature "
        "list from the authoritative inspected schema and runtime sample types when "
        "available: retain only scalar integer or floating-point fields, and exclude "
        "the target, target-derived fields, and all string, categorical, or structured "
        "fields. A variable named numerical_features or a semantic-looking column name "
        "does not establish numeric dtype. Use one explicit ordered list for every "
        "split, build X as a 2-D numeric matrix (for example with np.column_stack and "
        "dtype=np.float64), and extract y separately as a 1-D target. If the research "
        "plan intentionally uses categorical features, fit an encoder on training data "
        "only and apply it unchanged to held-out data. Do not parse identifiers such as "
        "IP addresses or silently coerce invalid strings to NaN/zero. Before fit, "
        "verify X.ndim == 2, X has numeric dtype, and X/y row counts match. Make a "
        "substantive correction before retrying; never resubmit the failed code hash."
    ),
    "list_of_records_access_error": (
        "A list represents rows, not named columns. Iterate records and extract a "
        "field, convert the records to a DataFrame, or use a datasets.Dataset column "
        "API. Do not use list['column']."
    ),
    "string_indexing_error": (
        "A string is being indexed as if it were a record or mapping. Match the "
        "traceback line number to the submitted code, identify the indexed variable, "
        "and trace its assignment. Do not assume the failure is in a Dataset.map "
        "callback unless the traceback points there. If the code directly unpacks "
        "Hugging Face Dataset.train_test_split(...), that method returns a DatasetDict: "
        "train_dataset, held_out_dataset = dataset.train_test_split(...) assigns the "
        "strings 'train' and 'test', not the split datasets. Instead use "
        "splits = dataset.train_test_split(test_size=0.2, seed=SKLEARN_SEED), then "
        "train_dataset = splits['train'] and held_out_dataset = splits['test']. "
        "Preserve the intended split proportions, stratification, and shared row "
        "indices for paired representations. Reported sizes 5 and 4 may be lengths "
        "of the split-name strings; verify the split objects and actual row counts "
        "before accessing records. Correct the assignment that produced the string; "
        "renaming a callback parameter does not repair it. If the string came from "
        "another source, correct that source or its access using the failing line."
    ),
    "non_finite_numeric_input": (
        "Locate the failing fit, predict, or metric call in the traceback and "
        "current code. Identify the actual objects passed to that call and whether "
        "the invalid values are in X, y/y_true, or y_pred; changing another input "
        "does not repair the failing one. Check the inputs of every compared model "
        "or representation on each required split, after feature computation, "
        "preprocessing, and dtype conversion and before each model call or metric. "
        "Inspect shapes, dtypes, and non-finite counts by input/column. Use np.isfinite "
        "for dense numeric arrays or torch.isfinite for tensors; inspect stored "
        "values of sparse inputs without densifying them. Trace invalid values back "
        "to missing source data, invalid arithmetic, zero denominators, or overflow. "
        "Finite but very large values can overflow during conversion or subsequent "
        "computation, so verify the intended numeric scale and dtype too. Where "
        "missing-value treatment is appropriate, convert +/-inf to NaN; dropna alone "
        "does not remove infinities. Choose removal, imputation, or a corrected "
        "computation according to the data meaning and research protocol; do not "
        "default to filling with zero or arbitrary clipping. Fit feature imputation "
        "or scaling on training data only and reuse that fitted transformation on "
        "held-out data. Do not impute targets by default unless the research design "
        "explicitly justifies it. For invalid predictions, repair the upstream "
        "computation instead of replacing predictions blindly. If excluding rows "
        "is justified, apply the same row mask to aligned features, targets, "
        "predictions, and row IDs within that split. For paired comparisons, retain "
        "the same evaluation rows across variants and report exclusion rules and "
        "counts; do not silently remove test cases or apply a training mask to a "
        "different split. Preserve the requested split and fixed seeds. Verify a "
        "small batch, then check the full actual inputs before training. When using "
        "exact-text edits, copy a unique existing fragment from the current code, "
        "including intervening lines and whitespace, and ensure new_text differs "
        "from old_text. If matched 0 is reported, locate a valid anchor or submit a "
        "corrected complete replacement rather than repeating the same patch. "
        "Make a substantive code change; never resubmit the failed code hash."
    ),
    "ragged_array_input": (
        "Rebuild the input representation before batching or fitting. Normalize every "
        "sample to one shape, mode, and dtype; validate the collection with np.stack "
        "or an equivalent fixed-shape operation before creating tensors. For images, "
        "apply the planned color-mode conversion and resize to every image before "
        "np.asarray and np.stack; converting to RGB alone does not unify image sizes. "
        "Use the same dimensions for training, augmentation, and validation, consistent "
        "with the model's input shape. Do not patch "
        "only downstream augmentation or model layers."
    ),
    "tensor_device_transfer_error": (
        "The value passed to .to(device) is a list or NumPy array, not a torch.Tensor. "
        "Inspect a dataset row and one DataLoader batch before training. Returning "
        "tensors from Hugging Face Dataset.map does not guarantee tensor-valued reads: "
        "Arrow-backed data can be returned as Python lists. For preprocessed numeric "
        "columns, assign dataset = dataset.with_format('torch', columns=[...]) using "
        "the actual input and label columns on every required split, or convert each "
        "sample with torch.as_tensor in Dataset.__getitem__ before collation. Do not "
        "try to fix this by adding .tolist() or only returning tensors from map again. "
        "For CNN images, verify floating-point CHW samples and NCHW batches; do not "
        "permute data already in that layout. For CrossEntropyLoss classification, "
        "return scalar integer class labels and verify a torch.long batch of shape [N]. "
        "When using edit_experiment_code, copy old_text exactly from the current "
        "experiment code, including return statements and indentation. If matched 0 "
        "is reported, choose an existing unique fragment or replace the experiment "
        "code; do not repeat the same rejected edit."
    ),
    "dataset_container_api_error": (
        "Keep datasets.Dataset, arrays, and list-of-records distinct. Use "
        "Dataset.train_test_split(seed=...) or split integer row indices; do not pass "
        "a Dataset directly to sklearn's splitter. Dataset.train_test_split returns "
        "a DatasetDict: store it as splits, then read splits['train'] and splits['test']. "
        "Direct unpacking yields the split-name strings. When a split returns lists, access "
        "fields by iterating records rather than list['column']. Loading without "
        "a split may already return a DatasetDict. Use its inspected train/test "
        "entries directly when available; it has no train_test_split method. If another split is "
        "required, call that method on one labeled Dataset, not the DatasetDict, "
        "and keep the designated held-out test set separate."
    ),
    "hf_dataset_column_indexing_error": (
        "The traceback indicates that Hugging Face Dataset row-index validation "
        "received a string that could not be converted to an integer. Locate the "
        "failing expression and determine whether it should select rows or columns. "
        "Use valid integer indices for rows, dataset['column'] for one column, or "
        "dataset.select_columns(column_names) for multiple columns. Do not remove "
        "a feature solely because its name appears in this conversion error. "
        "Choose the resulting representation for the downstream API: retain a "
        "Dataset when supported; if a numeric matrix is required, explicitly "
        "assemble appropriately preprocessed numeric columns into a two-dimensional "
        "array, preserving feature order and sample alignment with targets. "
        "Converting an entire Dataset directly with np.array can produce object "
        "rows rather than numeric features. Do not blindly coerce text, images, "
        "or other nonnumeric content to floating point."
    ),
    "missing_key_or_column": (
        "Use the traceback to identify the failing object and missing key. If the "
        "object is a DataFrame, inspect that exact frame's current columns, not "
        "only the original table. Trace where the required column is created and "
        "when each split is constructed: adding a column to the parent table does "
        "not add it to previously created train/validation/test frames. Create "
        "the feature on the frames actually used, or rebuild them from the updated "
        "table using their original row identities, order, and split boundaries. "
        "fillna() treats missing values in existing columns; it cannot repair a "
        "column absent from another frame. Do not replace an intended feature "
        "with zeros or drop it merely to avoid the error. Fit learned preprocessing "
        "on training data only; preserve temporal history and never use future "
        "values. Compare with the raw schema to distinguish a misspelling, a "
        "missing derived feature, or a column changed by encoding, removal, "
        "renaming, merging, or index operations. Preserve required target and "
        "analysis columns through transformations. For a Hugging Face Dataset, inspect the "
        "column_names of the actual failing split. Dataset.map returns a new dataset; "
        "assign its result and create every required derived column on both train "
        "and held-out splits. Mapping only the test split does not add columns to "
        "the training split. For paired representations, build both on the same "
        "rows and labels, and use each representation's own columns for training "
        "and evaluation; do not substitute the unmodified data. Default "
        "concatenate_datasets appends rows; it does not add the other variant's "
        "column to each original row and can introduce None values. Repair the "
        "feature creation and verify columns and missing values on every split "
        "and alignment with its targets before fitting a vectorizer or model. For "
        "a dictionary or another container, inspect its actual keys and access "
        "API instead of applying DataFrame advice. Verify the cause from the code "
        "and runtime evidence; a KeyError alone does not establish that a "
        "transformation caused it."
    ),
    "tensor_shape_mismatch": (
        "Trace tensor shapes through the failing operations and correct incompatible "
        "dimensions. During model training, trace every layer and replace the "
        "complete model definition when necessary."
    ),
    "resource_exhaustion": (
        "Reduce data, model, and intermediate tensor sizes before retrying; verify "
        "broadcasting shapes explicitly."
    ),
    "data_representation_error": (
        "Inspect the actual array layout and dtype, then rebuild the complete "
        "conversion pipeline consistently."
    ),
    "evaluation_subset_alignment": (
        "Keep features, labels, predictions, and group labels aligned when filtering "
        "or splitting rows. For evaluation, build one boolean mask from the "
        "evaluation split and use that same mask to subset y_true, y_pred, and every "
        "aligned group label. Verify equal lengths before computing metrics; do not "
        "use an indexer derived from the full split to index a group-filtered subset."
    ),
    "unavailable_model_asset": (
        "No pretrained assets are materialized. During the experiment stage, use "
        "a model trained from scratch with available local modules. Keep model "
        "training out of data preparation."
    ),
    "unavailable_dependency": (
        "Rewrite with modules and public APIs listed in the execution environment."
    ),
    "unavailable_torchvision": (
        "torchvision is unavailable in this execution environment. Do not install "
        "or download it, repeat its import, or remove the import while leaving "
        "transforms.* calls behind. Replace the entire torchvision-dependent "
        "preprocessing pipeline, including Compose, Resize, RandomHorizontalFlip, "
        "ToTensor, and Normalize wherever used in training and evaluation, with "
        "available PIL, NumPy, and torch operations. Use already-decoded PIL images "
        "directly rather than assuming image['path'] exists. Preserve the planned "
        "color mode/channel count, transform order, interpolation, image dimensions "
        "and aspect ratio. PIL resize takes (width, height), whereas a torchvision "
        "Resize tuple is (height, width). Apply PIL horizontal flipping only where "
        "the original training augmentation requires it, with the original "
        "probability and fixed seeds; keep validation free of random augmentation. "
        "For 8-bit pixels, np.asarray(image, dtype=np.float32) / 255.0 reproduces "
        "ToTensor scaling; do not divide already-scaled or floating-point pixels "
        "blindly. Add a channel axis for grayscale, transpose HWC to CHW, make the "
        "array contiguous with np.ascontiguousarray, then use torch.from_numpy. "
        "Preserve any original per-channel Normalize(mean, std) using "
        "(tensor - mean[:, None, None]) / std[:, None, None] with matching tensors. "
        "Check a small batch for float32 dtype, expected NCHW shape, labels, and a "
        "successful model forward pass before full training. This replacement "
        "must preserve the classifier, split, and augmentation comparison."
    ),
    "seed_policy": (
        "Use the exact symbolic fixed seed required by the seed policy. Prefer "
        "explicit keywords at the call: random_state=SKLEARN_SEED for sklearn, "
        "seed=SKLEARN_SEED for datasets.Dataset.train_test_split, and "
        "seed=NUMPY_SEED for np.random.default_rng. If **kwargs cannot be "
        "verified, move the seed out of the parameter dictionary and pass it "
        "explicitly in the same edit; do not remove the seed or pass it twice."
    ),
    "pandas_positional_indexing_error": (
        "Use the traceback to identify the failing pandas object and indexer. "
        ".iloc selects zero-based positions within that object's current length; "
        ".loc selects index labels. Filtering, slicing, and dropna() can preserve "
        "old row labels, so another frame's .index is not automatically a valid "
        "positional indexer. If selecting shared sample labels, verify they exist "
        "in the target index and use .loc in the intended order. If positions are "
        "intended, compute valid positions for the current object; do not blindly "
        "replace every .iloc with .loc or clip/drop invalid indices. Align features "
        "and targets by sample identity, using one common validity mask or a joint "
        "table before dropping missing rows. Equal lengths alone do not prove "
        "alignment. Do not reset the index on only one side. Check duplicate labels "
        "and verify row identities and ordering after selection. Keep comparison "
        "variants on the same evaluation samples and preserve chronological "
        "splits and causal lag definitions when working with time series."
    ),
    "indexing_error": (
        "Inspect the authoritative schema and replace brittle positional or string "
        "splitting assumptions."
    ),
    "code_execution_error": (
        "Use the traceback to make a substantive correction before retrying."
    ),
    "timeout": "Reduce computation or data size; execution exceeded its time limit.",
    "execution_infrastructure_error": (
        "Execution did not return a reliable result. Inspect the environment or "
        "worker failure; this does not establish a Python syntax error."
    ),
}


# 这是什么问题？
def classify_runtime_data_failure(error=None, traceback_text="", *, error_type=None):
    """Use executor status first, then diagnostic text; return None if unknown."""
    # 执行器已确认的状态优先于错误文字，两个阶段使用相同规则。
    if error_type in {
        "process_exit", "missing_result", "invalid_result", "result_transport_error",
        "execution_environment_error", "worker_setup_error", "result_collection_error",
    }:
        return "execution_infrastructure_error"
    if error_type == "timeout":
        return "timeout"

    text = "\n".join(str(value or "") for value in (error, traceback_text)).casefold()  # 统一写成小写
    for category, groups in _RUNTIME_DATA_FAILURE_PATTERNS:
        if all(
            any(fragment in text for fragment in group)
            for group in groups
        ):
            return category
    return None

# 对于这个问题，给模型什么建议？
def runtime_data_repair_hint(category):
    # 接收一个已经确定的错误类别。用类别作为键，从字典中取建议。
    return _RUNTIME_DATA_REPAIR_HINTS.get(category)

# 写代码前，通用指导提前告诉模型，模型在生成代码时会根据这个指导。
_SYNTAX_CONTEXT_RADIUS = 2  # 表示展示出错行前面最多两行、后面最多两行。
_SYNTAX_CONTEXT_LINE_LIMIT = 180    # 表示每行源码最多保留前 180 个字符，避免一行特别长，把错误提示撑得太大。

# 把出错位置附近的代码整理成容易阅读的文字。
def _syntax_error_context(source, error):
    """
    source:完整 Python 源码字符串
    error:Python 解析代码时产生的语法错误对象（SyntaxError）
    error.lineno  # 错误发生在第几行
    error.offset  # 错误发生在这一行的第几列
    error.msg     # 错误原因，例如 expected ':'
    """
    line_number = getattr(error, "lineno", None)    # getattr() 用来读取对象的属性。
    if not isinstance(line_number, int) or line_number < 1:
        return ""   # 无法根据这个行号整理附近代码。这里只是没有代码片段，不代表语法错误不存在。

    lines = str(source).splitlines()    # 先把 source 转为字符串，再按照换行拆成列表。
    if line_number > len(lines):    # 如果错误对象说“第 10 行出错”，但源码只有 7 行，就无法按这个位置取代码，因此返回空字符串。
        return ""

    error_index = line_number - 1   # 源码行号从 1 开始，Python 列表下标从 0 开始：
    start = max(0, error_index - _SYNTAX_CONTEXT_RADIUS)    # 计算前后两行的范围
    end = min(len(lines), error_index + _SYNTAX_CONTEXT_RADIUS + 1)
    rendered = []   # 创建一个空列表，用来存放最终显示的每一行文字。
    offset = getattr(error, "offset", None) # 从错误对象中读取列号。

    for index in range(start, end): # 在错误位置下面添加 ^
        text = lines[index].expandtabs(4)
        if len(text) > _SYNTAX_CONTEXT_LINE_LIMIT:
            text = text[:_SYNTAX_CONTEXT_LINE_LIMIT] + "..."
        marker = "->" if index == error_index else "  "
        prefix = f"{marker} {index + 1}: "
        rendered.append(prefix + text)
        if index == error_index and isinstance(offset, int) and offset > 0:
            caret_column = min(offset - 1, len(text))
            rendered.append(" " * len(prefix) + " " * caret_column + "^")

    return "\n".join(rendered)


# 检查代码有没有语法错误。
def python_syntax_error(code):
    """Return a syntax diagnostic, or None when AST parsing succeeds."""
    source = str(code)
    try:
        ast.parse(source)   # 如果遇到缺少冒号、括号没有闭合、缩进不合法等问题，ast.parse() 会抛出 SyntaxError。
    except SyntaxError as exc:
        message = f"Generated code is invalid Python: {exc}"
        context = _syntax_error_context(source, exc)
        if context:
            message += "\nSyntax source context:\n" + context
        return message + "\nCorrect the reported syntax error before retrying."
    return None
