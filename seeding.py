import ast
import contextlib
import contextvars
import os
import random
from dataclasses import dataclass


MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class SeedConfig:
    strategy: str = "fixed"
    python: int = 2001
    numpy: int = 3001
    sklearn: int = 4001
    pytorch: int = 5001
    pytorch_deterministic: bool = True

    @classmethod
    def from_mapping(cls, payload=None):
        if isinstance(payload, cls):
            return payload
        payload = dict(payload or {})
        supported = {
            "strategy",
            "python",
            "numpy",
            "sklearn",
            "pytorch",
            "pytorch-deterministic",
        }
        unexpected = sorted(set(payload) - supported)
        if unexpected:
            raise ValueError(f"Unsupported seed fields: {', '.join(unexpected)}")
        strategy = payload.get("strategy", "fixed")
        if strategy != "fixed":
            raise ValueError("seeds.strategy must be 'fixed'")

        def seed_value(name, default):
            value = payload.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"seeds.{name} must be an integer")
            if not 0 <= value <= MAX_SEED:
                raise ValueError(f"seeds.{name} must be between 0 and {MAX_SEED}")
            return value

        deterministic = payload.get("pytorch-deterministic", True)
        if not isinstance(deterministic, bool):
            raise ValueError("seeds.pytorch-deterministic must be true or false")
        defaults = cls()
        return cls(
            strategy=strategy,
            python=seed_value("python", defaults.python),
            numpy=seed_value("numpy", defaults.numpy),
            sklearn=seed_value("sklearn", defaults.sklearn),
            pytorch=seed_value("pytorch", defaults.pytorch),
            pytorch_deterministic=deterministic,
        )

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "python": self.python,
            "numpy": self.numpy,
            "sklearn": self.sklearn,
            "pytorch": self.pytorch,
            "pytorch-deterministic": self.pytorch_deterministic,
        }


_seed_config_var = contextvars.ContextVar(
    "agentlab_seed_config",
    default=SeedConfig(),
)


@contextlib.contextmanager
def seed_scope(config):
    resolved = SeedConfig.from_mapping(config)
    token = _seed_config_var.set(resolved)
    try:
        yield resolved
    finally:
        _seed_config_var.reset(token)


def current_seed_config():
    return _seed_config_var.get()


def seed_prompt_text(config=None):
    seeds = SeedConfig.from_mapping(config or current_seed_config())
    return (
        "The execution environment provides fixed constants "
        f"PYTHON_SEED={seeds.python}, NUMPY_SEED={seeds.numpy}, "
        f"SKLEARN_SEED={seeds.sklearn}, and PYTORCH_SEED={seeds.pytorch}. "
        "Use SKLEARN_SEED for every scikit-learn random_state argument and for "
        "datasets.Dataset.train_test_split(seed=...). Prefer explicit seed keywords, "
        "for example LogisticRegression(random_state=SKLEARN_SEED, ...). "
        "Simple, unchanged parameter dictionaries can also be checked statically. "
        "If **kwargs cannot be verified, move the seed out of the dictionary and "
        "pass it explicitly at the call; do not remove the seed or pass it twice. "
        "Pass NUMPY_SEED to "
        "np.random.default_rng. Do not replace these constants with "
        "other seed values. Python random, legacy NumPy random, and PyTorch are "
        "also initialized by the execution framework."
    )


def _imported_roots(tree):
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(item.name.split(".", 1)[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def execution_seed_globals(code, config=None):
    seeds = SeedConfig.from_mapping(config or current_seed_config())
    random.seed(seeds.python)

    import numpy as np

    np.random.seed(seeds.numpy)
    globals_dict = {
        "PYTHON_SEED": seeds.python,
        "NUMPY_SEED": seeds.numpy,
        "SKLEARN_SEED": seeds.sklearn,
        "PYTORCH_SEED": seeds.pytorch,
        "AGENTLAB_SEEDS": seeds.to_dict(),
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None and "torch" in _imported_roots(tree):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch

        torch.manual_seed(seeds.pytorch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds.pytorch)
        if seeds.pytorch_deterministic:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    return globals_dict


SKLEARN_RANDOM_CALLS = {
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "KMeans",
    "LogisticRegression",
    "MiniBatchKMeans",
    "MLPClassifier",
    "MLPRegressor",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RandomizedSearchCV",
    "SGDClassifier",
    "SGDRegressor",
    "SpectralClustering",
}


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _dotted_name(node):
    """Return a best-effort dotted name for a direct or attribute expression."""
    pieces = []
    while isinstance(node, ast.Attribute):
        pieces.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    pieces.append(node.id)
    return ".".join(reversed(pieces))


def _sklearn_train_test_split_bindings(tree):
    """Find direct and module aliases for sklearn.model_selection."""
    direct_names = {"train_test_split"}
    module_aliases = {"model_selection"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "sklearn.model_selection" and item.asname:
                    module_aliases.add(item.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "sklearn.model_selection":
                for item in node.names:
                    if item.name == "train_test_split":
                        direct_names.add(item.asname or item.name)
            elif node.module == "sklearn":
                for item in node.names:
                    if item.name == "model_selection":
                        module_aliases.add(item.asname or item.name)
    return direct_names, module_aliases


def _is_sklearn_train_test_split_call(node, direct_names, module_aliases):
    if isinstance(node.func, ast.Name):
        return node.func.id in direct_names
    path = _dotted_name(node.func)
    if path is None:
        return False
    if path.endswith(".model_selection.train_test_split"):
        return True
    return any(path == f"{alias}.train_test_split" for alias in module_aliases)


def _matches_seed_expression(node, constant_name, expected_value, mapping_key):
    if isinstance(node, ast.Name):
        return node.id == constant_name
    if isinstance(node, ast.Constant):
        return node.value == expected_value
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id != "AGENTLAB_SEEDS":
            return False
        slice_node = node.slice
        return isinstance(slice_node, ast.Constant) and slice_node.value == mapping_key
    return False


class _SeedKeywordResolver:
    """Resolve literal mappings and unmodified local/outer parameter dictionaries.

    This deliberately does not evaluate code or infer dynamic dictionary updates.
    A named mapping needs one unconditional assignment, and every read must be an
    unpack (which copies it). Other uses may mutate or expose an alias to it.
    """

    _scopes = (
        ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
        ast.ClassDef, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    )

    def __init__(self, tree):
        self.parents = {
            child: parent for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        self.bindings = {}
        self.unsafe_names = set()
        self.unsafe_bindings = set()
        self.has_wildcard_import = False
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names = [node.id]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.alias):
                names = [node.asname or node.name.split(".")[0]]
                if node.name == "*":
                    self.has_wildcard_import = True
                    self.unsafe_names.add("dict")
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names = [node.name]
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                self.unsafe_names.update(node.names)
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                self.unsafe_names.add(node.target.id)
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                names = [node.name]
            elif isinstance(node, ast.MatchMapping) and node.rest:
                names = [node.rest]
            for name in names:
                scope = next(self._scope_chain(self.parents.get(node)), None)
                self.bindings.setdefault((scope, name), []).append(node)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            parent = self.parents.get(node)
            unpack = isinstance(parent, ast.keyword) and parent.arg is None
            if isinstance(parent, ast.Dict):
                unpack = any(
                    key is None and value is node
                    for key, value in zip(parent.keys, parent.values)
                )
            if not unpack:
                self.unsafe_bindings.add(self._binding_key(node, node.id))

    def _scope_chain(self, node):
        first = True
        while node is not None:
            if isinstance(node, self._scopes):
                # A method cannot resolve bare names through its class namespace.
                if first or not isinstance(node, ast.ClassDef):
                    yield node
                first = False
            node = self.parents.get(node)

    def _binding_key(self, node, name):
        for scope in self._scope_chain(node):
            key = (scope, name)
            if key in self.bindings:
                return key
        return None

    def mapping(self, node, resolving=frozenset()):
        if isinstance(node, ast.Name):
            key = self._binding_key(node, node.id)
            bindings = self.bindings.get(key, [])
            if (self.has_wildcard_import
                    or node.id in self.unsafe_names or key in self.unsafe_bindings
                    or key in resolving or len(bindings) != 1):
                return None
            target = bindings[0]
            assignment = self.parents.get(target)
            if not isinstance(target, ast.Name):
                return None
            if isinstance(assignment, ast.Assign):
                if len(assignment.targets) != 1 or assignment.targets[0] is not target:
                    return None
            elif not isinstance(assignment, ast.AnnAssign):
                return None
            scope = key[0]
            if (not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef))
                    or self.parents.get(assignment) is not scope):
                return None
            if next(self._scope_chain(node), None) is scope:
                if (assignment.lineno, assignment.col_offset) >= (node.lineno, node.col_offset):
                    return None
            return self.mapping(assignment.value, resolving | {key})

        if isinstance(node, ast.Dict):
            entries = zip(node.keys, node.values)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "dict" and not node.args
                and "dict" not in self.unsafe_names
                and self._binding_key(node, "dict") is None):
            entries = (
                (ast.Constant(item.arg) if item.arg is not None else None, item.value)
                for item in node.keywords
            )
        else:
            return None

        result = {}
        for key, value in entries:
            if key is None:
                expanded = self.mapping(value, resolving)
                if expanded is None:
                    return None
                result.update(expanded)
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                result[key.value] = value
            else:
                return None
        return result

    def keywords(self, node):
        explicit = {item.arg: item.value for item in node.keywords if item.arg}
        result = dict(explicit)
        unresolved = False
        duplicates = set()
        for item in node.keywords:
            if item.arg is not None:
                continue
            expanded = self.mapping(item.value)
            if expanded is None:
                unresolved = True
                continue
            duplicates.update(result.keys() & expanded.keys())
            result.update(expanded)
        # An opaque expansion cannot prove a seed supplied by another expansion.
        return (explicit if unresolved else result), unresolved, duplicates


def validate_seed_usage(tree, config=None):
    seeds = SeedConfig.from_mapping(config or current_seed_config())
    violations = []
    sklearn_split_names, sklearn_module_aliases = _sklearn_train_test_split_bindings(
        tree
    )
    resolver = _SeedKeywordResolver(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        keywords, unresolved, duplicates = resolver.keywords(node)
        random_state = keywords.get("random_state")
        is_sklearn_split = _is_sklearn_train_test_split_call(
            node, sklearn_split_names, sklearn_module_aliases
        )
        seed_keyword = None
        if name in SKLEARN_RANDOM_CALLS or is_sklearn_split:
            seed_keyword = "random_state=SKLEARN_SEED"
        elif name == "train_test_split":
            seed_keyword = "seed=SKLEARN_SEED"
        elif name == "default_rng":
            seed_keyword = "seed=NUMPY_SEED"
        if seed_keyword:
            argument = seed_keyword.split("=", 1)[0]
            if duplicates:
                violations.append(
                    f"{name}() repeats keyword arguments: {', '.join(sorted(duplicates))}; "
                    f"pass {seed_keyword} exactly once"
                )
                continue
            if (unresolved and argument not in keywords
                    and not (name == "default_rng" and node.args)):
                violations.append(
                    f"{name}() cannot statically verify the seed in **kwargs; "
                    f"pass {seed_keyword} explicitly at the call and remove that "
                    "key from the expanded dictionary to avoid duplicate arguments"
                )
                continue
        if name == "train_test_split" or is_sklearn_split:
            if is_sklearn_split:
                if random_state is None:
                    violations.append(
                        "train_test_split() is missing random_state=SKLEARN_SEED"
                    )
                elif not _matches_seed_expression(
                    random_state,
                    "SKLEARN_SEED",
                    seeds.sklearn,
                    "sklearn",
                ):
                    violations.append(
                        "train_test_split() must use random_state=SKLEARN_SEED"
                    )
            else:
                seed = keywords.get("seed")
                if random_state is not None:
                    violations.append(
                        "datasets.Dataset.train_test_split() must use "
                        "seed=SKLEARN_SEED, not random_state"
                    )
                elif seed is None:
                    violations.append(
                        "datasets.Dataset.train_test_split() is missing "
                        "seed=SKLEARN_SEED"
                    )
                elif not _matches_seed_expression(
                    seed,
                    "SKLEARN_SEED",
                    seeds.sklearn,
                    "sklearn",
                ):
                    violations.append(
                        "datasets.Dataset.train_test_split() must use "
                        "seed=SKLEARN_SEED"
                    )
            continue
        if name in SKLEARN_RANDOM_CALLS and random_state is None:
            violations.append(f"{name}() is missing random_state=SKLEARN_SEED")
        elif random_state is not None and not _matches_seed_expression(
            random_state,
            "SKLEARN_SEED",
            seeds.sklearn,
            "sklearn",
        ):
            violations.append(
                f"{name or 'call'}() must use random_state=SKLEARN_SEED"
            )
        if name == "default_rng":
            seed_argument = node.args[0] if node.args else keywords.get("seed")
            if seed_argument is None or not _matches_seed_expression(
                seed_argument,
                "NUMPY_SEED",
                seeds.numpy,
                "numpy",
            ):
                violations.append("default_rng() must receive NUMPY_SEED")
    if violations:
        raise ValueError("Seed policy violations: " + "; ".join(sorted(set(violations))))
