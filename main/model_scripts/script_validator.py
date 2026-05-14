"""
script_validator.py
-------------------
Validates a user-supplied custom model script (.py file) against the
ModelScript interface defined in base.py, without executing train_model().

Public API
----------
    validate_custom_script(file_path, problem_type) -> tuple[bool, str]

Returns (True, "ok") when the file is safe to pass to the Orchestrator, or
(False, human-readable reason) when it should be rejected.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


# Names that the built-in scanner already skips; custom scripts must not
# accidentally shadow them.
_RESERVED_NAMES = {"__init__", "base", "utils", "script_validator", "example_custom_model"}

# Maximum file size accepted (100 KB).  Large files are almost certainly not
# model scripts and could be a resource-exhaustion attempt.
_MAX_BYTES = 100 * 1024


def validate_custom_script(
    file_path: str | Path,
    problem_type: str,
) -> tuple[bool, str]:
    """Validate a single custom model script.

    Checks (in order):
        1. File exists and is a .py file within the size limit.
        2. The module can be imported without errors.
        3. The module satisfies the structural contract enforced by
           validate_module() in base.py (MODEL_NAME, SUPPORTED_PROBLEM_TYPES,
           Model class with train_model method).
        4. problem_type appears in module.SUPPORTED_PROBLEM_TYPES.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the .py file to validate.
    problem_type:
        The problem type selected by the user, e.g. "classification",
        "regression", or "kmeans_clustering".

    Returns
    -------
    (True, "ok") on success.
    (False, reason_string) on any failure.  reason_string is a short
    human-readable message suitable for display in the UI.
    """
    file_path = Path(file_path)

    # ── 1. File-level checks ──────────────────────────────────────────────────
    if not file_path.exists():
        return False, f"file not found: {file_path.name}"

    if file_path.suffix.lower() != ".py":
        return False, f"not a Python file (got '{file_path.suffix}')"

    try:
        size = file_path.stat().st_size
    except OSError as exc:
        return False, f"cannot stat file: {exc}"

    if size > _MAX_BYTES:
        return False, (
            f"file too large ({size:,} bytes; limit is {_MAX_BYTES:,} bytes)"
        )

    if file_path.stem in _RESERVED_NAMES:
        return False, (
            f"'{file_path.stem}' is a reserved module name; "
            "please rename your script"
        )

    # ── 2. Import the module into an isolated namespace ───────────────────────
    # Use a unique module name to avoid colliding with anything already in
    # sys.modules (e.g. if the user uploads two scripts with the same stem).
    unique_mod_name = f"_custom_script_{file_path.stem}_{id(file_path)}"

    try:
        spec = importlib.util.spec_from_file_location(unique_mod_name, file_path)
        if spec is None or spec.loader is None:
            return False, "Python could not build a module spec for this file"

        module = importlib.util.module_from_spec(spec)
        # Register temporarily so relative imports inside the script (if any)
        # can resolve; we clean up afterwards.
        sys.modules[unique_mod_name] = module
        spec.loader.exec_module(module)   # type: ignore[union-attr]
    except SyntaxError as exc:
        _cleanup(unique_mod_name)
        return False, f"syntax error: {exc}"
    except Exception as exc:
        _cleanup(unique_mod_name)
        return False, f"import error: {exc}"

    # ── 3. Structural validation via base.validate_module() ───────────────────
    try:
        from main.model_scripts.base import validate_module  # type: ignore
        ok, reason = validate_module(module)
    except ImportError:
        # Fallback: run the same checks inline if the package import fails
        # (e.g. during testing outside the main package tree).
        ok, reason = _inline_validate(module)
    except Exception as exc:
        _cleanup(unique_mod_name)
        return False, f"validation machinery error: {exc}"

    if not ok:
        _cleanup(unique_mod_name)
        return False, reason

    # ── 4. Problem-type compatibility ─────────────────────────────────────────
    supported = getattr(module, "SUPPORTED_PROBLEM_TYPES", ())
    try:
        supported_lower = [s.lower() for s in supported]
    except Exception:
        _cleanup(unique_mod_name)
        return False, "SUPPORTED_PROBLEM_TYPES is not iterable"

    if problem_type.lower() not in supported_lower:
        _cleanup(unique_mod_name)
        supported_str = ", ".join(supported) if supported else "(none)"
        return False, (
            f"script does not support problem type '{problem_type}'; "
            f"it declares: {supported_str}"
        )

    # Leave the module in sys.modules so the Orchestrator can import it by
    # the same unique name if needed, but do not keep a hard reference here.
    # The Orchestrator will load it fresh via its own importlib call.
    _cleanup(unique_mod_name)
    return True, "ok"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cleanup(module_name: str) -> None:
    """Remove a temporarily registered module from sys.modules."""
    sys.modules.pop(module_name, None)


def _inline_validate(mod: types.ModuleType) -> tuple[bool, str]:
    """
    Minimal structural check used as a fallback when base.validate_module()
    cannot be imported (e.g. running tests outside the package tree).

    Mirrors the logic in base.validate_module().
    """
    if not hasattr(mod, "MODEL_NAME"):
        return False, "missing MODEL_NAME"
    if not hasattr(mod, "SUPPORTED_PROBLEM_TYPES"):
        return False, "missing SUPPORTED_PROBLEM_TYPES"
    if not hasattr(mod, "Model"):
        return False, "missing Model class"

    ModelClass = getattr(mod, "Model")
    if not isinstance(ModelClass, type):
        return False, "Model is not a class"
    if not hasattr(ModelClass, "train_model"):
        return False, "Model class has no train_model method"

    return True, "ok"