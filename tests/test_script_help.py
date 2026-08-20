"""Every script's --help must render.

argparse runs each help string through ``% params`` when it formats -h, so a
bare '%' in prose becomes a format spec and blows up the whole help screen --
'~0.1% of a bin' reads as space-flag octal.  Nothing else in a run touches that
code path, so the break only shows up when a user asks for help, which is
exactly when they are least able to work around it.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

_SCRIPTS = sorted((Path(__file__).resolve().parent.parent / "scripts")
                  .glob("*.py"))
assert _SCRIPTS, "no scripts found"


def _load(path):
    """Exec the script without running it, and hand back its module."""
    src = path.read_text().replace('if __name__ == "__main__":', "if False:")
    mod = type(sys)("_script_under_test")
    mod.__dict__["__name__"] = "_script_under_test"
    mod.__dict__["__file__"] = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("path", _SCRIPTS, ids=lambda p: p.name)
def test_help_renders(path):
    try:
        mod = _load(path)
    except Exception as exc:                     # a missing optional backend
        pytest.skip(f"{path.name} does not import here: {exc}")
    if not hasattr(mod, "main"):
        pytest.skip(f"{path.name} has no main()")
    if "argparse" not in path.read_text():
        pytest.skip(f"{path.name} takes bare sys.argv, so there is no -h")

    argv = sys.argv
    sys.argv = [str(path), "--help"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            try:
                mod.main(["--help"])
            except TypeError as exc:             # main() reads sys.argv itself
                if "positional argument" not in str(exc):
                    raise
                mod.main()
    except SystemExit:
        pass                                     # -h always exits 0
    finally:
        sys.argv = argv

    assert "usage" in out.getvalue().lower(), f"{path.name} printed no usage"
