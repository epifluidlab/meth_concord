"""Helpers for the reproduction notebooks.

Every step of the notebooks is one command.  ``run`` prints it, skips it when
its outputs are already there (so a notebook can be re-run, or resumed after an
interruption, without redoing hours of work), and otherwise runs it, streaming
its output into the notebook and into a log file.  ``show`` draws a PDF
figure inline.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


def done(outputs) -> bool:
    """True when every output exists: a file, or a directory holding at least
    one file."""
    for p in map(Path, outputs):
        if p.is_dir():
            if not any(p.iterdir()):
                return False
        elif not p.exists():
            return False
    return True


def run(cmd, outputs, inputs=(), log=None, dry_run=False, force=False,
        cwd=None):
    """Run ``cmd`` (a list of arguments) unless all of ``outputs`` exist.

    ``inputs`` are checked only when the step is to run, so a finished step
    is skipped even after its inputs are gone.  ``dry_run`` prints the command
    only; ``force`` runs it even when its outputs exist.  Raises
    ``subprocess.CalledProcessError`` on failure, so a notebook stops at the
    step that failed.
    """
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not force and outputs and done(outputs):
        print("  -> outputs present; skipped", flush=True)
        return
    if dry_run:
        miss = [str(p) for p in inputs if not Path(p).exists()]
        print("  -> dry run; not executed"
              + (f" (missing inputs: {', '.join(miss)})" if miss else ""),
              flush=True)
        return
    need(*inputs)
    t0 = time.time()
    fh = None
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "w")
        fh.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            cwd=cwd)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            if fh:
                fh.write(line)
    finally:
        proc.wait()
        if fh:
            fh.close()
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"  -> done in {(time.time() - t0) / 60:.1f} min", flush=True)


def need(*paths):
    """Stop with a clear message when an input is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("missing input(s):\n  " + "\n  ".join(missing))


def show(pdf, dpi=110):
    """Draw the first page of ``pdf`` inline (with pdftoppm), else link it."""
    from IPython.display import Image, Markdown, display
    pdf = Path(pdf)
    if not pdf.exists():
        display(Markdown(f"*not yet made:* `{pdf}`"))
        return
    if shutil.which("pdftoppm"):
        png = pdf.with_suffix(".preview.png")
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                        str(pdf), str(png.with_suffix(""))], check=True)
        display(Image(filename=str(png)))
    else:
        display(Markdown(f"[{pdf.name}]({pdf})  (install poppler's "
                         f"`pdftoppm` to see it here)"))


def compare(rows):
    """The checks against the manuscript at the end of each notebook: rows of
    (what, the manuscript's value as printed there, this run's value).  This
    run's value is rounded to as many decimals as the manuscript prints, and
    'matches' says whether the two then agree."""
    import pandas as pd
    out = []
    for what, printed, got in rows:
        printed = str(printed)
        dec = len(printed.split(".")[1]) if "." in printed else 0
        mine = round(float(got), dec)
        out.append((what, printed, f"{mine:.{dec}f}",
                    mine == round(float(printed), dec)))
    return pd.DataFrame(out, columns=["value", "manuscript", "this run",
                                      "matches"])
