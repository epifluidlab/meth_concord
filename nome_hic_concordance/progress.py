"""Progress reporting shared by the long-running distance scripts.

Lives in the package rather than in one script because two of them need
the identical behaviour, and a copy in each is a copy that drifts.
"""

from __future__ import annotations

import sys

__all__ = ["Progress"]


class Progress:
    """Count, rate and ETA on stderr, for a job whose length is known.

    ONE LINE IN PLACE ON A TERMINAL, A PERIODIC LINE OTHERWISE.  These runs
    are usually launched on a cluster with stderr redirected to a file, and
    a carriage-return bar there writes one line per update -- thousands of
    them, in a log someone has to read afterwards.  Checking isatty costs
    nothing and is the difference between a usable log and a useless one.

    The ETA is elapsed/done x remaining, which is honest only while the
    items cost about the same.  They do not here: distance bins differ in
    how many pairs they hold and cells differ in depth, so the estimate
    moves.  It is reported as an estimate and the elapsed time beside it is
    the number that is not.
    """

    def __init__(self, total, label, stream=None, min_interval=2.0):
        import time
        self.total = max(1, int(total))
        self.label = label
        self.stream = stream or sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.min_interval = 0.0 if self.tty else float(min_interval)
        self.done = 0
        self._t0 = time.time()
        self._last = 0.0
        self._shown = -1

    def _fmt(self, sec):
        sec = int(max(0, sec))
        if sec < 60:
            return f"{sec}s"
        if sec < 3600:
            return f"{sec // 60}m{sec % 60:02d}s"
        return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"

    def update(self, n=1, force=False):
        import time
        self.done += int(n)
        now = time.time()
        if not force and (now - self._last) < self.min_interval \
                and self.done < self.total:
            return
        self._last = now
        self._shown = self.done
        el = now - self._t0
        frac = self.done / self.total
        eta = (el / self.done * (self.total - self.done)) if self.done else 0
        msg = (f"  {self.label}: {self.done:,}/{self.total:,} "
               f"({frac:5.1%})  elapsed {self._fmt(el)}  "
               f"eta {self._fmt(eta)}")
        if self.tty:
            self.stream.write("\r" + msg + " " * 8)
        else:
            self.stream.write(msg + "\n")
        self.stream.flush()

    def close(self):
        # Only if the last line drawn is not already the final one -- a
        # duplicate 100% line is noise in a log and a flicker on a terminal.
        if self._shown != self.done:
            self.update(0, force=True)
        if self.tty:
            self.stream.write("\n")
        self.stream.flush()
