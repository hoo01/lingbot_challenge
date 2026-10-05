"""Periodic diagnostics with the GIL held while inspecting Python frames.

CPython 3.10.22 on the rented host segfaults in a CPU-only code-object churn
probe with dump_traceback_later. Its native watchdog inspects live frames
without owning the GIL. Calling dump_traceback from a Python thread avoids
that path. Hard stall/episode deadlines remain in the independent supervisor.
"""
import faulthandler
import math
import sys
import threading


class PeriodicTraceback:
    def __init__(self, interval=120, file=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Traceback interval must be finite and positive')
        self.interval = interval
        self.file = sys.stderr if file is None else file
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is not None:
            raise RuntimeError('Traceback reporter already started')
        self.thread = threading.Thread(target=self._run, name='eval-traceback', daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.wait(self.interval):
            # This Python thread owns the GIL on entry. Do not replace this
            # with faulthandler.dump_traceback_later or an async signal handler.
            faulthandler.dump_traceback(file=self.file, all_threads=True)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError('Traceback reporter failed to stop')
