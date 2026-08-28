"""Progress reporting shared by the optimizer and command-line interface."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

ProgressCallback = Callable[[int, str], None]


class ProgressBar:
    def __init__(self, stream: TextIO | None = None, width: int = 24):
        self.stream = stream or sys.stderr
        self.width = width
        self._last: tuple[int, str] | None = None
        self._open = False
        self._line_length = 0

    def update(self, percent: int, message: str) -> None:
        percent = max(0, min(100, percent))
        state = (percent, message)
        if state == self._last:
            return
        filled = round(self.width * percent / 100)
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"[{bar}] {percent:3}% {message}"
        if self.stream.isatty():
            self.stream.write(f"\r{line.ljust(self._line_length)}")
            self._line_length = len(line)
        else:
            self.stream.write(f"{line}\n")
        self.stream.flush()
        self._last = state
        self._open = True

    def close(self) -> None:
        if self._open and self.stream.isatty():
            self.stream.write("\n")
            self.stream.flush()
        self._open = False