import os
from enum import Enum
from typing import Tuple, Optional


class PrintTarget(Enum):
    NONE = 0
    CONSOLE = 1
    FILE = 2


class PrintWriter:
    # This is basically a buffered writer
    def __init__(self, target: Tuple[PrintTarget, Optional[str]], del_file: bool = True) -> None:
        self._target = target[0]
        if self._target is PrintTarget.FILE:
            assert target[1] is not None, "PrintTarget 'File' requires non-None file path."
            self._file_path = target[1]
            if os.path.exists(self._file_path) and del_file:
                os.remove(self._file_path)
        self._buffered_lines = []

    def write_line(self, line: str) -> None:
        tmp = self._buffered_lines[:]
        self._buffered_lines = [line]
        self.flush()
        self._buffered_lines = tmp[:]

    def buffer_line(self, line: str) -> None:
        if self._target is PrintTarget.NONE:
            return
        self._buffered_lines.append(line)

    def clear(self) -> None:
        self._buffered_lines = []

    def flush(self) -> None:
        if self._target is PrintTarget.CONSOLE:
            for l in self._buffered_lines:
                print(l)
        else:
            with open(self._file_path, "a") as f:
                f.writelines(map(lambda x: x + "\n", self._buffered_lines))
        self.clear()
