#!/usr/bin/env python3
"""Prefix SALT log lines, or supervise a command into a timestamped log."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import subprocess
import sys
from typing import Iterable, TextIO


def _stamp(lines: Iterable[str], output: TextIO) -> None:
    for line in lines:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        output.write(f"[{timestamp}] {line}")
        output.flush()


def _supervise(command: list[str], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def forward(signum: int, _frame: object) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGTERM, forward)
        assert process.stdout is not None
        _stamp(process.stdout, output)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if args.output is None:
        if command:
            parser.error("--output is required when supervising a command")
        _stamp(sys.stdin, sys.stdout)
        return 0
    if not command:
        parser.error("a command is required after --")
    return _supervise(command, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
