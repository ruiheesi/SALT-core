#!/usr/bin/env python3
"""Prefix streamed SALT log lines with local ISO-8601 timestamps."""

from __future__ import annotations

from datetime import datetime
import sys


def main() -> int:
    for line in sys.stdin:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        sys.stdout.write(f"[{timestamp}] {line}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
