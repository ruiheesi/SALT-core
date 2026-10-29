#!/usr/bin/env python3
"""Rebind the pinned Gemma expert-pool receipt to its current absolute path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import tempfile


POOL_BYTES = 12_846_366_744
POOL_SHA256 = "b06c91ecfae36581cbafdda80e116446bb32ea5ec832066e86cb672421a66e31"
COPY_MAP_SHA256 = "51fa40aca8cb910896328026fcfb3f07c36a7519d0bd76e6fce574ba3c178a3e"
SOURCE = {
    "authenticated_bytes": 15_373_375_402,
    "contract_sha256": "312e73b836f39d06f5c982f4b5a83575d7c4bb23539b787293a247cd2bc71949",
    "repository": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "revision": "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
}
ROOT_KEYS = {
    "copy_map_sha256", "copy_range_count", "pool", "runtime_blocker",
    "runtime_ready", "schema", "source", "status", "transaction_id",
}


def fail(message: str) -> None:
    raise SystemExit(f"gemma4 receipt rebase: {message}")


def regular_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        fail(f"cannot inspect {label} {path}: {exc}")
    if not stat.S_ISREG(info.st_mode):
        fail(f"{label} is not a regular file: {path}")
    return info


def identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    pool = args.pool.expanduser().resolve(strict=True)
    pool_info = regular_file(pool, "expert pool")
    if pool_info.st_size != POOL_BYTES:
        fail(f"expert pool extent {pool_info.st_size} != {POOL_BYTES}")

    receipt = args.receipt.expanduser()
    parent = receipt.parent.resolve(strict=True)
    target = parent / receipt.name
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        receipt_fd = os.open(target, flags)
    except OSError as exc:
        fail(f"cannot open import receipt {target}: {exc}")
    try:
        receipt_info = os.fstat(receipt_fd)
        if (not stat.S_ISREG(receipt_info.st_mode) or
                receipt_info.st_uid != os.geteuid() or
                receipt_info.st_size < 2 or receipt_info.st_size > 1 << 20):
            fail("import receipt file identity is unsafe")
        raw_receipt = os.read(receipt_fd, receipt_info.st_size + 1)
        if (len(raw_receipt) != receipt_info.st_size or
                identity(os.fstat(receipt_fd)) != identity(receipt_info)):
            fail("import receipt changed while reading")
    finally:
        os.close(receipt_fd)
    try:
        value = json.loads(raw_receipt)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read import receipt: {exc}")
    if not isinstance(value, dict) or set(value) != ROOT_KEYS:
        fail("import receipt key set drift")
    if (value["schema"] != "salt.gemma4-mlx4-import-receipt.v1" or
            value["status"] != "expert-pool-materialized-runtime-blocked" or
            value["runtime_ready"] is not False or
            value["runtime_blocker"] != "blocked-until-gemma-geglu-runtime-exists" or
            value["copy_map_sha256"] != COPY_MAP_SHA256 or
            value["copy_range_count"] != 34_560 or
            value["source"] != SOURCE):
        fail("import receipt authority drift")
    transaction_id = value["transaction_id"]
    if (not isinstance(transaction_id, str) or
            re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None):
        fail("import receipt transaction identity drift")
    pool_value = value["pool"]
    if (not isinstance(pool_value, dict) or
            set(pool_value) != {"bytes", "path", "sha256"} or
            pool_value["bytes"] != POOL_BYTES or
            pool_value["sha256"] != POOL_SHA256 or
            not isinstance(pool_value["path"], str)):
        fail("import receipt pool identity drift")

    current = str(pool)
    if pool_value["path"] == current:
        print(f"gemma4 receipt rebase: already current: {current}")
        return 0
    pool_value["path"] = current
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent,
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if identity(regular_file(target, "import receipt")) != identity(receipt_info):
            fail("import receipt changed before path publication")
        os.replace(temporary, target)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    print(f"gemma4 receipt rebase: established path {current}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
