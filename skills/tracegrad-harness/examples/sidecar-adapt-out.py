#!/usr/bin/env python3
"""Sidecar adapt-out: copy the applied Tracegrad template to the user path.

Run only after `tracegrad apply` has written the manifest template.
Copy this file next to the user repo. Do not move it into src/tracegrad/.

If --from and --to are the same path, this is a no-op. That is the usual
case when the app already loads the manifest `template_file`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="source",
        required=True,
        help="applied template (manifest template_file, resolved)",
    )
    parser.add_argument(
        "--to",
        dest="destination",
        required=True,
        help="path the user pipeline actually loads",
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    destination = Path(args.destination)
    if not source.is_file():
        print(f"tracegrad sidecar: applied template not found: {source}", file=sys.stderr)
        print("apply first; this adapter does not write the prompt itself", file=sys.stderr)
        return 1
    if source.resolve() == destination.resolve():
        print(f"adapt-out no-op: {source} is already the user path", file=sys.stderr)
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(f"copied {source} -> {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
