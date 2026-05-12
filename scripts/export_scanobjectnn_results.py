#!/usr/bin/env python
"""Preset exporter for the ScanObjectNN W&B sweep."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("export_wandb_results.py")
    cmd = [
        sys.executable,
        str(script),
        "--entity",
        "platonic-transformers-public",
        "--project",
        "Platonic-ScanObjectNN-CamReady",
        "--group-by",
        "model.solid_name",
        "model.attention",
        "--include-config",
        "seed",
        "--metric-prefix",
        "test",
        "--out-dir",
        "results/scanobjectnn",
        "--basename",
        "scanobjectnn_sweep",
    ]
    cmd.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
