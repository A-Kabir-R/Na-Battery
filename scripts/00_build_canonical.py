"""Build canonical Standard-cycling tables directly from raw IRD files."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.io.canonical import build_canonical_tables


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--write-samples", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    result = build_canonical_tables(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        limit_files=args.limit_files,
        write_samples=args.write_samples,
        checksum_files=not args.skip_checksums,
    )
    print(
        "[canonical] "
        f"files={len(result.manifest)} steps={len(result.steps)} "
        f"cycles={len(result.cycles)} rpt={len(result.rpt_measurements)} "
        f"qc_flags={len(result.qc_flags)}"
    )


if __name__ == "__main__":
    main()
