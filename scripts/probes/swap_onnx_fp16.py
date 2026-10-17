"""Road C: the swap's ONNX files converted to fp16 for TensorRT 11, by hand.

A thin command around :mod:`sha256.avatar.swap_models` (``write_fp16_onnx``):
the application does the same at its first start when a plan is missing.
Writes <name>-fp16.onnx beside each fp32 file in models/downloads/swap, the
nodes in models/metadata/swap_fp16_blocks.json (and their consumers) kept
fp32, Casts at every boundary, the graph's inputs and outputs float32.

Usage (from the project root):
  .venv\\Scripts\\python.exe -I scripts/probes/swap_onnx_fp16.py [--only inswapper] [--force]
      [--blocks models/metadata/swap_fp16_blocks.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    from sha256.avatar import swap_models as sm

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", action="append", choices=tuple(sm.ONNX_FILES))
    p.add_argument("--force", action="store_true")
    p.add_argument("--models", type=Path, default=PROJECT_ROOT / "models" / sm.SWAP_RELATIVE_DIR)
    p.add_argument(
        "--blocks",
        type=Path,
        default=PROJECT_ROOT / "models" / sm.SWAP_BLOCKS_RELATIVE_PATH,
        help="the range probe's flagged nodes per network (swap_fp16_range.py)",
    )
    args = p.parse_args(argv)
    flagged: dict = {}
    if args.blocks.is_file():
        flagged = json.loads(args.blocks.read_text(encoding="utf-8")).get("blocks", {})
    for net, relative in sm.ONNX_FILES.items():
        if args.only and net not in args.only:
            continue
        onnx_path = args.models / relative
        target = sm.fp16_onnx_path(onnx_path)
        if target.exists() and not args.force:
            print(f"{net}: {target.name} exists")
            continue
        meta = sm.write_fp16_onnx(onnx_path, target, flagged.get(net))
        print(
            f"{net}: {meta['size_bytes'] / 1e6:.0f} MB in {meta['seconds']} s;"
            f" {meta['nodes_fp16']} nodes fp16, {meta['nodes_fp32']} fp32"
            f" ({meta['blocked_nodes']} blocked by range), {meta['casts']} casts ({target.name})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
