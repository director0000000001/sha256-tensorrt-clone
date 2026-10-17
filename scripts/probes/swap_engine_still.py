"""Road C, C2: the swap ENGINE (sha256.avatar.engine_swap, the one the
application runs) on his captures, with the frame path's timings.

The engine is started as the application starts it (the plans built or
loaded, the avatar's identity read from its portrait), each capture's frame
and stored landmarks are fed to ``animate`` as the pipeline would, the
output frames are written to --out and the last frame's parts printed.

Usage (from the project root):
  .venv\\Scripts\\python.exe -I scripts/probes/swap_engine_still.py --out <dir>
      [--avatar bidzo_swap] [--capture captures/moment-...] [--repeat 10]
      [--option enhance=false]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--avatar", action="append", default=[], help="swap avatar id (repeatable); default both"
    )
    p.add_argument(
        "--capture", action="append", type=Path, default=[], help="captures/moment-* (repeatable)"
    )
    p.add_argument(
        "--repeat", type=int, default=10, help="animate calls per capture; the median is reported"
    )
    p.add_argument(
        "--option", action="append", default=[], metavar="KEY=VALUE", help="engine option override"
    )
    args = p.parse_args(argv)

    import cv2

    from sha256.avatar.engine_swap import SwapEngine
    from sha256.avatar.repository import FilesystemAvatarRepository
    from sha256.core.enums import PixelFormat
    from sha256.core.types import (
        FaceTrackingResult,
        FrameMetadata,
        HeadPose,
        LandmarkSet,
        VideoFrame,
    )

    models = PROJECT_ROOT / "models"
    engine = SwapEngine(
        model_path=models / "downloads" / "face_landmarker.task",
        models=models / "downloads" / "swap",
        engines=models / "tensorrt",
        blocks=models / "metadata" / "swap_fp16_blocks.json",
        progress=print,
    )
    started = time.perf_counter()
    engine.start()
    print(f"engine started in {time.perf_counter() - started:.1f} s")
    repository = FilesystemAvatarRepository(PROJECT_ROOT / "avatars")
    avatars = args.avatar or ["bidzo_swap", "my_avatar_swap"]
    captures = args.capture or sorted((PROJECT_ROOT / "captures").glob("moment-20261017-03*"))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for avatar_id in avatars:
        descriptor = repository.get(avatar_id)
        for item in args.option:
            key, _, raw = item.partition("=")
            try:
                descriptor.manifest.engine_options[key] = json.loads(raw)
            except json.JSONDecodeError:
                descriptor.manifest.engine_options[key] = raw
        started = time.perf_counter()
        engine.load_avatar(descriptor)
        vram = engine.estimated_vram_mb(avatar_id)
        print(f"{avatar_id} loaded in {time.perf_counter() - started:.2f} s (VRAM est. {vram} MB)")
        for folder in captures:
            z = np.load(folder / "tracking.npz", allow_pickle=True)
            frame = cv2.imread(str(folder / "camera.png"))
            meta = FrameMetadata(frame_id=0, capture_timestamp=0.0)
            tracking = FaceTrackingResult(
                metadata=meta,
                face_detected=True,
                landmarks=LandmarkSet(
                    points=z["landmarks"][-1].astype(np.float32), scheme="mediapipe_478"
                ),
                head_pose=HeadPose(*[float(v) for v in z["pose"][-1]]),
            )
            video = VideoFrame(data=frame, pixel_format=PixelFormat.BGR24, metadata=meta)
            timings: list[dict[str, float]] = []
            for _ in range(args.repeat):
                rendered = engine.animate(avatar_id, tracking, video)
                timings.append(dict(engine.last_timings))
            med = {k: float(np.median([t[k] for t in timings])) for k in timings[-1]}
            name = folder.name.split("-")[-1]
            cv2.imwrite(str(args.out / f"live_{name}_{avatar_id}.png"), rendered.data)
            row = {
                "capture": folder.name,
                "avatar": avatar_id,
                "yaw": round(float(z["pose"][-1][0]), 1),
                **{k: round(v, 2) for k, v in med.items()},
            }
            rows.append(row)
            print(
                f"  {name} x {avatar_id} (yaw {row['yaw']:+.0f}): frame {med['frame_ms']:.1f} ms"
                f" = align {med['align_ms']:.1f} + swap {med['swap_ms']:.1f}"
                f" + paste {med['swap_paste_ms']:.1f} + enhance {med['enhance_ms']:.1f}"
                f" + paste {med['enhance_paste_ms']:.1f}"
            )
    engine.stop()
    (args.out / "timings.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
