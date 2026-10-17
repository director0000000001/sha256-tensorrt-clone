"""Road C: the swap engine on chosen frames of a recorded clip, with the mouth zoomed.

The clip's frames are tracked in order (the application's tracker, video mode),
the chosen ones rendered by the swap engine, and for each a sheet is written:
the original face at 512, the swapped face at 512, the original mouth zoomed,
the swapped mouth zoomed. For the mouth work (his look 2026-10-17: "the mouth
does not follow well, the teeth on a smile"): driving.mkv's smile-with-teeth
(868-985), mouth-wide-open (986-1103) and talk (1104-1341) segments.

Usage (from the project root):
  .venv\\Scripts\\python.exe -I scripts/probes/swap_clip_stills.py --out <dir> --tag before
      [--clip recordings/driving.mkv] [--frames 900,960,1010,1070,1150,1250]
      [--avatar bidzo_swap] [--option enhance_strength=0.75]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES = "900,930,960,1010,1040,1070,1150,1200,1250,1300"
MOUTH_POINTS = (61, 291, 0, 17, 13, 14)  # corners, the outer lips' middles, the inner ones


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default="before")
    p.add_argument("--clip", type=Path, default=PROJECT_ROOT / "recordings" / "driving.mkv")
    p.add_argument("--frames", default=DEFAULT_FRAMES)
    p.add_argument("--avatar", default="bidzo_swap")
    p.add_argument(
        "--lead", type=int, default=30, help="frames tracked before the first chosen one"
    )
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    import cv2

    from sha256.avatar import swap_models as sm
    from sha256.avatar.engine_swap import SwapEngine
    from sha256.avatar.repository import FilesystemAvatarRepository
    from sha256.config.loader import load_settings
    from sha256.config.paths import AppPaths
    from sha256.core.enums import PixelFormat
    from sha256.core.types import FrameMetadata, VideoFrame
    from sha256.tracking.mediapipe_tracker import MediaPipeFaceTracker

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    wanted = sorted({int(f) for f in args.frames.split(",") if f.strip()})
    tracker = MediaPipeFaceTracker(
        model_path=models / "downloads" / "face_landmarker.task",
        smoothing=settings.tracking.smoothing,
        mouth_smoothing=settings.tracking.mouth_smoothing,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
        redetect_in_crop=settings.tracking.redetect_in_crop,
    )
    engine = SwapEngine(
        model_path=models / "downloads" / "face_landmarker.task",
        models=models / sm.SWAP_RELATIVE_DIR,
        engines=models / "tensorrt",
        blocks=models / sm.SWAP_BLOCKS_RELATIVE_PATH,
    )
    engine.start()
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))
    descriptor = repository.get(args.avatar)
    for item in args.option:
        key, _, raw = item.partition("=")
        try:
            descriptor.manifest.engine_options[key] = json.loads(raw)
        except json.JSONDecodeError:
            descriptor.manifest.engine_options[key] = raw
    engine.load_avatar(descriptor)
    print(f"options: {engine._loaded[args.avatar].options}")  # noqa: SLF001

    capture = cv2.VideoCapture(str(args.clip))
    if not capture.isOpened():
        raise SystemExit(f"{args.clip} could not be opened")
    first = max(wanted[0] - args.lead, 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    tracker.start()
    args.out.mkdir(parents=True, exist_ok=True)
    index = first
    timings = []
    started = time.perf_counter()
    while index <= wanted[-1]:
        ok, image = capture.read()
        if not ok:
            break
        meta = FrameMetadata(frame_id=index, capture_timestamp=index / 30.0, source_id="clip")
        frame = VideoFrame(
            data=np.ascontiguousarray(image), pixel_format=PixelFormat.BGR24, metadata=meta
        )
        tracking = tracker.process(frame)
        if index in wanted:
            rendered = engine.animate(args.avatar, tracking, frame)
            timings.append(dict(engine.last_timings))
            if tracking.landmarks is None:
                print(f"  f{index}: no face")
                index += 1
                continue
            points = tracking.landmarks.points.copy()
            points[:, 0] *= image.shape[1]
            points[:, 1] *= image.shape[0]
            kps = sm.five_points(points)
            m512 = sm.umeyama(kps, sm.FFHQ_DST_512)
            face_in = sm.warp(image, m512, 512)
            face_out = sm.warp(rendered.data, m512, 512)
            mouth = points[list(MOUTH_POINTS), :2]
            centre = mouth.mean(axis=0)
            side = max(float(np.ptp(mouth[:, 0])) * 2.2, 120.0)
            x0, y0 = int(centre[0] - side / 2), int(centre[1] - side / 2)
            x1, y1 = int(centre[0] + side / 2), int(centre[1] + side / 2)
            zoom_in = cv2.resize(image[y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_CUBIC)
            zoom_out = cv2.resize(
                rendered.data[y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_CUBIC
            )

            def label(img: np.ndarray, text: str) -> np.ndarray:
                out = img.copy()
                cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
                cv2.putText(
                    out,
                    text,
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                return out

            sheet = np.hstack(
                [
                    label(face_in, f"f{index} original"),
                    label(face_out, f"{args.tag}: swap"),
                    label(zoom_in, "mouth original"),
                    label(zoom_out, f"mouth {args.tag}"),
                ]
            )
            cv2.imwrite(str(args.out / f"mouth_{args.tag}_f{index}_{args.avatar}.png"), sheet)
            cv2.imwrite(
                str(args.out / f"full_{args.tag}_f{index}_{args.avatar}.png"), rendered.data
            )
            t = engine.last_timings
            print(
                f"  f{index}: frame {t['frame_ms']:.1f} ms (swap {t['swap_ms']:.1f}, enhance {t['enhance_ms']:.1f})"
            )
        index += 1
    tracker.stop()
    engine.stop()
    if timings:
        med = {k: round(float(np.median([t[k] for t in timings])), 2) for k in timings[-1]}
        print(f"median over {len(timings)} rendered frames: {med}")
    print(f"{index - first} frames in {time.perf_counter() - started:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
