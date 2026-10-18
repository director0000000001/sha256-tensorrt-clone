"""The frame budget, measured through the application's own pipeline on a
recorded clip -- step 0 of "The fast version" (ROADMAP, 2026-10-08).

Not a render script: the clip goes through :class:`PipelineController` and,
in ``--mode threads``, through :class:`PipelineRunner` itself, so what is
timed is what the application does per frame -- the camera read, the
tracker, the engine (crop / model / hair / mask / paste, from its own
``last_timings``), the renderer's canvas, the conditioner, the sink -- and
the frame that comes out is the frame the virtual camera would get.

Every published frame's pixels are hashed. Two runs with the same clip,
range and pacing must give the same hash on every frame when a change is
off the drive path; ``--compare`` says so, frame by frame. That is the
proof the plan asks for, and it holds across the single-thread path
(``--mode once``) and the two-thread one (``--mode threads --pace
lockstep``), which must agree to the byte.

Pacing (``--pace``):

* ``asap`` -- the next frame as soon as it is asked for; the stage timings
  are the thing measured, the frame rate is meaningless.
* ``lockstep`` -- the next frame only once the previous one is published;
  no frame is ever dropped, so the frame sequence is identical to ``once``
  and the hashes must match. The byte-identity proof for the threads.
* ``realtime`` -- a frame every 1/fps of wall time, the latest due one, as
  a camera would hand it; frames the pipeline is too slow for are skipped
  at the source. This is the measurement of the achievable frame rate.

Outputs: a JSON of per-frame rows (hash and timings) under ``--out``, a
summary table on stdout, and with ``--stills`` the output frames named,
as PNG beside the JSON -- those are the owner's face composited; delete
them once looked at. ``--profile`` runs the single-thread loop under
cProfile and prints where the time goes inside the engine.

Usage::

    uv run python scripts/probes/frame_budget.py --clip recordings/driving.mkv \\
        --first 300 --last 900 --mode once --out recordings/budget/before-once.json
    uv run python scripts/probes/frame_budget.py --clip recordings/driving.mkv \\
        --first 300 --last 900 --mode threads --pace lockstep \\
        --out recordings/budget/after-threads.json --compare recordings/budget/before-once.json
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sha256.avatar.engine_liveportrait import (  # noqa: E402
    DEFAULT_WEIGHTS_RELATIVE_PATH,
    LivePortraitEngine,
)
from sha256.avatar.liveportrait_tensorrt import TENSORRT_RELATIVE_DIR  # noqa: E402
from sha256.avatar.repository import FilesystemAvatarRepository  # noqa: E402
from sha256.config.loader import load_settings  # noqa: E402
from sha256.config.paths import AppPaths  # noqa: E402
from sha256.core.enums import PixelFormat  # noqa: E402
from sha256.core.types import FrameMetadata, RenderedFrame, VideoFrame  # noqa: E402
from sha256.performance.monitor import InProcessPerformanceMonitor  # noqa: E402
from sha256.pipeline.controller import PipelineComponents, PipelineController  # noqa: E402
from sha256.pipeline.runner import PipelineRunner, PipelineSnapshot  # noqa: E402
from sha256.rendering.null import NullRenderer  # noqa: E402
from sha256.tracking.mediapipe_tracker import (  # noqa: E402
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)
from sha256.utils.timing import now  # noqa: E402
from sha256.virtual_camera.null import NullVirtualCameraSink  # noqa: E402

ENGINE_KEYS = (
    "crop",
    "launch",
    "model",
    "hair",
    "mask",
    "paste",
    "paste_pre",
    "wait",
    "paste_fit",
    "paste_post",
    "upload",
    # the swap engine's parts (road C, 2026-10-17); None for LivePortrait
    "align",
    "swap",
    "swap_paste",
    "enhance",
    "enhance_paste",
    "shave",
    "lips",
    "skin",
)


class ClipCamera:
    """A :class:`CameraProvider` that reads a clip, paced three ways."""

    def __init__(
        self, path: Path, *, first: int, last: int | None, pace: str, fps: float | None
    ) -> None:
        import cv2

        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise SystemExit(f"cannot open {path}")
        self._fps = fps or float(self._capture.get(cv2.CAP_PROP_FPS)) or 30.0
        total = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last = total - 1 if last is None else min(last, total - 1)
        self.first = first
        self._pace = pace
        self._index = 0
        self._running = False
        # asap / lockstep stamp frames at a fixed origin, so two runs feed
        # the tracker's VIDEO mode and the engine's filters the very same
        # millisecond stamps -- a run's own clock would round the
        # millisecond differently and the landmarks' smoothing with it.
        self._t0 = 1000.0
        self._released = first - 1
        self._condition = threading.Condition()
        self.exhausted = False
        self.skipped = 0

    # --- Component -----------------------------------------------------------

    @property
    def backend_id(self) -> str:
        return "clip"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        # Decode and discard up to ``first``: a seek in an mkv lands on a
        # keyframe, not on the frame asked for.
        while self._index < self.first:
            ok, _ = self._capture.read()
            if not ok:
                raise SystemExit("the clip ended before --first")
            self._index += 1
        self._running = True
        if self._pace == "realtime":
            self._t0 = now()

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._released = 1 << 60
            self._condition.notify_all()

    def enumerate_devices(self) -> list[Any]:
        return []

    def frames(self) -> Any:
        while self._running:
            frame = self.read()
            if frame is not None:
                yield frame

    # --- pacing --------------------------------------------------------------

    def release(self, frame_id: int) -> None:
        """Lockstep: frame ``frame_id`` is published, the next may be read."""
        with self._condition:
            self._released = max(self._released, frame_id)
            self._condition.notify_all()

    def read(self) -> VideoFrame | None:
        if not self._running or self.exhausted:
            return None
        index = self._index
        if index > self.last:
            self.exhausted = True
            return None
        if self._pace == "lockstep":
            with self._condition:
                while self._released < index - 1 and self._running:
                    self._condition.wait(0.05)
                if not self._running:
                    return None
            stamp = self._t0 + (index - self.first) / self._fps
        elif self._pace == "realtime":
            due = self._t0 + (index - self.first) / self._fps
            current = now()
            if current < due:
                time.sleep(due - current)
            else:
                # Late: the camera has moved on. Hand the latest due frame.
                behind = int((current - due) * self._fps)
                for _ in range(min(behind, self.last - index)):
                    ok, _ = self._capture.read()
                    if not ok:
                        self.exhausted = True
                        return None
                    self._index += 1
                    self.skipped += 1
                index = self._index
            stamp = now()
        else:
            stamp = self._t0 + (index - self.first) / self._fps
        ok, image = self._capture.read()
        if not ok:
            self.exhausted = True
            return None
        self._index = index + 1
        return VideoFrame(
            data=np.ascontiguousarray(image),
            pixel_format=PixelFormat.BGR24,
            metadata=FrameMetadata(frame_id=index, capture_timestamp=stamp, source_id="clip"),
        )


class _Timed:
    """Delegates everything to ``inner``; times one method into ``self.ms``."""

    def __init__(self, inner: Any, method: str) -> None:
        self._inner = inner
        self._method = method
        self.ms = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _call(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return getattr(self._inner, self._method)(*args, **kwargs)
        finally:
            self.ms = (time.perf_counter() - started) * 1000.0


class TimedTracker(_Timed):
    def __init__(self, inner: Any) -> None:
        super().__init__(inner, "process")

    def process(self, frame: VideoFrame) -> Any:
        return self._call(frame)


class TimedEngine(_Timed):
    def __init__(self, inner: Any) -> None:
        super().__init__(inner, "animate")

    def animate(self, *args: Any, **kwargs: Any) -> Any:
        return self._call(*args, **kwargs)


class _Avatars:
    """What the controller and the runner read of the avatar manager."""

    def __init__(self, engine: Any, descriptor: Any) -> None:
        self.engine = engine
        self.active = descriptor
        self.active_id = descriptor.id


def frame_hash(frame: RenderedFrame) -> str:
    return hashlib.sha1(np.ascontiguousarray(frame.data).tobytes()).hexdigest()


def summarise(rows: list[dict[str, Any]], keys: list[str]) -> list[tuple[str, float, float, float]]:
    table = []
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        if not values:
            continue
        table.append(
            (
                key,
                float(np.percentile(values, 50)),
                float(np.percentile(values, 90)),
                float(np.percentile(values, 99)),
            )
        )
    return table


def print_table(title: str, table: list[tuple[str, float, float, float]]) -> None:
    print(f"\n{title}")
    print(f"  {'stage':<14}{'p50':>9}{'p90':>9}{'p99':>9}")
    for key, p50, p90, p99 in table:
        print(f"  {key:<14}{p50:>9.2f}{p90:>9.2f}{p99:>9.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--avatar", default="my_avatar_neural")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Override the clip's own rate.")
    parser.add_argument("--mode", choices=("once", "threads"), default="once")
    parser.add_argument("--pace", choices=("asap", "lockstep", "realtime"), default="asap")
    parser.add_argument("--hair", action="store_true", help="The avatar's hair switch on.")
    parser.add_argument("--warmup", type=int, default=30, help="Frames left out of the summary.")
    parser.add_argument("--out", type=Path, default=None, help="Per-frame JSON.")
    parser.add_argument("--compare", type=Path, default=None, help="A previous run's JSON.")
    parser.add_argument("--stills", default="", help="Frame ids to write as PNG beside --out.")
    parser.add_argument("--profile", action="store_true", help="cProfile the once-loop.")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--cpu-paste",
        action="store_true",
        help="The face downloaded and pasted on the CPU, as before 2026-10-08.",
    )
    parser.add_argument(
        "--cpu-fit",
        action="store_true",
        help="The GPU paste, but the colour fit on the CPU from downloaded cells.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="cuDNN's benchmark off and its deterministic kernels on before the avatar "
        "loads, so two runs pick the same kernels and the hashes can be compared; "
        "the model is a little slower this way.",
    )
    parser.add_argument(
        "--pixel-backend",
        choices=("torch", "auto", "tensorrt"),
        default="torch",
        help="Which networks draw the face (2026-10-09, B006): torch, as always; "
        "tensorrt, the engine under models/tensorrt/ for this GPU (an error without "
        "it); auto, the engine when it is there. The application's own default is auto.",
    )
    parser.add_argument(
        "--engine",
        choices=("liveportrait", "swap"),
        default="liveportrait",
        help="Which engine draws the face (road C, 2026-10-17): liveportrait, as always; "
        "swap, the face swap (inswapper + GFPGAN on TensorRT) with a 'swap' avatar.",
    )
    args = parser.parse_args()

    if args.mode == "threads" and args.pace == "asap":
        print("note: threads + asap drops frames by design; use lockstep for the hash proof")
    stills = {int(s) for s in args.stills.split(",") if s.strip()}

    paths = AppPaths.resolve()
    settings = load_settings(paths)
    models = paths.resolve_relative(settings.models.root_path)
    repository = FilesystemAvatarRepository(paths.resolve_relative(settings.avatars.library_path))

    tracker = MediaPipeFaceTracker(
        model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
        smoothing=settings.tracking.smoothing,
        mouth_smoothing=settings.tracking.mouth_smoothing,
        min_detection_confidence=settings.tracking.min_detection_confidence,
        min_presence_confidence=settings.tracking.min_presence_confidence,
        min_tracking_confidence=settings.tracking.min_tracking_confidence,
        redetect_in_crop=settings.tracking.redetect_in_crop,
    )
    engine: Any
    if args.engine == "swap":
        from sha256.avatar.engine_swap import SwapEngine
        from sha256.avatar.swap_models import SWAP_BLOCKS_RELATIVE_PATH, SWAP_RELATIVE_DIR

        engine = SwapEngine(
            model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
            models=models / SWAP_RELATIVE_DIR,
            engines=models / TENSORRT_RELATIVE_DIR,
            blocks=models / SWAP_BLOCKS_RELATIVE_PATH,
        )
        engine.start()
        print("engine: swap (inswapper + GFPGAN on TensorRT)")
    else:
        engine = LivePortraitEngine(
            model_path=models / DEFAULT_MODEL_RELATIVE_PATH,
            weights=models / DEFAULT_WEIGHTS_RELATIVE_PATH,
            pixel_backend=args.pixel_backend,
            engines=models / TENSORRT_RELATIVE_DIR,
        )
        engine.start()
        print(f"pixel networks: {engine.pixel_backend} (asked: {args.pixel_backend})")
        if args.cpu_paste:
            engine.paste_on_gpu = False
            print("paste: on the CPU")
        if args.cpu_fit:
            engine.fit_on_gpu = False
            print("colour fit: on the CPU")
    if args.deterministic:
        import torch

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("deterministic: cudnn.benchmark off, cudnn.deterministic on")
    descriptor = repository.get(args.avatar)
    for item in args.option:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--option expects KEY=VALUE, got {item!r}")
        try:
            descriptor.manifest.engine_options[key] = json.loads(raw)
        except json.JSONDecodeError:
            descriptor.manifest.engine_options[key] = raw
    started = time.perf_counter()
    engine.load_avatar(descriptor)
    print(f"avatar {args.avatar} loaded in {time.perf_counter() - started:.1f} s")
    if args.hair:
        if not engine.set_hair(descriptor.id, True):
            raise SystemExit("the hair switch could not be turned on")
        print("hair: on")

    camera = ClipCamera(args.clip, first=args.first, last=args.last, pace=args.pace, fps=args.fps)
    timed_tracker = TimedTracker(tracker)
    timed_engine = TimedEngine(engine)
    monitor = InProcessPerformanceMonitor(window_size=200_000)
    sink = NullVirtualCameraSink()
    sink.configure(1280, 720, 30.0)
    components = PipelineComponents(
        camera=camera,  # type: ignore[arg-type]
        tracker=timed_tracker,  # type: ignore[arg-type]
        renderer=NullRenderer(width=1280, height=720),
        virtual_camera=sink,
        avatars=_Avatars(timed_engine, descriptor),  # type: ignore[arg-type]
        monitor=monitor,
    )
    controller = PipelineController(components, publish_output=True)

    rows: list[dict[str, Any]] = []
    out_dir = args.out.parent if args.out is not None else REPO_ROOT / "recordings" / "budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out.stem if args.out is not None else f"{args.mode}-{args.pace}"

    def record(frame: RenderedFrame, *, total_ms: float | None, published_at: float) -> None:
        row: dict[str, Any] = {
            "frame_id": frame.metadata.frame_id,
            "hash": frame_hash(frame),
            "tracker": timed_tracker.ms,
            "animate": timed_engine.ms,
            "latency_ms": frame.pipeline_latency * 1000.0,
            "published_at": published_at,
            "passthrough": bool(frame.is_passthrough),
        }
        for key in ENGINE_KEYS:
            value = engine.last_timings.get(f"{key}_ms")
            row[key] = None if value is None else float(value)
        if total_ms is not None:
            row["total"] = total_ms
        rows.append(row)
        if frame.metadata.frame_id in stills:
            import cv2

            target = out_dir / f"{tag}-f{frame.metadata.frame_id}.png"
            cv2.imwrite(str(target), np.ascontiguousarray(frame.data))
            print(f"still written: {target} (his face; delete when looked at)")

    wall_started = time.perf_counter()
    if args.mode == "once":
        controller.start()
        profiler = cProfile.Profile() if args.profile else None
        try:
            while not camera.exhausted:
                if profiler is not None and len(rows) == args.warmup:
                    profiler.enable()
                started = time.perf_counter()
                frame = controller.process_once()
                total_ms = (time.perf_counter() - started) * 1000.0
                if frame is None:
                    continue
                record(frame, total_ms=total_ms, published_at=now())
        finally:
            if profiler is not None:
                profiler.disable()
            controller.stop()
        if profiler is not None:
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats("tottime").print_stats(45)
            print(stream.getvalue())
    else:
        done = threading.Event()

        def observer(snapshot: PipelineSnapshot) -> None:
            if snapshot.frame is None:
                return
            record(snapshot.frame, total_ms=None, published_at=now())
            camera.release(snapshot.frame_id)
            if snapshot.frame_id >= camera.last:
                done.set()

        runner = PipelineRunner(controller)
        runner.add_observer(observer)
        runner.start()
        try:
            last_count = -1
            quiet_since = time.perf_counter()
            while not done.is_set():
                time.sleep(0.05)
                if runner.last_error:
                    print(f"pipeline error: {runner.last_error}")
                    break
                if len(rows) != last_count:
                    last_count = len(rows)
                    quiet_since = time.perf_counter()
                elif camera.exhausted and time.perf_counter() - quiet_since > 2.0:
                    break
                elif time.perf_counter() - quiet_since > 30.0:
                    print("no frame published for 30 s; giving up")
                    break
        finally:
            runner.stop()
        print(f"runner stats: {runner.stats}; source frames skipped: {camera.skipped}")
    wall = time.perf_counter() - wall_started

    engine.stop()

    measured = rows[args.warmup :] if len(rows) > args.warmup else rows
    print(f"\n{args.mode} / {args.pace}: {len(rows)} frames published, {len(measured)} measured")
    if len(measured) >= 2:
        intervals = np.diff([r["published_at"] for r in measured]) * 1000.0
        print(
            f"  publish interval ms: p50 {np.percentile(intervals, 50):.2f}  "
            f"p90 {np.percentile(intervals, 90):.2f}  -> {1000.0 / np.mean(intervals):.1f} fps"
        )
    print(f"  wall {wall:.1f} s, {len(rows) / wall:.1f} frames/s overall")
    keys = ["tracker", "animate", *ENGINE_KEYS, "latency_ms"]
    if args.mode == "once":
        keys.append("total")
    print_table("per frame, ms", summarise(measured, keys))
    snapshot = monitor.snapshot()
    stage_table = []
    for stage, summary in sorted(snapshot.durations.items()):
        stage_table.append(
            (
                stage,
                summary.p50 * 1000.0,
                summary.p95 * 1000.0,
                getattr(summary, "p99", summary.p95) * 1000.0,
            )
        )
    if stage_table:
        print("\nmonitor stages, ms (p50 / p95 / p99)")
        for stage, p50, p95, p99 in stage_table:
            print(f"  {stage:<14}{p50:>9.2f}{p95:>9.2f}{p99:>9.2f}")

    if args.out is not None:
        payload = {
            "clip": str(args.clip),
            "first": args.first,
            "last": camera.last,
            "mode": args.mode,
            "pace": args.pace,
            "hair": args.hair,
            "avatar": args.avatar,
            "rows": rows,
        }
        args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nrows written: {args.out}")

    if args.compare is not None:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        theirs = {r["frame_id"]: r["hash"] for r in other["rows"]}
        mine = {r["frame_id"]: r["hash"] for r in rows}
        shared = sorted(set(theirs) & set(mine))
        same = [f for f in shared if theirs[f] == mine[f]]
        differ = [f for f in shared if theirs[f] != mine[f]]
        print(
            f"\ncompare against {args.compare.name}: {len(shared)} frames in both, "
            f"{len(same)} identical, {len(differ)} differ"
        )
        if differ:
            print(f"  differing frames: {differ[:20]}{' ...' if len(differ) > 20 else ''}")
        only_mine = sorted(set(mine) - set(theirs))
        only_theirs = sorted(set(theirs) - set(mine))
        if only_mine or only_theirs:
            print(f"  only here: {len(only_mine)}; only there: {len(only_theirs)}")
        return 0 if not differ else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
