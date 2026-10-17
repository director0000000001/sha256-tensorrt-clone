"""Composition root.

The single place where concrete implementations are chosen and wired. Every
other module depends on protocols only; this one does the ``import`` and the
``registry.create(...)``.

Keeping it in one file has a practical payoff: to find out what sha256 is
actually running, you read this file, and nothing else can quietly introduce a
dependency on a specific tracker or engine behind your back.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sha256.app.kill_switch import KillSwitch
from sha256.app.registry import Registries
from sha256.avatar.cache import ResidencyCache
from sha256.avatar.engine_liveportrait import LivePortraitEngine
from sha256.avatar.engine_null import NullAvatarEngine
from sha256.avatar.engine_swap import SwapEngine
from sha256.avatar.engine_warp import WarpAvatarEngine
from sha256.avatar.manager import AvatarManager
from sha256.avatar.repository import FilesystemAvatarRepository
from sha256.calibration.profile import PROFILE_FILENAME, DriverProfile
from sha256.camera.null import NullCameraProvider
from sha256.camera.opencv_capture import OpenCVCameraProvider
from sha256.config.loader import load_settings
from sha256.config.paths import AppPaths
from sha256.config.schema import Settings
from sha256.core.errors import EngineError
from sha256.core.interfaces import (
    AvatarEngine,
    CameraProvider,
    FaceTracker,
    PerformanceMonitor,
    Renderer,
    VirtualCameraSink,
)
from sha256.performance.monitor import InProcessPerformanceMonitor, NullPerformanceMonitor
from sha256.pipeline.conditioning import OutputConditioner
from sha256.pipeline.controller import PipelineComponents, PipelineController
from sha256.rendering.null import NullRenderer
from sha256.tracking.mediapipe_tracker import (
    DEFAULT_MODEL_RELATIVE_PATH,
    MediaPipeFaceTracker,
)
from sha256.tracking.null import NullFaceTracker
from sha256.utils.logging import configure_logging, get_logger
from sha256.virtual_camera.null import NullVirtualCameraSink
from sha256.virtual_camera.pyvirtualcam_sink import PyVirtualCameraSink

__all__ = [
    "ApplicationContext",
    "apply_driver_profile",
    "build_application",
    "register_builtin_backends",
]

_LOG = get_logger(__name__)


def register_builtin_backends(registries: Registries) -> None:
    """Register every backend this build ships with.

    Only placeholders exist today. When a real backend lands it is added here
    with its ``requires`` list; nothing else in the codebase changes, and the
    module is not imported unless the backend is selected.
    """
    registries.camera.register(
        "null",
        NullCameraProvider,
        description="Synthetic frame generator (no hardware required).",
    )
    registries.camera.register(
        "opencv",
        OpenCVCameraProvider,
        description="Physical camera capture via OpenCV (Media Foundation / DirectShow).",
        requires=("cv2",),
    )
    registries.tracker.register(
        "null",
        NullFaceTracker,
        description="Synthetic tracking data. Not face tracking.",
    )
    registries.tracker.register(
        "mediapipe",
        MediaPipeFaceTracker,
        description="MediaPipe Face Landmarker — 478 landmarks, 52 blendshapes, CPU.",
        requires=("mediapipe",),
    )
    registries.engine.register(
        "null",
        NullAvatarEngine,
        description="Procedural placeholder face. Runs no model.",
    )
    registries.engine.register(
        "warp",
        WarpAvatarEngine,
        description="Warps a portrait photograph onto the tracked face. No model weights.",
        requires=("cv2",),
    )
    registries.engine.register(
        "liveportrait",
        LivePortraitEngine,
        description="Generates the portrait's face every frame with LivePortrait. Needs CUDA.",
        requires=("cv2", "torch"),
    )
    registries.engine.register(
        "swap",
        SwapEngine,
        description=(
            "Paints the portrait's identity onto the camera's face (inswapper + GFPGAN "
            "on TensorRT). Needs CUDA."
        ),
        requires=("cv2", "torch", "tensorrt"),
    )
    registries.renderer.register(
        "null",
        NullRenderer,
        description="CPU letterbox compositor.",
    )
    registries.virtual_camera.register(
        "null",
        NullVirtualCameraSink,
        description="Discards frames. Exposes nothing to the OS.",
    )
    registries.virtual_camera.register(
        "pyvirtualcam",
        PyVirtualCameraSink,
        description=(
            "Publishes to an existing virtual camera device (OBS). "
            "GPL-2.0 — development only, never distributed."
        ),
        requires=("pyvirtualcam",),
    )


@dataclass(slots=True)
class ApplicationContext:
    """Everything a running sha256 instance owns.

    Passed to the UI rather than reached for through globals, so a second
    instance (a benchmark harness, a test) is fully isolated.
    """

    settings: Settings
    paths: AppPaths
    registries: Registries
    monitor: PerformanceMonitor
    avatars: AvatarManager
    controller: PipelineController
    # Day 16: the driver profile that was laid over the engine at start,
    # if the machine has one (configs/local.driver.json) -- None otherwise.
    driver_profile: DriverProfile | None = None
    # Day 16 (D1/D2): the courtesy switch (unconfigured = inert), and what
    # the start had to say to the person -- a fallback taken, a card too
    # small -- for the window to show once.
    kill_switch: KillSwitch | None = None
    startup_notes: tuple[str, ...] = ()

    def shutdown(self) -> None:
        """Release everything, in the safe order. Idempotent."""
        if self.kill_switch is not None:
            self.kill_switch.stop()
        self.controller.stop()
        self.avatars.unload_all()
        engine = self.avatars.engine
        if engine is not None:
            engine.stop()


def build_application(
    *,
    settings: Settings | None = None,
    paths: AppPaths | None = None,
    setup_logging: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ApplicationContext:
    """Construct a fully wired application.

    Args:
        settings: Pre-built settings. Loaded from files/env when omitted.
        paths: Pre-resolved paths. Resolved when omitted.
        setup_logging: Configure the root logger. Tests pass ``False`` to keep
            pytest's own capture intact.
        progress: Told, in a sentence, what the start is doing -- the
            start-up window (2026-10-09); the neural engine says the most
            (the networks loading; at the first start on a machine, the
            fast engine being built for its graphics card, minutes).
    """
    paths = paths or AppPaths.resolve()
    paths.ensure()
    settings = settings or load_settings(paths)
    if progress is not None:
        progress("Starting")

    if setup_logging:
        configure_logging(
            level=settings.log_level,
            log_dir=paths.logs,
            json_format=settings.log_json,
        )

    for warning in settings.warnings:
        _LOG.warning("Configuration: %s", warning)

    registries = Registries()
    register_builtin_backends(registries)

    monitor: PerformanceMonitor = (
        InProcessPerformanceMonitor(
            window_size=settings.performance.window_size,
            enabled=True,
        )
        if settings.performance.enabled
        else NullPerformanceMonitor()
    )

    camera: CameraProvider = registries.camera.create(
        settings.camera.backend,
        device_index=settings.camera.device_index,
        width=settings.camera.width,
        height=settings.camera.height,
        target_fps=settings.camera.target_fps,
        use_mjpeg=settings.camera.use_mjpeg,
    )
    tracker: FaceTracker = _create_tracker(registries, settings, paths)
    renderer: Renderer = registries.renderer.create(
        settings.rendering.backend,
        width=settings.rendering.output_width,
        height=settings.rendering.output_height,
    )
    virtual_camera: VirtualCameraSink = _create_virtual_camera(registries, settings)
    engine, notes = _start_engine_with_fallback(registries, settings, paths, progress=progress)
    driver_profile = apply_driver_profile(engine, paths.configs / PROFILE_FILENAME)
    kill_switch = KillSwitch(
        settings.kill_switch.url,
        interval_seconds=settings.kill_switch.check_hours * 3600.0,
        timeout=settings.kill_switch.timeout_seconds,
    )

    if progress is not None:
        progress("Loading the avatar")
    avatars = _build_avatar_manager(settings, paths, engine)

    output = settings.output
    conditioner = OutputConditioner(
        quality=output.quality,
        failure_text=output.failure_text,
        auto_blackout=output.auto_blackout,
        lost_face_seconds=output.lost_face_seconds,
        lost_face_seconds_turned=output.lost_face_seconds_turned,
        min_blackout_seconds=output.min_blackout_seconds,
        recovery_seconds=output.recovery_seconds,
    )

    controller = PipelineController(
        PipelineComponents(
            camera=camera,
            tracker=tracker,
            renderer=renderer,
            virtual_camera=virtual_camera,
            avatars=avatars,
            monitor=monitor,
            conditioner=conditioner,
        ),
        publish_output=settings.virtual_camera.auto_start,
    )

    _LOG.info(
        "sha256 wired: camera=%s tracker=%s engine=%s renderer=%s vcam=%s",
        settings.camera.backend,
        settings.tracking.backend,
        settings.engine.backend,
        settings.rendering.backend,
        settings.virtual_camera.backend,
    )

    return ApplicationContext(
        settings=settings,
        paths=paths,
        registries=registries,
        monitor=monitor,
        avatars=avatars,
        controller=controller,
        driver_profile=driver_profile,
        kill_switch=kill_switch,
        startup_notes=tuple(notes),
    )


def _start_engine_with_fallback(
    registries: Registries,
    settings: Settings,
    paths: AppPaths,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[AvatarEngine, list[str]]:
    """Start the configured engine; when the neural engine cannot (no
    NVIDIA GPU, no CUDA, the weights missing), start the photograph
    engine in its place and say so in a plain sentence (day 16, D2) --
    the five run on their own machines and a machine without a card must
    still open. Also a note when the card is smaller than the neural
    avatars want."""
    notes: list[str] = []
    engine: AvatarEngine = _create_engine(registries, settings, paths, progress=progress)
    try:
        engine.start()
    except EngineError as exc:
        if settings.engine.backend not in ("liveportrait", "swap"):
            raise
        _LOG.warning("The neural engine cannot start here (%s); using the photograph engine", exc)
        notes.append(
            "This computer has no usable NVIDIA GPU for the neural avatar, so the "
            "photograph engine is running instead. Only 'warp' avatars will load."
        )
        fallback = settings.model_copy(
            update={"engine": settings.engine.model_copy(update={"backend": "warp"})}
        )
        engine = _create_engine(registries, fallback, paths)
        engine.start()
        return engine, notes
    vram = _gpu_memory_mb()
    if settings.engine.backend == "liveportrait" and vram is not None and vram < 6000:
        notes.append(
            f"The graphics card has {vram} MB; the neural avatar wants about 4500 MB free "
            "and may not fit beside a browser and the call."
        )
    return engine, notes


def _gpu_memory_mb() -> int | None:
    """The first CUDA device's memory in MB, or None without torch / CUDA."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    except Exception:  # a driver that answers strangely must not stop the start
        return None


def apply_driver_profile(engine: object, path: Path) -> DriverProfile | None:
    """Lay the driver profile at ``path`` (day 16, "Calibrate me") over the
    engine's avatars, for an engine that takes one -- ``set_driver_options``
    is the neural engine's, not part of the protocol, so any other engine
    is left alone. Before the default avatar is selected, so the first
    load already carries it. A profile the engine refuses (a key that is
    not the driver's) is reported and skipped, never fatal."""
    setter = getattr(engine, "set_driver_options", None)
    if setter is None:
        return None
    profile = DriverProfile.load(path)
    if profile is None:
        return None
    try:
        setter(profile.overlay())
    except ValueError as exc:
        _LOG.warning("Driver profile %s not applied: %s", path, exc)
        return None
    _LOG.info(
        "Driver profile from %s (measured %s): %s",
        path.name,
        profile.measured_at or "?",
        ", ".join(sorted(profile.overlay())) or "nothing",
    )
    return profile


def _create_virtual_camera(registries: Registries, settings: Settings) -> VirtualCameraSink:
    """Build the output sink and declare its format.

    Configured here, not at pipeline start, because the output format is a
    property of the wiring rather than of a run. Nothing was calling
    ``configure()`` at all, which made ``virtual_camera.auto_start = true``
    fail on the very check that exists to catch that ordering mistake — the
    placeholder sink caught the bug exactly as designed, and this is the fix.

    The two backends take different arguments for the same reason the trackers
    do: a sink that *creates* a device can be told what to call it, and a sink
    that writes to somebody else's device can only be told which one to pick.
    """
    virtual_camera = settings.virtual_camera

    if virtual_camera.backend == "pyvirtualcam":
        sink: VirtualCameraSink = registries.virtual_camera.create(
            "pyvirtualcam", device=virtual_camera.device
        )
    else:
        sink = registries.virtual_camera.create(
            virtual_camera.backend, device_name=virtual_camera.device_name
        )

    sink.configure(
        settings.rendering.output_width,
        settings.rendering.output_height,
        settings.rendering.target_fps,
    )
    return sink


def _create_engine(
    registries: Registries,
    settings: Settings,
    paths: AppPaths,
    progress: Callable[[str], None] | None = None,
) -> AvatarEngine:
    """Build the configured avatar engine.

    The warp engine needs the face-landmark model to read the avatar's own
    landmarks when it loads a portrait — the same model the tracker uses, so the
    two meshes are the same topology by construction.
    """
    models_root = paths.resolve_relative(settings.models.root_path)
    if settings.engine.backend == "liveportrait":
        # The same landmark model, for the portrait's face and the driver's
        # crop, plus LivePortrait's own weights under the models root.
        from sha256.avatar.engine_liveportrait import DEFAULT_WEIGHTS_RELATIVE_PATH
        from sha256.avatar.liveportrait_tensorrt import TENSORRT_RELATIVE_DIR

        neural: AvatarEngine = registries.engine.create(
            "liveportrait",
            model_path=models_root / DEFAULT_MODEL_RELATIVE_PATH,
            weights=models_root / DEFAULT_WEIGHTS_RELATIVE_PATH,
            # Phase 2 (2026-10-09): the pixel networks on TensorRT when the
            # setting and the engine under models/tensorrt/ allow.
            pixel_backend=settings.engine.pixel_backend,
            engines=models_root / TENSORRT_RELATIVE_DIR,
            progress=progress,
        )
        return neural
    if settings.engine.backend == "swap":
        # Road C (2026-10-17): the landmarker for the portrait's five points,
        # the swap's ONNX files under models/downloads/swap, the plans beside
        # LivePortrait's, the fp16 block list in models/metadata.
        from sha256.avatar.liveportrait_tensorrt import TENSORRT_RELATIVE_DIR
        from sha256.avatar.swap_models import SWAP_BLOCKS_RELATIVE_PATH, SWAP_RELATIVE_DIR

        swap: AvatarEngine = registries.engine.create(
            "swap",
            model_path=models_root / DEFAULT_MODEL_RELATIVE_PATH,
            models=models_root / SWAP_RELATIVE_DIR,
            engines=models_root / TENSORRT_RELATIVE_DIR,
            blocks=models_root / SWAP_BLOCKS_RELATIVE_PATH,
            progress=progress,
        )
        return swap
    if settings.engine.backend != "warp":
        placeholder: AvatarEngine = registries.engine.create(settings.engine.backend)
        return placeholder

    engine: AvatarEngine = registries.engine.create(
        "warp", model_path=models_root / DEFAULT_MODEL_RELATIVE_PATH
    )
    return engine


def _create_tracker(registries: Registries, settings: Settings, paths: AppPaths) -> FaceTracker:
    """Build the configured face tracker.

    Camera backends share one constructor signature, so the camera is created
    inline above. Trackers do not: MediaPipe needs a model bundle resolved
    against the models root, and the placeholder needs nothing at all. Resolving
    that difference is exactly the composition root's job — it is the one place
    allowed to know which backend is which, and keeping it here is what stops
    that knowledge leaking into the pipeline or the UI.
    """
    tracking = settings.tracking

    if tracking.backend != "mediapipe":
        placeholder: FaceTracker = registries.tracker.create(tracking.backend)
        return placeholder

    models_root = paths.resolve_relative(settings.models.root_path)
    configured = tracking.model_path
    model_path = (
        models_root / DEFAULT_MODEL_RELATIVE_PATH
        if configured is None
        else (Path(configured) if Path(configured).is_absolute() else models_root / configured)
    )

    tracker: FaceTracker = registries.tracker.create(
        "mediapipe",
        model_path=model_path,
        delegate=tracking.delegate,
        num_faces=tracking.num_faces,
        min_detection_confidence=tracking.min_detection_confidence,
        min_presence_confidence=tracking.min_presence_confidence,
        min_tracking_confidence=tracking.min_tracking_confidence,
        smoothing=tracking.smoothing,
        mouth_smoothing=tracking.mouth_smoothing,
        redetect_in_crop=tracking.redetect_in_crop,
    )
    return tracker


def _build_avatar_manager(
    settings: Settings, paths: AppPaths, engine: AvatarEngine
) -> AvatarManager:
    library_path: Path = paths.resolve_relative(settings.avatars.library_path)
    repository = FilesystemAvatarRepository(library_path)
    cache = ResidencyCache(
        max_resident=settings.avatars.max_resident,
        vram_budget_mb=settings.gpu.vram_budget_mb,
    )
    manager = AvatarManager(
        repository,
        cache,
        engine=engine,
        verify_checksums=settings.avatars.verify_checksums,
    )
    manager.refresh()

    default_id = settings.avatars.default_avatar_id
    if default_id and repository.exists(default_id):
        try:
            manager.select(default_id)
        except Exception:
            _LOG.exception("Could not select default avatar %s", default_id)

    return manager
