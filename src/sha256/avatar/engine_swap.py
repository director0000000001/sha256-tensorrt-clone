"""The one-shot face swap engine (road C, 2026-10-17).

LivePortrait draws the avatar's photograph and moves it with the driver's
face; this engine goes the other way: it takes the DRIVER's face as the camera
sees it (pose, expression, turn, light) and paints the avatar's IDENTITY onto
it. One ArcFace embedding of the portrait at load; per frame the tracker's
five points align a 128 px crop, inswapper_128 re-draws it with that identity,
GFPGAN restores the soft result at 512, and both are pasted back inside the
driver's own face oval. No photograph to turn, no calibration, any face; the
output is the driver's face shape with the avatar's features.

Manifest: ``engine.engine_id = "swap"``, one asset with role
``source_portrait``. ``engine_options`` (all optional):

- ``enhance`` (true): GFPGAN after the swap.
- ``enhance_strength`` (0.75): how much of GFPGAN's face replaces the swap's.
- ``swap_mask`` ("hull"): "hull" keeps the avatar inside the driver's face oval
  (nothing on the hair); "box" is InsightFace's whole-crop paste.
- ``mask_grow`` (0.03), ``mask_feather`` (0.04): the oval's dilation and
  feather as a share of the face's height.
- ``lips_keep`` (1.0): the driver's own mouth, lips and all, through a hole in
  the paste (his look 2026-10-17: inswapper drew his talking mouth a third
  open and the restorer invented teeth; the mouth is where the motion is, the
  identity sits in the eyes, nose and skin). ``mouth_hole`` (1.0): the opening
  alone, when ``lips_keep`` is below 1.
- ``shave`` ("auto"): the driver's stubble smoothed and tinted away under
  the nose before the restorer draws clean skin; "auto" reads the portrait
  (a bearded avatar keeps it), 0 or 1 says so outright.
- ``swap_zoom`` (1.0), ``swap_shift`` (0.0): the swap crop drawn smaller or
  higher (more chin); measured, neither helped the mouth, so both stay off.
- ``skin_match`` (0.0): the face's skin colour, 0 the driver's own (the
  swap's and the restorer's warm cast corrected, the face the colour of his
  neck) and 1 the portrait's, eased over frames; ``"off"`` leaves the colour
  as the networks drew it (his look 2026-10-18, "the app gives the avatar a
  tan": the swap keeps the driver's lightness and adds red and yellow).
- ``forehead_fade`` (0.8): the paste fading from the brows up to the hairline,
  so the driver's own forehead shows through (2026-10-18: the avatar's
  forehead came paler and flatter than his, a band at the hairline).

See :mod:`sha256.avatar.swap_models` for the networks and the TensorRT plans.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sha256.avatar import swap_models as sm
from sha256.core.avatar import AvatarDescriptor
from sha256.core.enums import DeviceKind, PixelFormat
from sha256.core.errors import AvatarLoadError, EngineError
from sha256.core.types import FaceTrackingResult, RenderedFrame, VideoFrame
from sha256.utils.logging import get_logger
from sha256.utils.timing import now

__all__ = ["ENGINE_ID", "SwapEngine"]

_LOG = get_logger(__name__)

ENGINE_ID = "swap"
PORTRAIT_ROLE = "source_portrait"
LANDMARK_SCHEME = "mediapipe_478"
SWAP_SIZE = 128
ENHANCE_SIZE = 512
DEFAULT_OPTIONS: dict[str, Any] = {
    "enhance": True,
    "enhance_strength": 0.75,
    "swap_mask": "hull",
    "mask_grow": 0.03,
    "mask_feather": 0.04,
    "mouth_hole": 1.0,
    "lips_keep": 0.0,
    "lips_source": "avatar",
    "shave": "auto",
    "skin_match": 0.0,
    "forehead_fade": 0.8,
    "swap_zoom": 1.0,
    "swap_shift": 0.0,
}


def _skin_match(raw: Any) -> float | None:
    """The ``skin_match`` option as a strength, ``None`` when it is off."""
    if raw is None or raw is False or (isinstance(raw, str) and raw.lower() in ("off", "none")):
        return None
    return float(raw)


@dataclass
class _Loaded:
    descriptor: AvatarDescriptor
    latent: np.ndarray
    embedding: np.ndarray
    options: dict[str, Any] = field(default_factory=dict)
    portrait512: np.ndarray | None = None
    portrait_pts512: np.ndarray | None = None
    portrait_hull512: np.ndarray | None = None
    portrait_lab: np.ndarray | None = None
    # the skin match's eased shifts, one per layer (see ``animate``)
    skin_shift128: np.ndarray | None = None
    skin_shift512: np.ndarray | None = None


class SwapEngine:
    """The driver's face with the avatar's identity, on TensorRT."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        models: Path | str | None = None,
        engines: Path | str | None = None,
        blocks: Path | str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        # The face-landmark model, to read the portrait's five points at load.
        self._model_path = Path(model_path) if model_path is not None else None
        self._models = Path(models) if models is not None else None
        self._engines = Path(engines) if engines is not None else None
        self._blocks = Path(blocks) if blocks is not None else None
        self._progress = progress
        self._running = False
        self._nets: sm.SwapNets | None = None
        self._loaded: dict[str, _Loaded] = {}
        # The last frame's parts in ms, for the debug panel and frame_budget.
        self.last_timings: dict[str, float] = {}

    # --- Component ---------------------------------------------------------

    @property
    def backend_id(self) -> str:
        return ENGINE_ID

    @property
    def engine_id(self) -> str:
        return ENGINE_ID

    @property
    def device(self) -> DeviceKind:
        return DeviceKind.CUDA

    @property
    def is_running(self) -> bool:
        return self._running

    def _say(self, text: str) -> None:
        if self._progress is not None:
            self._progress(text)

    def start(self) -> None:
        if self._running:
            return
        try:
            import torch
        except ImportError as exc:
            msg = "the swap engine needs torch and tensorrt installed"
            raise EngineError(msg) from exc
        if not torch.cuda.is_available():
            msg = "the swap engine needs an NVIDIA GPU with CUDA, and none is available here"
            raise EngineError(msg)
        if self._models is None or self._engines is None:
            msg = "the swap engine needs the models and engines folders"
            raise EngineError(msg)
        try:
            plans = sm.ensure_plans(
                self._models, self._engines, self._blocks, progress=self._progress
            )
            self._say("Loading the face swap engines")
            emap = sm.load_emap(self._models)
            self._nets = sm.SwapNets(plans, emap)
        except Exception as exc:
            msg = f"the swap engine could not start: {exc}"
            raise EngineError(msg) from exc
        self._running = True
        _LOG.info(
            "swap engine ready on %s: %s",
            torch.cuda.get_device_name(0),
            ", ".join(p.name for p in plans.values()),
        )

    def stop(self) -> None:
        for avatar_id in list(self._loaded):
            self.unload_avatar(avatar_id)
        if self._nets is not None:
            self._nets.close()
            self._nets = None
        self._running = False

    # --- AvatarEngine ------------------------------------------------------

    def supports(self, descriptor: AvatarDescriptor) -> bool:
        """Avatars made for this engine, and any other with a portrait: the
        swap needs one photograph of the face and nothing else, so the
        library's LivePortrait and photograph avatars drive it as they are
        (his ask 2026-10-17: "can I not use the other avatars?")."""
        if descriptor.manifest.engine.engine_id == ENGINE_ID:
            return True
        return descriptor.manifest.asset_by_role(PORTRAIT_ROLE) is not None

    def load_avatar(self, descriptor: AvatarDescriptor) -> None:
        if descriptor.id in self._loaded:
            return
        if not self.supports(descriptor):
            msg = f"avatar {descriptor.id!r} has no {PORTRAIT_ROLE!r} photograph for the swap"
            raise AvatarLoadError(msg)
        if self._nets is None:
            msg = "the swap engine is not started"
            raise AvatarLoadError(msg)
        ref = descriptor.manifest.asset_by_role(PORTRAIT_ROLE)
        if ref is None:
            msg = f"avatar {descriptor.id!r} declares no {PORTRAIT_ROLE!r} asset"
            raise AvatarLoadError(msg)
        portrait_path = descriptor.asset_path(ref)
        started = time.perf_counter()
        embedding, portrait, p_points = self._identity(portrait_path)
        latent = self._nets.latent(embedding)
        options = dict(DEFAULT_OPTIONS)
        options.update(
            {k: v for k, v in descriptor.manifest.engine_options.items() if k in DEFAULT_OPTIONS}
        )
        # the portrait at GFPGAN's alignment, for its lips (lips_source "avatar")
        m512 = sm.umeyama(sm.five_points(p_points), sm.FFHQ_DST_512)
        portrait512 = sm.warp(portrait, m512, ENHANCE_SIZE)
        portrait_pts = sm.points_in_crop(p_points, m512)
        portrait_hull = sm.crop_mask(ENHANCE_SIZE, mode="hull", points_px=p_points, m=m512)
        portrait_lab = sm.lab_median(portrait512, portrait_hull)
        if options.get("shave") == "auto":
            # a clean-shaven portrait asks for the driver's stubble to go
            signal = sm.beard_signal(portrait512, p_points, m512)
            options["shave"] = 0.0 if signal >= sm.BEARD_THRESHOLD else 1.0
            _LOG.info(
                "avatar %s: beard signal %.2f -> shave %.0f",
                descriptor.id,
                signal,
                options["shave"],
            )
        self._loaded[descriptor.id] = _Loaded(
            descriptor,
            latent,
            embedding,
            options,
            portrait512,
            portrait_pts,
            portrait_hull,
            portrait_lab,
        )
        _LOG.info(
            "Loaded avatar %s into the swap engine in %.1f s (identity from %s)",
            descriptor.id,
            time.perf_counter() - started,
            portrait_path.name,
        )

    def _identity(self, portrait_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The portrait's ArcFace embedding: its five points from the same
        landmarker the tracker uses, the 112 px ArcFace crop, the network."""
        import cv2

        from sha256.core.types import FrameMetadata
        from sha256.tracking.mediapipe_tracker import MediaPipeFaceTracker

        if self._nets is None:
            msg = "the swap engine is not started"
            raise AvatarLoadError(msg)
        portrait = cv2.imread(str(portrait_path), cv2.IMREAD_COLOR)
        if portrait is None:
            msg = f"{portrait_path.name} could not be read as an image"
            raise AvatarLoadError(msg)
        if self._model_path is None:
            msg = "the swap engine needs the face-landmark model to read the portrait"
            raise AvatarLoadError(msg)
        tracker = MediaPipeFaceTracker(
            model_path=self._model_path, smoothing=0.0, mouth_smoothing=0.0, running_mode="image"
        )
        tracker.start()
        try:
            result = tracker.process(
                VideoFrame(
                    data=np.ascontiguousarray(portrait),
                    pixel_format=PixelFormat.BGR24,
                    metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
                )
            )
        finally:
            tracker.stop()
        if not result.face_detected or result.landmarks is None:
            msg = f"no face found in {portrait_path.name}"
            raise AvatarLoadError(msg)
        points = result.landmarks.points.copy()
        points[:, 0] *= portrait.shape[1]
        points[:, 1] *= portrait.shape[0]
        kps = sm.five_points(points)
        crop = sm.warp(portrait, sm.estimate_norm(kps, 112), 112)
        return self._nets.embed(crop), portrait, points

    def unload_avatar(self, avatar_id: str) -> None:
        self._loaded.pop(avatar_id, None)

    @property
    def loaded_avatars(self) -> Sequence[str]:
        return tuple(self._loaded)

    def estimated_vram_mb(self, avatar_id: str) -> int | None:
        """The plans' weights and activations, shared by every avatar (an
        avatar itself is a 512-vector)."""
        del avatar_id
        if self._nets is None:
            return None
        weights = sum(
            p.plan.stat().st_size
            for p in (self._nets.arcface, self._nets.swapper, self._nets.enhancer)
        )
        return int((weights + self._nets.device_memory_bytes) / (1024 * 1024))

    def reset_anchor(self, avatar_id: str) -> None:
        """No-op: nothing is learned from the driver; the identity is the portrait's."""
        del avatar_id

    def rest_status(self, avatar_id: str) -> str | None:
        del avatar_id
        return None

    def set_driver_options(self, options: Mapping[str, Any] | None) -> None:
        """Accepted for the profile machinery; this engine has no driver keys yet."""
        del options

    # --- the frame path ----------------------------------------------------

    def animate(
        self, avatar_id: str, tracking: FaceTrackingResult, frame: VideoFrame
    ) -> RenderedFrame:
        loaded = self._loaded.get(avatar_id)
        if loaded is None:
            msg = f"avatar {avatar_id!r} is not loaded"
            raise AvatarLoadError(msg)
        if self._nets is None:
            msg = "the swap engine is not started"
            raise EngineError(msg)
        landmarks = tracking.landmarks
        if (
            not tracking.face_detected
            or landmarks is None
            or landmarks.scheme != LANDMARK_SCHEME
            or landmarks.count < 478
        ):
            return self._passthrough(frame, tracking, avatar_id)
        import cv2

        t0 = time.perf_counter()
        bgr_in = frame.pixel_format is PixelFormat.BGR24
        scene: NDArray[np.uint8]
        if bgr_in:
            scene = frame.data.copy()
        elif frame.pixel_format is PixelFormat.RGB24:
            scene = np.asarray(cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        else:
            return self._passthrough(frame, tracking, avatar_id)
        height, width = scene.shape[:2]
        points = landmarks.points.copy()
        points[:, 0] *= width
        points[:, 1] *= height
        kps = sm.five_points(points)
        options = loaded.options

        # the swap: the aligned 128 px crop with the avatar's identity
        lips_source = str(options.get("lips_source", "avatar"))
        lips_keep = 1.0 if lips_source == "driver" else float(options.get("lips_keep", 0.0))
        mouth_hole = float(options.get("mouth_hole", 1.0))
        grow = float(options.get("mask_grow", 0.03))
        feather = float(options.get("mask_feather", 0.04))
        m128 = sm.estimate_norm(
            kps,
            SWAP_SIZE,
            zoom=float(options.get("swap_zoom", 1.0)),
            shift=float(options.get("swap_shift", 0.0)),
        )
        crop = sm.warp(scene, m128, SWAP_SIZE)
        fade = float(options.get("forehead_fade", 0.0))
        mask = sm.crop_mask(
            SWAP_SIZE,
            mode=str(options.get("swap_mask", "hull")),
            points_px=points,
            m=m128,
            grow=grow,
            feather=feather,
            mouth_hole=mouth_hole,
            lips_keep=lips_keep,
            forehead_fade=fade,
        )
        # the skin colour this frame aims at: the driver's own (his crop, before
        # the swap) moved skin_match of the way to the portrait's; the swap's
        # layer and the restorer's below both land on it
        skin_match = _skin_match(options.get("skin_match", 0.0))
        skin_goal: np.ndarray | None = None
        self.last_timings["skin_ms"] = 0.0
        if skin_match is not None and loaded.portrait_lab is not None:
            tk = time.perf_counter()
            own = sm.lab_median(crop, mask, stride=1)
            skin_goal = sm.skin_goal(loaded.portrait_lab, own, skin_match)
            self.last_timings["skin_ms"] += (time.perf_counter() - tk) * 1000.0
        t1 = time.perf_counter()
        swapped = self._nets.swap(crop, loaded.latent)
        t2 = time.perf_counter()
        if skin_goal is not None:
            tk = time.perf_counter()
            median = sm.lab_median(swapped, mask, stride=1)
            loaded.skin_shift128 = sm.skin_shift(skin_goal, median, loaded.skin_shift128)
            swapped = cv2.LUT(swapped, sm.gain_lut(median, median + loaded.skin_shift128))
            self.last_timings["skin_ms"] += (time.perf_counter() - tk) * 1000.0
        sm.paste(scene, swapped, m128, mask)
        t3 = time.perf_counter()

        # the enhancer: the swapped face at 512 -- the chin shaved first when
        # asked, restored, the photograph's lips laid on -- pasted inside the oval
        t4 = t3
        self.last_timings["shave_ms"] = 0.0
        self.last_timings["lips_ms"] = 0.0
        if bool(options.get("enhance", True)):
            m512 = sm.umeyama(kps, sm.FFHQ_DST_512)
            face = sm.warp(scene, m512, ENHANCE_SIZE)
            # the oval at half size (its edge is a wide feather): a quarter of the work
            hull = np.asarray(
                cv2.resize(
                    sm.crop_mask(
                        ENHANCE_SIZE // 2,
                        mode="hull",
                        points_px=points,
                        m=m512 * 0.5,
                        grow=grow,
                        feather=feather,
                        forehead_fade=fade,
                    ),
                    (ENHANCE_SIZE, ENHANCE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                ),
                dtype=np.float32,
            )
            full = np.zeros((ENHANCE_SIZE, ENHANCE_SIZE), dtype=np.float32)
            shave = float(options.get("shave", 0.0))
            if shave > 0.0:
                ts = time.perf_counter()
                zone = sm.shave_zone(ENHANCE_SIZE, points, m512, hull) * shave
                face = sm.shave(face, zone, hull)
                full = np.maximum(full, zone)
                self.last_timings["shave_ms"] = (time.perf_counter() - ts) * 1000.0
            t4 = time.perf_counter()
            restored = self._nets.enhance(face)
            t5 = time.perf_counter()
            if skin_goal is not None:
                tk = time.perf_counter()
                median = sm.lab_median(restored, hull, stride=4)
                loaded.skin_shift512 = sm.skin_shift(skin_goal, median, loaded.skin_shift512)
                restored = cv2.LUT(restored, sm.gain_lut(median, median + loaded.skin_shift512))
                self.last_timings["skin_ms"] += (time.perf_counter() - tk) * 1000.0
            if (
                lips_source == "avatar"
                and loaded.portrait512 is not None
                and loaded.portrait_pts512 is not None
                and loaded.portrait_lab is not None
            ):
                tl = time.perf_counter()
                lips = sm.lips_from_portrait(
                    loaded.portrait512,
                    loaded.portrait_pts512,
                    sm.points_in_crop(points, m512),
                    ENHANCE_SIZE,
                )
                if lips is not None:
                    layer, band, box = lips
                    y0, y1, x0, x1 = box
                    layer = sm.lab_shift(layer, loaded.portrait_lab, restored, hull, box=box)
                    restored[y0:y1, x0:x1] = sm.blend(
                        restored[y0:y1, x0:x1], layer[y0:y1, x0:x1], band[y0:y1, x0:x1]
                    )
                    full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], band[y0:y1, x0:x1])
                self.last_timings["lips_ms"] = (time.perf_counter() - tl) * 1000.0
            mask = hull
            if lips_keep > 0.0:
                lips_ring = sm.ring_mask(
                    ENHANCE_SIZE, points, m512, sm.OUTER_LIPS, erode=0.0, feather=0.15
                )
                if lips_ring is not None:
                    mask = mask * (1.0 - lips_keep * lips_ring)
            if mouth_hole > 0.0 and lips_keep < 1.0:
                hole = sm.mouth_opening(ENHANCE_SIZE, points, m512)
                if hole is not None:
                    mask = mask * (1.0 - mouth_hole * hole)
            # the restorer's share of the face, whole where the lips and the
            # shave were laid (a part of the swap's lips would show through)
            strength = float(options.get("enhance_strength", 0.75))
            mask = mask * (strength + (1.0 - strength) * np.clip(full, 0.0, 1.0))
            sm.paste(scene, restored, m512, mask)
            self.last_timings["enhance_ms"] = (t5 - t4) * 1000.0
            self.last_timings["enhance_paste_ms"] = (time.perf_counter() - t5) * 1000.0
        else:
            self.last_timings["enhance_ms"] = 0.0
            self.last_timings["enhance_paste_ms"] = 0.0
        self.last_timings["align_ms"] = (t1 - t0) * 1000.0
        self.last_timings["swap_ms"] = (t2 - t1) * 1000.0
        self.last_timings["swap_paste_ms"] = (t3 - t2) * 1000.0
        self.last_timings["model_ms"] = (
            self.last_timings["swap_ms"] + self.last_timings["enhance_ms"]
        )
        self.last_timings["frame_ms"] = (time.perf_counter() - t0) * 1000.0

        output: NDArray[np.uint8] = (
            scene if bgr_in else np.asarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        )
        return RenderedFrame(
            data=output,
            pixel_format=frame.pixel_format,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=False,
        )

    @staticmethod
    def _passthrough(
        frame: VideoFrame, tracking: FaceTrackingResult, avatar_id: str
    ) -> RenderedFrame:
        """No face to draw over: the camera's frame, marked so the
        conditioning stage never lets it out as it is."""
        return RenderedFrame(
            data=frame.data,
            pixel_format=frame.pixel_format,
            metadata=tracking.metadata,
            avatar_id=avatar_id,
            render_timestamp=now(),
            is_placeholder=True,
            is_passthrough=True,
        )
