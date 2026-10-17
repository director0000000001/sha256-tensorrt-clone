"""Road C, step C1: a one-shot face swap on a still, before any pipeline work.

Takes a capture folder (captures/moment-*/: camera.png + tracking.npz whose last row
holds the press frame's 478 MediaPipe landmarks) and an avatar portrait, paints the
avatar's identity onto the owner's face with inswapper_128, then restores the face with
GFPGAN 1.4. Writes a contact sheet and the full frames into --out, prints the likeness
numbers (ArcFace cosine to the avatar and to the owner) and the per-network ONNX Runtime
times.

Research-terms test only: InsightFace's weights are non-commercial without a licence.

Usage (from the project root):
  .venv\\Scripts\\python.exe -I scripts/probes/swap_still.py \\
      --capture captures/moment-20261017-031408 --avatar avatars/bidzo_new/portrait.png \\
      --out C:/path/to/scratchpad/c1

The alignment follows InsightFace's norm_crop (ArcFace 112 template, +8 px for 128) and
GFPGAN's FFHQ 512 template; the swap paste follows InsightFace's INSwapper.get(paste_back)
(a box mask eroded by a tenth and blurred); the enhancer paste uses the face hull.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = PROJECT_ROOT / "models" / "downloads" / "swap"
DEFAULT_LANDMARKER = PROJECT_ROOT / "models" / "downloads" / "face_landmarker.task"

# InsightFace's ArcFace template for a 112x112 crop (face_align.arcface_dst).
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
# facexlib's FFHQ template for a 512x512 crop (FaceRestoreHelper.face_template).
FFHQ_DST_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

# MediaPipe face mesh indices (478 points with the irises).
IRIS_A, IRIS_B = 468, 473
NOSE_TIP = 1
MOUTH_A, MOUTH_B = 61, 291
# The oval + brows, for the enhancer's hull mask (the silhouette, no hair).
FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


# ----------------------------------------------------------------------------- geometry
def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (scale, rotation, translation) src -> dst, as 2x3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1] = -1.0
    rot = u @ np.diag(d) @ vt
    var_src = (src_c**2).sum() / n
    scale = (s * d).sum() / var_src
    t = dst_mean - scale * rot @ src_mean
    m = np.zeros((2, 3), dtype=np.float64)
    m[:, :2] = scale * rot
    m[:, 2] = t
    return m


def estimate_norm(kps: np.ndarray, size: int) -> np.ndarray:
    """InsightFace's estimate_norm: the ArcFace template scaled to `size`."""
    if size % 112 == 0:
        ratio = size / 112.0
        diff_x = 0.0
    elif size % 128 == 0:
        ratio = size / 128.0
        diff_x = 8.0 * ratio
    else:
        raise ValueError(f"size must be a multiple of 112 or 128, got {size}")
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    return umeyama(kps, dst)


def warp(img: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.warpAffine(img, m.astype(np.float32), (size, size), borderValue=0)


def five_points(points_px: np.ndarray) -> np.ndarray:
    """ArcFace's five points from the 478 MediaPipe landmarks (image left to right)."""
    eyes = np.stack([points_px[IRIS_A, :2], points_px[IRIS_B, :2]])
    eyes = eyes[np.argsort(eyes[:, 0])]
    mouth = np.stack([points_px[MOUTH_A, :2], points_px[MOUTH_B, :2]])
    mouth = mouth[np.argsort(mouth[:, 0])]
    nose = points_px[NOSE_TIP, :2]
    return np.stack([eyes[0], eyes[1], nose, mouth[0], mouth[1]]).astype(np.float32)


# ----------------------------------------------------------------------------- landmarks
def landmarks_from_capture(folder: Path) -> np.ndarray:
    import cv2

    frame = cv2.imread(str(folder / "camera.png"), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"{folder / 'camera.png'} could not be read")
    z = np.load(folder / "tracking.npz", allow_pickle=True)
    pts = np.asarray(z["landmarks"][-1], dtype=np.float32).copy()
    h, w = frame.shape[:2]
    pts[:, 0] *= w
    pts[:, 1] *= h
    pose = np.asarray(z["pose"][-1], dtype=np.float32)
    return frame, pts, pose


def landmarks_from_still(bgr: np.ndarray, landmarker: Path) -> np.ndarray:
    from sha256.core.enums import PixelFormat
    from sha256.core.types import FrameMetadata, VideoFrame
    from sha256.tracking.mediapipe_tracker import MediaPipeFaceTracker

    tracker = MediaPipeFaceTracker(
        model_path=landmarker, smoothing=0.0, mouth_smoothing=0.0, running_mode="image"
    )
    tracker.start()
    try:
        result = tracker.process(
            VideoFrame(
                data=np.ascontiguousarray(bgr),
                pixel_format=PixelFormat.BGR24,
                metadata=FrameMetadata(frame_id=0, capture_timestamp=0.0),
            )
        )
    finally:
        tracker.stop()
    if not result.face_detected or result.landmarks is None:
        raise SystemExit("no face found in the portrait")
    pts = np.asarray(result.landmarks.points, dtype=np.float32).copy()
    h, w = bgr.shape[:2]
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts


# ----------------------------------------------------------------------------- models
class SwapModels:
    def __init__(self, models_dir: Path, provider: str) -> None:
        import contextlib

        import onnx
        import onnxruntime as ort

        # torch first: its lib folder carries the CUDA 12 / cuDNN 9 DLLs ORT's CUDA EP needs.
        import torch  # noqa: F401
        from onnx import numpy_helper

        with contextlib.suppress(Exception):  # older ORT, or no nvidia pip packages
            ort.preload_dlls()
        if provider == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.arcface = ort.InferenceSession(
            str(models_dir / "buffalo_l" / "w600k_r50.onnx"), opts, providers=providers
        )
        self.swapper = ort.InferenceSession(
            str(models_dir / "inswapper_128.onnx"), opts, providers=providers
        )
        self.enhancer = ort.InferenceSession(
            str(models_dir / "gfpgan_1.4.onnx"), opts, providers=providers
        )
        self.providers = {
            "arcface": self.arcface.get_providers()[0],
            "inswapper": self.swapper.get_providers()[0],
            "gfpgan": self.enhancer.get_providers()[0],
        }
        # inswapper's `emap`: the 512-d ArcFace embedding is projected by it before the swap.
        # InsightFace's INSwapper reads it as the graph's LAST initializer (a 512x512 matrix).
        model = onnx.load(str(models_dir / "inswapper_128.onnx"), load_external_data=False)
        emap = None
        for init in model.graph.initializer:
            if list(init.dims) == [512, 512]:
                emap = numpy_helper.to_array(init)
        if emap is None:
            emap = numpy_helper.to_array(model.graph.initializer[-1])
        if emap.shape != (512, 512):
            raise SystemExit(
                f"inswapper_128.onnx: no 512x512 emap found (last initializer {emap.shape})"
            )
        self.emap = emap.astype(np.float32)
        self.arc_in = self.arcface.get_inputs()[0].name
        swap_inputs = self.swapper.get_inputs()
        self.swap_target = next(i.name for i in swap_inputs if len(i.shape) == 4)
        self.swap_source = next(i.name for i in swap_inputs if len(i.shape) == 2)
        self.enh_in = self.enhancer.get_inputs()[0].name
        self.enh_shape = self.enhancer.get_inputs()[0].shape

    def embed(self, bgr112: np.ndarray) -> np.ndarray:
        rgb = bgr112[:, :, ::-1].astype(np.float32)
        blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None]
        out = self.arcface.run(None, {self.arc_in: np.ascontiguousarray(blob)})[0][0]
        return out / np.linalg.norm(out)

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        lat = embedding.reshape(1, -1) @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def swap(self, bgr128: np.ndarray, latent: np.ndarray) -> np.ndarray:
        rgb = bgr128[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        pred = self.swapper.run(None, {self.swap_target: blob, self.swap_source: latent})[0]
        out = pred.transpose(0, 2, 3, 1)[0]
        return np.clip(255.0 * out, 0, 255).astype(np.uint8)[:, :, ::-1].copy()

    def enhance(self, bgr512: np.ndarray) -> np.ndarray:
        rgb = bgr512[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None])
        out = self.enhancer.run(None, {self.enh_in: blob})[0][0]
        out = (np.clip(out, -1.0, 1.0) + 1.0) / 2.0
        return (out.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)[:, :, ::-1].copy()


def median_ms(fn, warm: int = 3, runs: int = 10) -> float:
    for _ in range(warm):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ----------------------------------------------------------------------------- paste
def paste_box(frame: np.ndarray, crop: np.ndarray, m: np.ndarray) -> np.ndarray:
    """InsightFace's paste_back: the crop's box, eroded by a tenth, blurred, blended."""
    import cv2

    h, w = frame.shape[:2]
    im = cv2.invertAffineTransform(m.astype(np.float32))
    warped = cv2.warpAffine(crop, im, (w, h), borderValue=0)
    white = np.full(crop.shape[:2], 255.0, dtype=np.float32)
    mask = cv2.warpAffine(white, im, (w, h), borderValue=0)
    mask[mask > 20] = 255
    ys, xs = np.where(mask == 255)
    if ys.size == 0:
        return frame
    mask_size = int(np.sqrt((ys.max() - ys.min()) * (xs.max() - xs.min())))
    k = max(mask_size // 10, 10)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    k = max(mask_size // 20, 5)
    mask = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), 0)
    mask = (mask / 255.0)[:, :, None]
    return (mask * warped + (1.0 - mask) * frame).round().astype(np.uint8)


def paste_hull(
    frame: np.ndarray, crop: np.ndarray, m: np.ndarray, points_px: np.ndarray, grow: float
) -> np.ndarray:
    """The crop pasted inside the face's silhouette (oval), grown and feathered."""
    import cv2

    h, w = frame.shape[:2]
    im = cv2.invertAffineTransform(m.astype(np.float32))
    warped = cv2.warpAffine(crop, im, (w, h), borderValue=0)
    oval = points_px[FACE_OVAL, :2].astype(np.int32)
    hull = cv2.convexHull(oval)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    face_h = float(oval[:, 1].max() - oval[:, 1].min())
    k = max(int(face_h * grow), 1)
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    # keep the enhancer inside the crop's own box
    white = np.full(crop.shape[:2], 255.0, dtype=np.float32)
    box = cv2.warpAffine(white, im, (w, h), borderValue=0)
    box = cv2.erode(box, np.ones((9, 9), np.uint8))
    mask = np.minimum(mask.astype(np.float32), box)
    f = max(int(face_h * 0.04), 3)
    mask = cv2.GaussianBlur(mask, (2 * f + 1, 2 * f + 1), 0)
    mask = (mask / 255.0)[:, :, None]
    return (mask * warped + (1.0 - mask) * frame).round().astype(np.uint8)


# ----------------------------------------------------------------------------- sheet
def label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def up(img: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------------- main
def run_one(
    models: SwapModels,
    capture: Path,
    avatar: Path,
    landmarker: Path,
    out_dir: Path,
    swap_mask: str,
    hull_grow: float,
    debug_points: bool,
    timing_runs: int,
    dump_inputs: Path | None = None,
) -> dict:
    import cv2

    frame, pts, pose = landmarks_from_capture(capture)
    kps = five_points(pts)
    portrait = cv2.imread(str(avatar), cv2.IMREAD_COLOR)
    if portrait is None:
        raise SystemExit(f"{avatar} could not be read")
    av_pts = landmarks_from_still(portrait, landmarker)
    av_kps = five_points(av_pts)

    # identity: the avatar's embedding (once per avatar in the app, at load)
    av112 = warp(portrait, estimate_norm(av_kps, 112), 112)
    av_emb = models.embed(av112)
    latent = models.latent(av_emb)
    me112 = warp(frame, estimate_norm(kps, 112), 112)
    me_emb = models.embed(me112)

    # swap
    m128 = estimate_norm(kps, 128)
    in128 = warp(frame, m128, 128)
    out128 = models.swap(in128, latent)
    swapped = (
        paste_box(frame, out128, m128)
        if swap_mask == "box"
        else paste_hull(frame, out128, m128, pts, hull_grow)
    )

    # enhance
    m512 = umeyama(kps, FFHQ_DST_512)
    enh_in = warp(swapped, m512, 512)
    enh_out = models.enhance(enh_in)
    final = paste_hull(swapped, enh_out, m512, pts, hull_grow)

    if dump_inputs is not None and not dump_inputs.exists():
        # the real tensors for the TensorRT probe (fp16 vs fp32 on a real face, not noise)
        np.savez(
            dump_inputs, in128=in128, latent=latent, out128=out128, enh_in=enh_in, enh_out=enh_out
        )

    # numbers
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    sw_emb = models.embed(warp(swapped, estimate_norm(kps, 112), 112))
    fi_emb = models.embed(warp(final, estimate_norm(kps, 112), 112))
    numbers = {
        "capture": capture.name,
        "avatar": str(avatar.relative_to(PROJECT_ROOT))
        if avatar.is_relative_to(PROJECT_ROOT)
        else str(avatar),
        "yaw": round(float(pose[0]), 1),
        "cos_me_avatar": round(cos(me_emb, av_emb), 3),
        "cos_swap_avatar": round(cos(sw_emb, av_emb), 3),
        "cos_swap_me": round(cos(sw_emb, me_emb), 3),
        "cos_final_avatar": round(cos(fi_emb, av_emb), 3),
        "cos_final_me": round(cos(fi_emb, me_emb), 3),
    }

    # files
    tag = f"{capture.name}_{avatar.parent.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"full_{tag}.png"), final)
    cv2.imwrite(str(out_dir / f"swaponly_{tag}.png"), swapped)
    orig512 = warp(frame, m512, 512)
    sw512 = warp(swapped, m512, 512)
    fi512 = warp(final, m512, 512)
    av512 = warp(portrait, umeyama(av_kps, FFHQ_DST_512), 512)
    row1 = np.hstack(
        [
            label(orig512, f"original yaw {numbers['yaw']:+.0f}"),
            label(sw512, f"swap only  cos av {numbers['cos_swap_avatar']:.2f}"),
            label(fi512, f"swap+GFPGAN  cos av {numbers['cos_final_avatar']:.2f}"),
            label(av512, f"avatar {avatar.parent.name}"),
        ]
    )
    row2 = np.hstack(
        [
            label(up(in128, 512), "128 in"),
            label(up(out128, 512), "128 swapped"),
            label(enh_in, "GFPGAN in 512"),
            label(enh_out, "GFPGAN out 512"),
        ]
    )
    cv2.imwrite(str(out_dir / f"sheet_{tag}.png"), np.vstack([row1, row2]))
    if debug_points:
        dbg = frame.copy()
        for x, y in kps:
            cv2.circle(dbg, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.imwrite(str(out_dir / f"points_{capture.name}.png"), dbg)

    # timings (ORT, not TensorRT; indicative only)
    n = timing_runs
    numbers["ms_arcface"] = round(median_ms(lambda: models.embed(av112), 1, n), 2)
    numbers["ms_inswapper"] = round(median_ms(lambda: models.swap(in128, latent), 1, n), 2)
    numbers["ms_gfpgan"] = round(median_ms(lambda: models.enhance(enh_in), 1, n), 2)
    return numbers


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--capture",
        action="append",
        required=True,
        type=Path,
        help="captures/moment-* folder (repeatable)",
    )
    p.add_argument(
        "--avatar", action="append", required=True, type=Path, help="avatar portrait (repeatable)"
    )
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--swap-mask", choices=("box", "hull"), default="box")
    p.add_argument(
        "--hull-grow", type=float, default=0.03, help="hull dilation as a share of the face height"
    )
    p.add_argument("--debug-points", action="store_true")
    p.add_argument("--timing-runs", type=int, default=10, help="runs per network for the median ms")
    p.add_argument(
        "--dump-inputs", type=Path, default=None, help="save the first pair's real tensors (npz)"
    )
    args = p.parse_args(argv)

    models = SwapModels(args.models, args.provider)
    print("providers:", models.providers)
    print("gfpgan input:", models.enh_in, models.enh_shape)
    rows = []
    for capture in args.capture:
        for avatar in args.avatar:
            rows.append(
                run_one(
                    models,
                    capture,
                    avatar,
                    args.landmarker,
                    args.out,
                    args.swap_mask,
                    args.hull_grow,
                    args.debug_points,
                    args.timing_runs,
                    args.dump_inputs,
                )
            )
            r = rows[-1]
            print(
                f"{r['capture']} x {Path(r['avatar']).parent.name}: yaw {r['yaw']:+.1f} | "
                f"me~avatar {r['cos_me_avatar']:.3f} | swap: ~avatar {r['cos_swap_avatar']:.3f}"
                f" ~me {r['cos_swap_me']:.3f} | final: ~avatar {r['cos_final_avatar']:.3f}"
                f" ~me {r['cos_final_me']:.3f} | ms arcface {r['ms_arcface']}"
                f" inswapper {r['ms_inswapper']} gfpgan {r['ms_gfpgan']}"
            )
    (args.out / "numbers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
