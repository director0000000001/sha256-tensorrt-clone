"""The one-shot face swap's networks, geometry and paste (road C, 2026-10-17).

Three ONNX networks under ``models/downloads/swap``: InsightFace's ArcFace
(``buffalo_l/w600k_r50.onnx``, the avatar's identity as a 512-vector, once at
load), ``inswapper_128.onnx`` (the driver's aligned 128 px face re-drawn with
that identity, every frame) and ``gfpgan_1.4.onnx`` (the soft 128 px face
restored at 512, every frame). Each runs as a TensorRT plan built on this GPU
at the first start, like LivePortrait's (``liveportrait_tensorrt``).

TensorRT 11 builds strongly typed networks only -- no FP16 flag; the plan's
precision is the graph's own dtypes -- so the fp16 plans are built from fp16
ONNX files this module writes beside the fp32 ones. The public fp16 converters
(onnxconverter_common, onnxruntime.transformers.float16) mis-wire a partially
converted graph, so :func:`to_fp16` is our own: every float tensor in fp16
except the op types in :data:`KEEP_FP32` and the nodes
``models/metadata/swap_fp16_blocks.json`` names (the instance-norm variances
of inswapper and the StyleGAN2 demodulations of GFPGAN overflow fp16 on a real
face: up to 3e6) plus their consumers, Casts at every boundary, the graph's
inputs and outputs float32. Measured 2026-10-17 on the RTX 5070 Ti: inswapper
2.9 ms, GFPGAN 5.7 ms, within 2-3 grey levels of the fp32 outputs.

Licence: InsightFace's weights are for non-commercial research unless licensed
(docs/development/ROADMAP.md, "Road C"); GFPGAN is Apache-2.0.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sha256.utils.logging import get_logger

__all__ = [
    "ARCFACE_DST",
    "BEARD_THRESHOLD",
    "FACE_OVAL",
    "FFHQ_DST_512",
    "INNER_LIPS",
    "KEEP_FP32",
    "LIP_TRIANGLES",
    "ONNX_FILES",
    "OUTER_LIPS",
    "SWAP_BLOCKS_RELATIVE_PATH",
    "SWAP_RELATIVE_DIR",
    "SwapNets",
    "TrtNet",
    "beard_signal",
    "blend",
    "blob_enhance",
    "blob_swap",
    "crop_mask",
    "ensure_plans",
    "estimate_norm",
    "five_points",
    "fp16_onnx_path",
    "fp16_source",
    "gain_lut",
    "lab_median",
    "lab_shift",
    "lips_from_portrait",
    "load_emap",
    "mouth_opening",
    "paste",
    "plan_path",
    "points_in_crop",
    "ring_mask",
    "shave",
    "shave_zone",
    "skin_goal",
    "skin_shift",
    "to_fp16",
    "toposort",
    "umeyama",
    "unblob_enhance",
    "unblob_swap",
    "warp",
    "warp_triangles",
]

_LOG = get_logger(__name__)

SWAP_RELATIVE_DIR = Path("downloads") / "swap"
SWAP_BLOCKS_RELATIVE_PATH = Path("metadata") / "swap_fp16_blocks.json"
ONNX_FILES: dict[str, Path] = {
    "arcface": Path("buffalo_l") / "w600k_r50.onnx",
    "inswapper": Path("inswapper_128.onnx"),
    "gfpgan": Path("gfpgan_1.4.onnx"),
}
EMAP_CACHE_NAME = "inswapper_emap.npy"

# Op types that stay fp32 in every fp16 conversion (the converters' usual list:
# index arithmetic and resampling that fp16 would coarsen).
KEEP_FP32 = frozenset(
    {"Resize", "Upsample", "Range", "CumSum", "Min", "Max", "NonMaxSuppression", "TopK", "RoiAlign"}
)
FP16_MAX = 65504.0
FP16_TINY = 6.0e-8

# InsightFace's ArcFace template for a 112x112 crop (face_align.arcface_dst):
# eyes, nose tip, mouth corners, image left to right.
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
# facexlib's FFHQ template for GFPGAN's 512x512 crop (FaceRestoreHelper.face_template).
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

# MediaPipe face mesh indices (the 478-point scheme with the irises).
IRIS_A, IRIS_B = 468, 473
NOSE_TIP = 1
MOUTH_A, MOUTH_B = 61, 291
# The inner lips, in ring order: the mouth's opening. His look 2026-10-17 ("the
# teeth on a smile"): the restorer invents teeth, so the opening stays the
# driver's own -- a hole in the paste's mask, nothing when the lips are shut.
INNER_LIPS = (
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
)
# The outer lips, in ring order: the whole mouth. inswapper draws the mouth
# part-way toward the portrait's (his talk frames: a third of the opening),
# so the lips too can stay the driver's own (``lips_keep``).
OUTER_LIPS = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
)
# The face oval: the silhouette from the forehead (10) round the jaw (152);
# the paste's hull, so nothing of the avatar lands on the driver's hair.
FACE_OVAL = (
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
)


# --- geometry ----------------------------------------------------------------


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The least-squares similarity (scale, rotation, translation) taking
    ``src`` points onto ``dst``, as a 2x3 affine (skimage's ``_umeyama`` with
    the scale, which InsightFace's alignment uses)."""
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


def estimate_norm(
    kps: np.ndarray, size: int, *, zoom: float = 1.0, shift: float = 0.0
) -> np.ndarray:
    """InsightFace's ``estimate_norm``: the ArcFace template scaled to ``size``
    (112 multiples as they are; 128 multiples shifted 8 px right, the way
    inswapper's crops were made). ``zoom`` < 1 draws the face smaller in the
    crop (more chin and forehead around it), ``shift`` > 0 moves it up by that
    share of the crop (more chin below): both leave the training alignment,
    so each is measured before it is used (the chin at an open mouth)."""
    if size % 112 == 0:
        ratio = size / 112.0
        diff_x = 0.0
    elif size % 128 == 0:
        ratio = size / 128.0
        diff_x = 8.0 * ratio
    else:
        msg = f"size must be a multiple of 112 or 128, got {size}"
        raise ValueError(msg)
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    if zoom != 1.0:
        centre = np.array([size / 2.0, size / 2.0], dtype=np.float32)
        dst = (dst - centre) * float(zoom) + centre
    if shift:
        dst[:, 1] -= float(shift) * size
    return umeyama(kps, dst)


def warp(image: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    """``image`` under the 2x3 affine ``m`` as a ``size`` x ``size`` crop."""
    import cv2

    return np.asarray(cv2.warpAffine(image, m.astype(np.float32), (size, size), borderValue=0))


def five_points(points_px: np.ndarray) -> np.ndarray:
    """ArcFace's five points from the 478 MediaPipe landmarks in pixels: the
    iris centres, the nose tip, the mouth corners -- image left to right, the
    template's order, whichever way the head is turned."""
    eyes = np.stack([points_px[IRIS_A, :2], points_px[IRIS_B, :2]])
    eyes = eyes[np.argsort(eyes[:, 0])]
    mouth = np.stack([points_px[MOUTH_A, :2], points_px[MOUTH_B, :2]])
    mouth = mouth[np.argsort(mouth[:, 0])]
    nose = points_px[NOSE_TIP, :2]
    return np.stack([eyes[0], eyes[1], nose, mouth[0], mouth[1]]).astype(np.float32)


# --- the networks' tensors -----------------------------------------------------


def blob_swap(bgr128: np.ndarray) -> np.ndarray:
    """inswapper's input: RGB in 0..1, NCHW."""
    rgb = bgr128[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def unblob_swap(pred: np.ndarray) -> np.ndarray:
    out = pred.transpose(0, 2, 3, 1)[0]
    return np.asarray(np.clip(255.0 * out, 0, 255).astype(np.uint8)[:, :, ::-1].copy())


def blob_enhance(bgr512: np.ndarray) -> np.ndarray:
    """GFPGAN's input: RGB in -1..1, NCHW."""
    rgb = bgr512[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None])


def unblob_enhance(pred: np.ndarray) -> np.ndarray:
    out = (np.clip(pred[0], -1.0, 1.0) + 1.0) / 2.0
    return np.asarray((out.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)[:, :, ::-1].copy())


def blob_arcface(bgr112: np.ndarray) -> np.ndarray:
    """ArcFace's input: RGB, (x - 127.5) / 127.5, NCHW."""
    rgb = bgr112[:, :, ::-1].astype(np.float32)
    return np.ascontiguousarray(((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None])


# --- the paste -----------------------------------------------------------------


def crop_mask(
    size: int,
    *,
    mode: str = "hull",
    points_px: np.ndarray | None = None,
    m: np.ndarray | None = None,
    grow: float = 0.03,
    feather: float = 0.04,
    mouth_hole: float = 0.0,
    lips_keep: float = 0.0,
    forehead_fade: float = 0.0,
) -> np.ndarray:
    """A float32 mask in the crop's own space (``size`` x ``size``, 0..1).

    ``"box"``: InsightFace's paste -- the whole crop, eroded by a tenth of its
    side and blurred by a twentieth. ``"hull"``: the driver's face oval
    (``points_px``, in the frame) mapped into the crop by ``m``, grown by
    ``grow`` of the face's height and feathered by ``feather`` of it, cut to
    the crop's eroded box -- the avatar stays inside the driver's skin, off
    the hair and the background. ``mouth_hole`` (0..1) takes that share of the
    mask away inside the inner lips, so the driver's own teeth and tongue
    show through an open mouth; ``lips_keep`` (0..1) does the same inside the
    outer lips, the whole mouth then the driver's own. ``forehead_fade``
    (0..1) lets the mask fall off from the brow line up to the oval's top,
    where it is ``1 - forehead_fade`` of itself: the driver's own forehead
    shows through (his look 2026-10-18: the avatar's forehead came paler and
    flatter than his, a band at the hairline; no identity sits there)."""
    import cv2

    box: Any
    if mode == "box":
        box = np.full((size, size), 255.0, dtype=np.float32)
        k = max(size // 10, 10)
        # the border counts as black, or a full box never erodes
        box = cv2.erode(
            box,
            np.ones((k, k), np.uint8),
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        k = max(size // 20, 5)
        box = cv2.GaussianBlur(box, (2 * k + 1, 2 * k + 1), 0)
        return np.asarray(box / 255.0, dtype=np.float32)
    if mode != "hull":
        msg = f"mask mode must be 'box' or 'hull', not {mode!r}"
        raise ValueError(msg)
    if points_px is None or m is None:
        msg = "the hull mask needs the landmarks and the crop's affine"
        raise ValueError(msg)
    oval = points_px[list(FACE_OVAL), :2].astype(np.float64)
    ones = np.ones((oval.shape[0], 1))
    in_crop = (np.hstack([oval, ones]) @ np.asarray(m, dtype=np.float64).T).astype(np.int32)
    hull: Any = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(in_crop), 255)
    face_h = float(in_crop[:, 1].max() - in_crop[:, 1].min())
    k = max(int(face_h * grow), 1)
    if k > 1:
        hull = cv2.dilate(hull, np.ones((k, k), np.uint8), iterations=1)
    out: Any = hull.astype(np.float32)
    # inside the crop's own edge, so the feather never reads the border
    edge = max(size // 32, 2)
    out[:edge, :] = 0.0
    out[-edge:, :] = 0.0
    out[:, :edge] = 0.0
    out[:, -edge:] = 0.0
    f = max(int(face_h * feather), 2)
    out = cv2.GaussianBlur(out, (2 * f + 1, 2 * f + 1), 0)
    out = out / 255.0
    if forehead_fade > 0.0:
        brows = points_px[list(BROW_LINE), :2].astype(np.float64)
        brow_y = float(
            (np.hstack([brows, np.ones((brows.shape[0], 1))]) @ np.asarray(m).T)[:, 1].mean()
        )
        top_y = float(in_crop[:, 1].min())
        if brow_y - top_y > 2.0:
            ys = np.arange(size, dtype=np.float32)
            ramp = np.clip((brow_y - ys) / (brow_y - top_y), 0.0, 1.0)
            out = out * (1.0 - float(forehead_fade) * ramp)[:, None]
    if lips_keep > 0.0:
        lips = ring_mask(size, points_px, m, OUTER_LIPS, erode=0.0, feather=0.15)
        if lips is not None:
            out = out * (1.0 - float(lips_keep) * lips)
    if mouth_hole > 0.0 and lips_keep < 1.0:
        hole = mouth_opening(size, points_px, m)
        if hole is not None:
            out = out * (1.0 - float(mouth_hole) * hole)
    return np.asarray(out, dtype=np.float32)


def ring_mask(
    size: int,
    points_px: np.ndarray,
    m: np.ndarray,
    ring: tuple[int, ...],
    *,
    erode: float = 0.0,
    feather: float = 0.2,
) -> np.ndarray | None:
    """A landmark ring (in the frame) as a soft float32 mask in the crop's
    space: 1 inside, eroded by ``erode`` of the ring's height, feathered by
    ``feather`` of it; ``None`` when the ring has no height to speak of."""
    import cv2

    pts = points_px[list(ring), :2].astype(np.float64)
    ones = np.ones((pts.shape[0], 1))
    in_crop = np.hstack([pts, ones]) @ np.asarray(m, dtype=np.float64).T
    height = float(in_crop[:, 1].max() - in_crop[:, 1].min())
    if height < 3.0:
        return None
    area: Any = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(area, [np.round(in_crop).astype(np.int32)], 255)
    k = max(int(height * erode), 1) if erode > 0.0 else 1
    if k > 1:
        area = cv2.erode(area, np.ones((k, k), np.uint8), iterations=1)
    if not area.any():
        return None
    f = max(int(height * feather), 1)
    soft: Any = cv2.GaussianBlur(area.astype(np.float32), (2 * f + 1, 2 * f + 1), 0)
    return np.asarray(soft / 255.0, dtype=np.float32)


def mouth_opening(size: int, points_px: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    """The inner lips' ring: the mouth's opening, eroded a little so the lips'
    edge keeps the avatar's colour; ``None`` when the lips are shut."""
    return ring_mask(size, points_px, m, INNER_LIPS, erode=0.15, feather=0.2)


def paste(
    frame: np.ndarray, crop: np.ndarray, m: np.ndarray, mask: np.ndarray, *, strength: float = 1.0
) -> None:
    """``crop`` (drawn under the affine ``m``) blended into ``frame`` IN PLACE
    through ``mask`` (the crop's space, 0..1) -- only the rows and columns the
    crop lands on are touched, a few milliseconds less than the whole frame."""
    import cv2

    h, w = frame.shape[:2]
    size = crop.shape[0]
    im = cv2.invertAffineTransform(np.asarray(m, dtype=np.float32))
    corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], dtype=np.float64)
    back = corners @ np.asarray(im, dtype=np.float64).T
    x0 = int(max(np.floor(back[:, 0].min()) - 1, 0))
    y0 = int(max(np.floor(back[:, 1].min()) - 1, 0))
    x1 = int(min(np.ceil(back[:, 0].max()) + 1, w))
    y1 = int(min(np.ceil(back[:, 1].max()) + 1, h))
    if x1 <= x0 or y1 <= y0:
        return
    shifted = im.copy()
    shifted[0, 2] -= x0
    shifted[1, 2] -= y0
    roi = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(crop, shifted, roi, flags=cv2.INTER_LINEAR, borderValue=0)
    weight = cv2.warpAffine(mask, shifted, roi, flags=cv2.INTER_LINEAR, borderValue=0.0)
    if strength != 1.0:
        weight = weight * float(strength)
    region = frame[y0:y1, x0:x1]
    # cv2.blendLinear: (warped * w + region * (1 - w)) in uint8, under a
    # millisecond for a 600 px face; numpy's float broadcast took nine.
    region[...] = cv2.blendLinear(warped, np.ascontiguousarray(region), weight, 1.0 - weight)


# --- the lips from the photograph, the chin shaved ---------------------------------

# The lips as four rows from the left corner (61 / 78) to the right (291 / 308):
# the outer and the inner edge of the upper and of the lower lip, 11 points each.
UPPER_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
UPPER_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
LOWER_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
LOWER_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
NOSE_BASE = 2
CHIN = 152
FOREHEAD_TOP = 10
# The upper edge of the brows, left and right: the line the forehead fade starts from.
BROW_LINE = (105, 66, 107, 334, 296, 336)


def _lip_triangles() -> tuple[tuple[int, int, int], ...]:
    """The lip band as triangles between the outer and the inner edge."""
    tris: list[tuple[int, int, int]] = []
    for outer, inner in ((UPPER_OUTER, UPPER_INNER), (LOWER_OUTER, LOWER_INNER)):
        for i in range(len(outer) - 1):
            tris.append((outer[i], outer[i + 1], inner[i]))
            tris.append((outer[i + 1], inner[i + 1], inner[i]))
    return tuple(tris)


LIP_TRIANGLES = _lip_triangles()


def points_in_crop(points_px: np.ndarray, m: np.ndarray) -> np.ndarray:
    """All landmarks (frame pixels) under the crop's affine, as (N, 2) float32."""
    pts = np.asarray(points_px[:, :2], dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    return (np.hstack([pts, ones]) @ np.asarray(m, dtype=np.float64).T).astype(np.float32)


def blend(base: np.ndarray, layer: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``layer`` over ``base`` by the float32 ``weight`` (0..1), in uint8."""
    import cv2

    w = np.ascontiguousarray(weight, dtype=np.float32)
    return np.asarray(cv2.blendLinear(layer, base, w, 1.0 - w))


def warp_triangles(
    src: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    triangles: tuple[tuple[int, int, int], ...],
    size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """``src``'s picture inside each triangle of ``src_pts`` moved onto the
    same triangle of ``dst_pts``: a ``size`` x ``size`` layer, the float32
    coverage (1 where a triangle landed) and the box (y0, y1, x0, x1) that
    holds it all. A triangle without area on either side (the shut lips) is
    skipped."""
    import cv2

    layer = np.zeros((size, size, 3), dtype=np.uint8)
    cover = np.zeros((size, size), dtype=np.uint8)
    src_h, src_w = src.shape[:2]
    index = np.asarray(triangles, dtype=np.intp)
    s_all = np.asarray(src_pts, dtype=np.float32)[index]  # (T, 3, 2)
    d_all = np.asarray(dst_pts, dtype=np.float32)[index]

    def areas(t: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.abs(
                (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
            )
        )

    keep = (areas(s_all) >= 1.0) & (areas(d_all) >= 1.0)
    s_min = np.floor(s_all.min(axis=1)).astype(np.int64)
    s_max = np.ceil(s_all.max(axis=1)).astype(np.int64) + 1
    d_min = np.floor(d_all.min(axis=1)).astype(np.int64)
    d_max = np.ceil(d_all.max(axis=1)).astype(np.int64) + 1
    box = [size, 0, size, 0]
    for i in np.flatnonzero(keep):
        sx0, sy0 = max(int(s_min[i, 0]), 0), max(int(s_min[i, 1]), 0)
        sx1, sy1 = min(int(s_max[i, 0]), src_w), min(int(s_max[i, 1]), src_h)
        dx0, dy0 = max(int(d_min[i, 0]), 0), max(int(d_min[i, 1]), 0)
        dx1, dy1 = min(int(d_max[i, 0]), size), min(int(d_max[i, 1]), size)
        if sx1 <= sx0 or sy1 <= sy0 or dx1 <= dx0 or dy1 <= dy0:
            continue
        s_local = s_all[i] - np.array([sx0, sy0], dtype=np.float32)
        d_local = d_all[i] - np.array([dx0, dy0], dtype=np.float32)
        affine = cv2.getAffineTransform(s_local, d_local)
        patch = cv2.warpAffine(
            src[sy0:sy1, sx0:sx1],
            affine,
            (dx1 - dx0, dy1 - dy0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = np.zeros((dy1 - dy0, dx1 - dx0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(d_local).astype(np.int32), 255)
        inside = mask > 0
        region = layer[dy0:dy1, dx0:dx1]
        region[inside] = patch[inside]
        cover[dy0:dy1, dx0:dx1] = np.maximum(cover[dy0:dy1, dx0:dx1], mask)
        box = [min(box[0], dy0), max(box[1], dy1), min(box[2], dx0), max(box[3], dx1)]
    if box[1] <= box[0]:
        box = [0, 0, 0, 0]
    return layer, cover.astype(np.float32) / 255.0, (box[0], box[1], box[2], box[3])


def lab_median(image: np.ndarray, mask: np.ndarray, stride: int = 2) -> np.ndarray:
    """The Lab median of ``image`` inside ``mask``, on every ``stride``-th pixel."""
    import cv2

    # the strided picture converted, not the whole one (0.05 ms at 512 / 4, not 0.5)
    sub = np.ascontiguousarray(image[::stride, ::stride])
    lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = mask[::stride, ::stride] > 0.5
    if inside.sum() < 16:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(np.median(lab[inside], axis=0), dtype=np.float32)


def lab_shift(
    image: np.ndarray,
    src_median: np.ndarray,
    dst_ref: np.ndarray,
    dst_mask: np.ndarray,
    *,
    lightness: float = 1.0,
    box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """``image`` moved from the light its ``src_median`` (Lab, see
    :func:`lab_median`) was taken in into ``dst_ref``'s inside ``dst_mask``:
    the two medians differ by the lighting, and that difference is added
    (``lightness`` scales the L part); only ``box`` (y0, y1, x0, x1) is touched."""
    import cv2

    dst_median = lab_median(dst_ref, dst_mask, stride=4)
    if not dst_median.any():
        return image
    shift = dst_median - src_median
    shift[0] *= float(lightness)
    y0, y1, x0, x1 = box if box is not None else (0, image.shape[0], 0, image.shape[1])
    if y1 <= y0 or x1 <= x0:
        return image
    part = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32) + shift
    out = image.copy()
    out[y0:y1, x0:x1] = cv2.cvtColor(np.clip(part, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


# The most a skin match may move a channel, in 8-bit Lab: ~16 L*, 20 a*, 20 b*.
SKIN_SHIFT_LIMIT = (40.0, 20.0, 20.0)
# The gain a channel may get from a skin match (the lookup table saturates above).
SKIN_GAIN_LIMIT = (0.7, 1.35)


def skin_goal(portrait_skin: np.ndarray, median: np.ndarray, strength: float) -> np.ndarray | None:
    """The Lab skin colour a frame aims at: the face's skin ``median``
    (:func:`lab_median`) moved ``strength`` (0..1) of the way to
    ``portrait_skin`` (the portrait's); ``None`` when there was no skin to
    read. Every layer of the frame then lands on this one colour
    (:func:`skin_shift`), so ``strength`` means what it says whatever the
    layers' own blend."""
    median = np.asarray(median, dtype=np.float32)
    if not median.any():
        return None
    portrait = np.asarray(portrait_skin, dtype=np.float32)
    return np.asarray(median + float(strength) * (portrait - median), dtype=np.float32)


def skin_shift(
    goal: np.ndarray,
    median: np.ndarray,
    previous: np.ndarray | None = None,
    *,
    smoothing: float = 0.2,
) -> np.ndarray:
    """The Lab shift that lands a layer's skin ``median`` on ``goal``
    (:func:`skin_goal`), each channel held within :data:`SKIN_SHIFT_LIMIT`,
    and eased from ``previous`` (an exponential average, ``smoothing`` of the
    way per call) so the face does not flicker from frame to frame. With no
    skin to read (a zero median), ``previous`` stands."""
    median = np.asarray(median, dtype=np.float32)
    if not median.any():
        return previous if previous is not None else np.zeros(3, dtype=np.float32)
    limit = np.asarray(SKIN_SHIFT_LIMIT, dtype=np.float32)
    target = np.clip(np.asarray(goal, dtype=np.float32) - median, -limit, limit)
    if previous is None:
        return np.asarray(target, dtype=np.float32)
    return np.asarray(previous + float(smoothing) * (target - previous), dtype=np.float32)


def gain_lut(lab_from: np.ndarray, lab_to: np.ndarray) -> np.ndarray:
    """A per-channel BGR lookup table (256 x 1 x 3, for ``cv2.LUT``) that
    moves the colour ``lab_from`` to ``lab_to`` (8-bit Lab) by a gain per
    channel, each within :data:`SKIN_GAIN_LIMIT`: an exposure-like change
    that keeps the shadows dark and the highlights in place, where a Lab
    offset lifts every pixel alike (a foundation look), at 0.3 ms for a
    512 px face against 2.5."""
    import cv2

    pair = np.clip(np.stack([lab_from, lab_to]), 0, 255).astype(np.uint8).reshape(1, 2, 3)
    bgr = cv2.cvtColor(pair, cv2.COLOR_LAB2BGR).astype(np.float32).reshape(2, 3)
    gains = np.clip(bgr[1] / np.maximum(bgr[0], 1.0), *SKIN_GAIN_LIMIT)
    ramp = np.arange(256, dtype=np.float32)
    table = np.clip(np.round(ramp[:, None] * gains[None, :]), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(table.reshape(256, 1, 3))


def lips_from_portrait(
    portrait: np.ndarray, portrait_pts: np.ndarray, face_pts: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    """The photograph's lips on the driver's lip geometry: the lip band's
    triangles warped from the portrait's crop onto the face's, the band
    three pixels inside the ring and feathered, and the box (y0, y1, x0, x1)
    it lives in. ``None`` when nothing landed."""
    import cv2

    layer, cover, box = warp_triangles(portrait, portrait_pts, face_pts, LIP_TRIANGLES, size)
    y0, y1, x0, x1 = box
    if y1 <= y0:
        return None
    pad = 6
    y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
    y1, x1 = min(y1 + pad, size), min(x1 + pad, size)
    # the outer three pixels of the band are the photograph's lip EDGE (a dark
    # line drawn over the face's own edge): left out, the rest feathered
    part: Any = cv2.erode(cover[y0:y1, x0:x1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    part = cv2.GaussianBlur(part, (7, 7), 0)
    band = np.zeros_like(cover)
    band[y0:y1, x0:x1] = part
    return layer, band, (y0, y1, x0, x1)


def shave_zone(size: int, points_px: np.ndarray, m: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Where a beard grows: the face oval below the nose's base (a soft start
    over 6% of the face's height), the lips and a little around them left out."""
    import cv2

    pts = points_in_crop(points_px, m)
    y_nose = float(pts[NOSE_BASE, 1])
    face_h = max(float(pts[CHIN, 1] - pts[FOREHEAD_TOP, 1]), 1.0)
    ramp = max(face_h * 0.06, 2.0)
    ys = np.arange(size, dtype=np.float32)[:, None]
    zone: Any = hull * np.clip((ys - y_nose) / ramp, 0.0, 1.0)
    lips = ring_mask(size, points_px, m, OUTER_LIPS, erode=0.0, feather=0.1)
    if lips is not None:
        k = max(int(face_h * 0.02), 3)
        room = cv2.dilate(lips, np.ones((k, k), np.uint8))
        zone = zone * (1.0 - np.clip(room, 0.0, 1.0))
    return np.asarray(zone, dtype=np.float32)


def shave(face: np.ndarray, zone: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """The stubble taken out of ``zone``: the picture there smoothed at half
    size by a median (the hairs are a few pixels; the chin's shading is not)
    and its colour moved to the skin's (``skin``: the oval, the zone
    excluded), the lightness only part way so the chin keeps its own shadow.
    The restorer then draws clean skin over the smooth patch."""
    import cv2

    ys, xs = np.where(zone > 0.02)
    if ys.size < 64:
        return face
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    part = face[y0:y1, x0:x1]
    half = cv2.resize(part, ((x1 - x0) // 2, (y1 - y0) // 2), interpolation=cv2.INTER_AREA)
    half = cv2.medianBlur(half, 5)
    smooth = cv2.resize(half, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    lab_part = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = zone[y0:y1, x0:x1] > 0.5
    if inside.sum() < 64:
        return face
    reference = lab_median(face, skin * (zone < 0.05), stride=4)
    if not reference.any():
        return face
    shift = reference - np.median(lab_part[::2, ::2][inside[::2, ::2]], axis=0)
    shift[0] *= 0.6
    tinted = cv2.cvtColor(np.clip(lab_part + shift, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = face.copy()
    out[y0:y1, x0:x1] = blend(part, tinted, zone[y0:y1, x0:x1])
    return out


def beard_signal(face512: np.ndarray, points_px: np.ndarray, m: np.ndarray) -> float:
    """How much more speckled the chin is than the cheeks in a 512 crop: the
    high-frequency energy of the lightness inside the shave zone over that
    of the cheeks (between the eyes' line and the nose's base, the nose's
    column left out). Stubble is a few-pixel speckle the cheeks do not have.
    On the avatars of 2026-10-17: the clean-shaven ones 1.18-1.44, the
    bearded one 1.77; the threshold is :data:`BEARD_THRESHOLD`."""
    import cv2

    size = face512.shape[0]
    hull = crop_mask(size, mode="hull", points_px=points_px, m=m)
    zone = shave_zone(size, points_px, m, hull)
    pts = points_in_crop(points_px, m)
    y_eye = float(pts[IRIS_A, 1] + pts[IRIS_B, 1]) / 2.0
    y_nose = float(pts[NOSE_BASE, 1])
    nose_x = float(pts[NOSE_TIP, 0])
    ys = np.arange(size)[:, None]
    xs = np.arange(size)[None, :]
    cheeks = (hull > 0.5) & (ys > y_eye + 10) & (ys < y_nose) & (np.abs(xs - nose_x) > size * 0.09)
    chin = zone > 0.5
    if chin.sum() < 256 or cheeks.sum() < 256:
        return 0.0
    light = cv2.cvtColor(face512, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    speckle = light - cv2.GaussianBlur(light, (0, 0), 3)
    e_chin = float(np.sqrt(np.mean(speckle[chin] ** 2)))
    e_cheek = float(np.sqrt(np.mean(speckle[cheeks] ** 2)))
    return e_chin / max(e_cheek, 1e-3)


BEARD_THRESHOLD = 1.6


# --- fp16 ONNX ---------------------------------------------------------------------


def fp16_onnx_path(onnx_path: Path) -> Path:
    return onnx_path.with_name(onnx_path.stem + "-fp16.onnx")


def toposort(graph: Any) -> None:
    """``graph.node`` reordered so every node follows its producers (Kahn's
    algorithm; the swap's graphs have no subgraphs)."""
    available = {i.name for i in graph.input} | {t.name for t in graph.initializer} | {""}
    remaining = list(graph.node)
    ordered = []
    while remaining:
        ready = [n for n in remaining if all(i in available for i in n.input)]
        if not ready:
            missing = sorted({i for n in remaining for i in n.input if i not in available})[:5]
            msg = f"the graph has no topological order; unresolved inputs e.g. {missing}"
            raise RuntimeError(msg)
        for n in ready:
            ordered.append(n)
            available.update(o for o in n.output)
        ready_ids = {id(n) for n in ready}
        remaining = [n for n in remaining if id(n) not in ready_ids]
    del graph.node[:]
    graph.node.extend(ordered)


def _tensor_to_fp16(t: Any) -> Any:
    from onnx import numpy_helper

    arr = np.asarray(numpy_helper.to_array(t), dtype=np.float32)
    arr = np.clip(arr, -FP16_MAX, FP16_MAX)
    small = (arr != 0) & (np.abs(arr) < FP16_TINY)
    arr = np.where(small, np.sign(arr) * FP16_TINY, arr)
    return numpy_helper.from_array(arr.astype(np.float16), t.name)


def blocked_nodes(model: Any, flagged: list[dict[str, Any]]) -> set[str]:
    """The flagged nodes plus every consumer of a flagged tensor: the value
    has to come back into fp16's range (a Sqrt, a Div) before the Cast down."""
    tensors = {r["tensor"] for r in flagged}
    names = {r["node"] for r in flagged}
    for node in model.graph.node:
        if any(i in tensors for i in node.input):
            names.add(node.name)
    return {n for n in names if n}


def to_fp16(
    model: Any,
    *,
    keep_ops: frozenset[str] | set[str] = KEEP_FP32,
    blocked: set[str] | None = None,
    pin_batch: bool = True,
) -> tuple[Any, dict[str, int]]:
    """``model`` with every float tensor in fp16 except around the kept op
    types and the blocked nodes; the graph's inputs and outputs float32; a
    dynamic batch pinned to 1 (TensorRT then needs no profile)."""
    import onnx
    from onnx import TensorProto, helper, shape_inference

    blocked = blocked or set()
    FLOAT, HALF = TensorProto.FLOAT, TensorProto.FLOAT16  # noqa: N806  (onnx's own names)
    model = onnx.ModelProto.FromString(model.SerializeToString())
    if pin_batch:
        for i in model.graph.input:
            dims = i.type.tensor_type.shape.dim
            if dims and not dims[0].dim_value:
                dims[0].ClearField("dim_param")
                dims[0].dim_value = 1
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    g = inferred.graph
    dtype: dict[str, int] = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        dtype[vi.name] = vi.type.tensor_type.elem_type
    inits = {t.name: t for t in g.initializer}
    for t in g.initializer:
        dtype[t.name] = t.data_type
    graph_inputs = [i.name for i in g.input if i.name not in inits]
    graph_outputs = [o.name for o in g.output]

    def is16(node: Any) -> bool:
        return node.op_type not in keep_ops and node.name not in blocked

    untyped = 0
    for n in g.node:
        for o in n.output:
            if o and o not in dtype:
                untyped += 1
                if any(dtype.get(i) == FLOAT for i in n.input):
                    dtype[o] = FLOAT
    prec: dict[str, int] = {}
    for name in graph_inputs:
        if dtype.get(name) == FLOAT:
            prec[name] = FLOAT
    for n in g.node:
        p = HALF if is16(n) else FLOAT
        for o in n.output:
            if o and dtype.get(o) == FLOAT:
                prec[o] = p
    consumers16 = dict.fromkeys(inits, False)
    consumers32 = dict.fromkeys(inits, False)
    for n in g.node:
        for i in n.input:
            if i in inits and inits[i].data_type == FLOAT:
                if is16(n):
                    consumers16[i] = True
                else:
                    consumers32[i] = True
    new_inits = []
    renamed16: dict[str, str] = {}
    for t in g.initializer:
        if t.data_type != FLOAT or not consumers16[t.name]:
            new_inits.append(t)
            continue
        if consumers32[t.name]:
            new_inits.append(t)
            t16 = _tensor_to_fp16(t)
            t16.name = t.name + "_fp16"
            renamed16[t.name] = t16.name
            new_inits.append(t16)
        else:
            new_inits.append(_tensor_to_fp16(t))
    new_nodes: list[Any] = []
    casts: dict[tuple[str, int], str] = {}

    def cast_to(name: str, to: int) -> str:
        key = (name, to)
        if key not in casts:
            out = name + ("_to16" if to == HALF else "_to32")
            new_nodes.append(helper.make_node("Cast", [name], [out], name=out + "_cast", to=to))
            casts[key] = out
        return casts[key]

    for n in g.node:
        node = onnx.NodeProto()
        node.CopyFrom(n)
        want = HALF if is16(n) else FLOAT
        for k, i in enumerate(node.input):
            if not i:
                continue
            if i in inits:
                if inits[i].data_type == FLOAT and want == HALF:
                    node.input[k] = renamed16.get(i, i)
                continue
            if dtype.get(i) != FLOAT:
                continue
            if prec.get(i, FLOAT) != want:
                node.input[k] = cast_to(i, want)
        if node.op_type == "Constant" and want == HALF:
            for a in node.attribute:
                if a.name == "value" and a.t.data_type == FLOAT:
                    a.t.CopyFrom(_tensor_to_fp16(a.t))
        new_nodes.append(node)
    for o in graph_outputs:
        if prec.get(o) == HALF:
            for node in new_nodes:
                for k, name in enumerate(node.output):
                    if name == o:
                        node.output[k] = o + "_16"
                for k, name in enumerate(node.input):
                    if name == o:
                        node.input[k] = o + "_16"
            new_nodes.append(
                helper.make_node("Cast", [o + "_16"], [o], name=o + "_out_cast", to=FLOAT)
            )
    out = onnx.ModelProto()
    out.CopyFrom(model)
    del out.graph.node[:]
    out.graph.node.extend(new_nodes)
    del out.graph.initializer[:]
    out.graph.initializer.extend(new_inits)
    del out.graph.value_info[:]
    toposort(out.graph)
    stats = {
        "nodes_fp16": sum(1 for n in g.node if is16(n)),
        "nodes_fp32": sum(1 for n in g.node if not is16(n)),
        "casts": len(casts),
        "untyped_outputs_assumed": untyped,
    }
    return out, stats


def write_fp16_onnx(
    onnx_path: Path, target: Path, flagged: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """The fp16 ONNX for one network, checked, written beside the fp32 file."""
    import onnx

    started = time.perf_counter()
    model = onnx.load(str(onnx_path))
    blocked = blocked_nodes(model, flagged) if flagged else set()
    model16, stats = to_fp16(model, blocked=blocked)
    onnx.checker.check_model(model16)
    partial = target.with_name(target.name + ".part")
    onnx.save(model16, str(partial))
    partial.replace(target)
    meta = {
        "onnx": onnx_path.name,
        "fp16": target.name,
        "blocked_nodes": len(blocked),
        **stats,
        "seconds": round(time.perf_counter() - started, 1),
        "size_bytes": target.stat().st_size,
    }
    Path(str(target) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    _LOG.info(
        "wrote %s: %d nodes fp16, %d fp32 (%d blocked), %d casts, %.1f s",
        target.name,
        stats["nodes_fp16"],
        stats["nodes_fp32"],
        len(blocked),
        stats["casts"],
        meta["seconds"],
    )
    return meta


# --- plans -------------------------------------------------------------------------


def plan_path(engines: Path, net: str, compute_capability: tuple[int, int], version: str) -> Path:
    major, minor = compute_capability
    return engines / f"swap-{net}-fp16-sm{major}{minor}-trt{version}.plan"


def fp16_source(models: Path, relative: Path) -> tuple[Path, Path | None]:
    """The fp16 ONNX a plan is built from and, when it has still to be
    written, the fp32 original to write it from (``None`` once the fp16
    copy is there). A package ships the fp16 copies only
    (``scripts/build_exe.py --engine swap``), so the original is asked for
    only when it is needed (2026-10-18: the first start on another graphics
    card died asking for it); neither there is a :class:`FileNotFoundError`."""
    onnx_path = models / relative
    onnx16 = fp16_onnx_path(onnx_path)
    if onnx16.is_file():
        return onnx16, None
    if onnx_path.is_file():
        return onnx16, onnx_path
    msg = (
        f"the swap's network is missing: {onnx16} "
        f"(nor the original {onnx_path.name} to write it from)"
    )
    raise FileNotFoundError(msg)


def ensure_plans(
    models: Path,
    engines: Path,
    blocks_path: Path | None,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """The three fp16 plans for this GPU, built at the first start from the
    fp16 ONNX files (written first when missing). ``progress`` hears a
    sentence per step for the start-up window."""
    import tensorrt as trt
    import torch

    from sha256.avatar.liveportrait_tensorrt import build_engine_watched

    cc = torch.cuda.get_device_capability(0)
    version = str(trt.__version__)
    flagged: dict[str, list[dict[str, Any]]] = {}
    if blocks_path is not None and blocks_path.is_file():
        flagged = json.loads(blocks_path.read_text(encoding="utf-8")).get("blocks", {})
    plans: dict[str, Path] = {}
    for net, relative in ONNX_FILES.items():
        plan = plan_path(engines, net, cc, version)
        plans[net] = plan
        if plan.is_file():
            continue
        onnx16, original = fp16_source(models, relative)
        if original is not None:
            if progress is not None:
                progress(f"Preparing the {net} network for this graphics card")
            write_fp16_onnx(original, onnx16, flagged.get(net))
        head = f"Building the {net} engine for this graphics card (once, a minute)"
        if progress is not None:
            progress(head)

        def tell(seconds: float, head: str = head) -> None:
            if progress is not None:
                progress(f"{head} {seconds:.0f} s")

        build_engine_watched(onnx16, plan, tell if progress is not None else None)
    return plans


def load_emap(models: Path) -> np.ndarray:
    """inswapper's ``emap`` (the 512x512 matrix the ArcFace embedding is
    projected by before the swap -- the graph's last initializer), cached as
    ``inswapper_emap.npy`` beside the ONNX after the first read."""
    cache = models / EMAP_CACHE_NAME
    if cache.is_file():
        cached = np.load(cache)
        if cached.shape == (512, 512):
            return np.asarray(cached, dtype=np.float32)
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(models / ONNX_FILES["inswapper"]), load_external_data=False)
    found: Any = None
    for init in model.graph.initializer:
        if list(init.dims) == [512, 512]:
            found = numpy_helper.to_array(init)
    if found is None:
        found = numpy_helper.to_array(model.graph.initializer[-1])
    if found.shape != (512, 512):
        msg = f"inswapper_128.onnx: no 512x512 emap initializer (last is {found.shape})"
        raise RuntimeError(msg)
    emap = np.asarray(found, dtype=np.float32)
    try:
        np.save(cache, emap)
    except OSError as exc:
        _LOG.warning("the emap could not be cached at %s: %s", cache, exc)
    return emap


# --- TensorRT at run time ------------------------------------------------------------


class TrtNet:
    """One plan run from torch: the output buffers allocated once, the inputs
    copied into device tensors per call, one ``execute_async_v3`` on torch's
    current stream (the same discipline as ``liveportrait_tensorrt.PixelEngine``)."""

    def __init__(self, plan: Path) -> None:
        import tensorrt as trt
        import torch

        self.plan = Path(plan)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(self.plan.read_bytes())
        if self.engine is None:
            msg = f"TensorRT could not deserialise {self.plan}"
            raise RuntimeError(msg)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            msg = f"TensorRT could not create an execution context for {self.plan}"
            raise RuntimeError(msg)
        dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = tuple(1 if d < 0 else d for d in self.engine.get_tensor_shape(name))
            self.context.set_input_shape(name, shape)
            tensor = torch.empty(
                shape, dtype=dtypes[self.engine.get_tensor_dtype(name)], device="cuda"
            )
            self.inputs[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            tensor = torch.empty(
                shape, dtype=dtypes[self.engine.get_tensor_dtype(name)], device="cuda"
            )
            self.outputs[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.device_memory_bytes = int(self.engine.device_memory_size_v2)

    def input_names(self) -> list[str]:
        return list(self.inputs)

    def run_tensors(self, **tensors: Any) -> Any:
        """Device tensors (or arrays) in by input name; the first output
        tensor back, still on the device, valid until the next call."""
        import torch

        for name, value in tensors.items():
            tensor = self.inputs[name]
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(np.ascontiguousarray(value))
            tensor.copy_(value)
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            msg = f"TensorRT execute_async_v3 failed for {self.plan.name}"
            raise RuntimeError(msg)
        return next(iter(self.outputs.values()))

    def run(self, **arrays: np.ndarray) -> np.ndarray:
        """The arrays in (by input name), the first output back on the CPU."""
        return np.asarray(self.run_tensors(**arrays).float().cpu().numpy())

    def close(self) -> None:
        self.inputs.clear()
        self.outputs.clear()
        self.context = None
        self.engine = None


class SwapNets:
    """The three networks as plans, with the swap's pre- and post-processing.
    One lock: the avatar's load (ArcFace) and the frame path (the other two)
    may come from different threads and share the CUDA stream."""

    def __init__(self, plans: dict[str, Path], emap: np.ndarray) -> None:
        self._lock = threading.Lock()
        self.arcface = TrtNet(plans["arcface"])
        self.swapper = TrtNet(plans["inswapper"])
        self.enhancer = TrtNet(plans["gfpgan"])
        self.emap = emap
        self._arc_in = self.arcface.input_names()[0]
        names = self.swapper.input_names()
        self._swap_target = next(n for n in names if self.swapper.inputs[n].dim() == 4)
        self._swap_source = next(n for n in names if self.swapper.inputs[n].dim() == 2)
        self._enh_in = self.enhancer.input_names()[0]

    @property
    def device_memory_bytes(self) -> int:
        return (
            self.arcface.device_memory_bytes
            + self.swapper.device_memory_bytes
            + self.enhancer.device_memory_bytes
        )

    def embed(self, bgr112: np.ndarray) -> np.ndarray:
        """The unit ArcFace embedding of an aligned 112 px face."""
        with self._lock:
            out = self.arcface.run(**{self._arc_in: blob_arcface(bgr112)})[0]
        return np.asarray(out / max(float(np.linalg.norm(out)), 1e-12), dtype=np.float32)

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        """inswapper's identity input: the embedding through ``emap``, unit length."""
        lat = embedding.reshape(1, -1) @ self.emap
        return np.asarray(lat / max(float(np.linalg.norm(lat)), 1e-12), dtype=np.float32)

    @staticmethod
    def _upload_rgb(bgr: np.ndarray) -> Any:
        """A BGR uint8 crop as the device's 1x3xHxW float RGB in 0..1 (the
        channel flip and the scaling on the GPU: 3 ms of numpy at 512 become
        a fraction of one)."""
        import torch

        t = torch.from_numpy(np.ascontiguousarray(bgr)).to("cuda", non_blocking=False)
        return t[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

    @staticmethod
    def _download_bgr(rgb01: Any) -> np.ndarray:
        """The device's 1x3xHxW float RGB in 0..1 as a BGR uint8 crop."""
        import torch

        t = rgb01[0].clamp_(0.0, 1.0).mul_(255.0).round_().to(dtype=torch.uint8)
        return np.asarray(t.permute(1, 2, 0)[:, :, [2, 1, 0]].contiguous().cpu().numpy())

    def swap(self, bgr128: np.ndarray, latent: np.ndarray) -> np.ndarray:
        """inswapper: the aligned 128 px crop (RGB 0..1 in) with the identity."""
        with self._lock:
            x = self._upload_rgb(bgr128)
            pred = self.swapper.run_tensors(**{self._swap_target: x, self._swap_source: latent})
            return self._download_bgr(pred.float().clone())

    def enhance(self, bgr512: np.ndarray) -> np.ndarray:
        """GFPGAN: the 512 px crop (RGB -1..1 in and out) restored."""
        with self._lock:
            x = self._upload_rgb(bgr512).sub_(0.5).div_(0.5)
            pred = self.enhancer.run_tensors(**{self._enh_in: x})
            out = pred.float().clamp(-1.0, 1.0).add_(1.0).div_(2.0)
            return self._download_bgr(out)

    def close(self) -> None:
        for net in (self.arcface, self.swapper, self.enhancer):
            net.close()
