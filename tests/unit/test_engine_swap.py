"""The swap engine's CPU side (road C, 2026-10-17): the alignment, the
masks, the paste, the fp16 graph conversion and the engine's contract. No
GPU: the networks are not run here (scripts/probes/swap_engine_still.py and
swap_trt_time.py measure those on the card)."""

from __future__ import annotations

import numpy as np
import pytest

from sha256.avatar import swap_models as sm
from sha256.avatar.engine_swap import ENGINE_ID, SwapEngine
from sha256.core.interfaces import AvatarEngine

cv2 = pytest.importorskip("cv2")


def _face_points(centre=(640.0, 360.0), scale=1.0, angle=0.0) -> np.ndarray:
    """478 points: a synthetic face whose five ArcFace points and oval are
    where MediaPipe would put them, under a similarity."""
    pts = np.zeros((478, 3), dtype=np.float32)
    rng = np.random.default_rng(7)
    pts[:, :2] = rng.uniform(-80, 80, size=(478, 2))
    pts[sm.IRIS_A, :2] = (-45.0, -30.0)
    pts[sm.IRIS_B, :2] = (45.0, -30.0)
    pts[sm.NOSE_TIP, :2] = (0.0, 20.0)
    pts[sm.MOUTH_A, :2] = (-35.0, 70.0)
    pts[sm.MOUTH_B, :2] = (35.0, 70.0)
    oval = np.array(
        [
            (130.0 * np.sin(a), -150.0 * np.cos(a) + 10.0)
            for a in np.linspace(0, 2 * np.pi, len(sm.FACE_OVAL), endpoint=False)
        ]
    )
    pts[list(sm.FACE_OVAL), :2] = oval
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    rot = np.array([[c, -s], [s, c]])
    pts[:, :2] = (pts[:, :2] @ rot.T) * scale + np.asarray(centre)
    return pts


# --- geometry -------------------------------------------------------------------


def test_umeyama_recovers_a_similarity():
    src = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 3]], dtype=np.float64)
    angle = np.radians(30.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = 2.5 * (src @ rot.T) + np.array([100.0, -40.0])
    m = sm.umeyama(src, dst)
    back = np.hstack([src, np.ones((5, 1))]) @ m.T
    assert np.allclose(back, dst, atol=1e-6)


def test_estimate_norm_matches_insightface_templates():
    """112: the template itself; 128: the template shifted 8 px right, the way
    inswapper's crops were made. The five points land on the template."""
    pts = _face_points()
    kps = sm.five_points(pts)
    for size, shift in ((112, 0.0), (128, 8.0)):
        m = sm.estimate_norm(kps, size)
        mapped = np.hstack([kps, np.ones((5, 1))]) @ m.T
        target = sm.ARCFACE_DST + np.array([shift, 0.0])
        # a similarity cannot place five points exactly; the fit is within a pixel
        assert np.abs(mapped - target).max() < 1.5, (size, mapped, target)
    with pytest.raises(ValueError):
        sm.estimate_norm(kps, 100)


def test_five_points_are_image_left_to_right_whichever_way_the_head_turns():
    pts = _face_points()
    mirrored = pts.copy()
    mirrored[:, 0] = 1280.0 - mirrored[:, 0]
    a = sm.five_points(pts)
    b = sm.five_points(mirrored)
    assert a[0, 0] < a[1, 0] and a[3, 0] < a[4, 0]
    assert b[0, 0] < b[1, 0] and b[3, 0] < b[4, 0]
    assert np.allclose(a[2], pts[sm.NOSE_TIP, :2])


# --- masks and the paste --------------------------------------------------------


def test_hull_mask_stays_inside_the_oval_and_off_the_crop_edge():
    pts = _face_points()
    kps = sm.five_points(pts)
    m = sm.umeyama(kps, sm.FFHQ_DST_512)
    mask = sm.crop_mask(512, mode="hull", points_px=pts, m=m)
    assert mask.shape == (512, 512) and mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert mask[0, :].max() == 0.0 and mask[:, 0].max() == 0.0
    # the oval's centre is fully pasted, a far corner is not
    centre = np.hstack([pts[sm.NOSE_TIP, :2], 1.0]) @ m.T
    assert mask[int(centre[1]), int(centre[0])] > 0.99
    assert mask[20, 20] == 0.0


def test_box_mask_is_insightfaces_eroded_blurred_box():
    mask = sm.crop_mask(128, mode="box")
    assert mask[64, 64] > 0.99
    assert mask[2, 2] < 0.05
    with pytest.raises(ValueError):
        sm.crop_mask(128, mode="hull")
    with pytest.raises(ValueError):
        sm.crop_mask(128, mode="disc")


def test_paste_touches_only_the_crops_region_and_blends_by_the_mask():
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    pts = _face_points()
    kps = sm.five_points(pts)
    m = sm.umeyama(kps, sm.FFHQ_DST_512)
    crop = np.full((512, 512, 3), 200, dtype=np.uint8)
    mask = sm.crop_mask(512, mode="hull", points_px=pts, m=m)
    out = frame.copy()
    sm.paste(out, crop, m, mask)
    # the face's centre is the crop, a corner of the frame is untouched
    centre = pts[sm.NOSE_TIP, :2].astype(int)
    assert tuple(out[centre[1], centre[0]]) == (200, 200, 200)
    assert np.array_equal(out[:50, :50], frame[:50, :50])
    assert np.array_equal(out[-50:, -50:], frame[-50:, -50:])
    # half strength: halfway between
    half = frame.copy()
    sm.paste(half, crop, m, mask, strength=0.5)
    assert abs(int(half[centre[1], centre[0], 0]) - 120) <= 1


def test_paste_at_the_frames_edge_does_not_raise():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    pts = _face_points(centre=(30.0, 700.0))
    kps = sm.five_points(pts)
    m = sm.estimate_norm(kps, 128)
    crop = np.full((128, 128, 3), 255, dtype=np.uint8)
    sm.paste(frame, crop, m, sm.crop_mask(128, mode="box"))
    assert frame.max() == 255


def _open_mouth(pts: np.ndarray, opening: float) -> np.ndarray:
    """The inner and outer lips of the synthetic face, ``opening`` px apart."""
    pts = pts.copy()
    upper = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lower = [324, 318, 402, 317, 14, 87, 178, 88, 95]
    xs = np.linspace(-30, 30, len(upper))
    for i, x in zip(upper, xs, strict=True):
        pts[i, :2] = (640 + x, 430 - opening / 2)
    xs = np.linspace(30, -30, len(lower))
    for i, x in zip(lower, xs, strict=True):
        pts[i, :2] = (640 + x, 430 + opening / 2)
    outer_u = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    outer_l = [375, 321, 405, 314, 17, 84, 181, 91, 146]
    xs = np.linspace(-38, 38, len(outer_u))
    for i, x in zip(outer_u, xs, strict=True):
        pts[i, :2] = (640 + x, 430 - opening / 2 - 16)
    xs = np.linspace(38, -38, len(outer_l))
    for i, x in zip(outer_l, xs, strict=True):
        pts[i, :2] = (640 + x, 430 + opening / 2 + 16)
    return pts


def test_shut_lips_leave_no_hole_and_open_lips_do():
    pts = _open_mouth(_face_points(), 0.0)
    kps = sm.five_points(pts)
    m = sm.umeyama(kps, sm.FFHQ_DST_512)
    assert sm.mouth_opening(512, pts, m) is None
    full = sm.crop_mask(512, mode="hull", points_px=pts, m=m, mouth_hole=1.0)
    centre = np.hstack([[640.0, 430.0], 1.0]) @ m.T
    assert full[int(centre[1]), int(centre[0])] > 0.99
    pts = _open_mouth(_face_points(), 40.0)
    holed = sm.crop_mask(512, mode="hull", points_px=pts, m=m, mouth_hole=1.0)
    assert holed[int(centre[1]), int(centre[0])] < 0.02
    # the lips themselves (between the rings) are still the avatar's with the hole alone
    lip = np.hstack([[640.0, 430.0 - 28.0], 1.0]) @ m.T
    assert holed[int(lip[1]), int(lip[0])] > 0.5
    # and the driver's own with lips_keep
    kept = sm.crop_mask(512, mode="hull", points_px=pts, m=m, lips_keep=1.0)
    assert kept[int(lip[1]), int(lip[0])] < 0.05
    assert kept[int(centre[1]), int(centre[0])] < 0.02
    # the eye is untouched by either
    eye = np.hstack([pts[sm.IRIS_A, :2], 1.0]) @ m.T
    assert kept[int(eye[1]), int(eye[0])] > 0.99


# --- the fp16 conversion -----------------------------------------------------------


def _tiny_model():
    """x -> Conv -> Mul(x, x) -> ReduceMean -> Add(eps) -> Sqrt -> Div: an
    instance norm's variance path, the shape that overflows fp16."""
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5, "w")
    eps = numpy_helper.from_array(np.array(1e-5, dtype=np.float32), "eps")
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], name="conv"),
        helper.make_node("Mul", ["c", "c"], ["sq"], name="square"),
        helper.make_node("ReduceMean", ["sq"], ["var"], name="mean", axes=[2, 3], keepdims=1),
        helper.make_node("Add", ["var", "eps"], ["var_eps"], name="add_eps"),
        helper.make_node("Sqrt", ["var_eps"], ["std"], name="sqrt"),
        helper.make_node("Div", ["c", "std"], ["y"], name="div"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])],
        initializer=[w, eps],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def test_to_fp16_keeps_blocked_nodes_fp32_with_casts_and_pins_the_batch():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto

    model = _tiny_model()
    flagged = [{"tensor": "sq", "node": "square"}, {"tensor": "var", "node": "mean"}]
    blocked = sm.blocked_nodes(model, flagged)
    assert blocked == {"square", "mean", "add_eps"}  # add_eps consumes 'var'
    model16, stats = sm.to_fp16(model, blocked=blocked)
    onnx.checker.check_model(model16)
    ops = {n.name: n for n in model16.graph.node}
    assert stats["nodes_fp32"] == 3 and stats["nodes_fp16"] == 3
    assert model16.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
    assert model16.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model16.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    inits = {t.name: t.data_type for t in model16.graph.initializer}
    assert inits["w"] == TensorProto.FLOAT16  # fp16 consumer only
    assert inits["eps"] == TensorProto.FLOAT  # blocked consumer only
    # the boundary casts: into the fp32 island, out of it, and at the graph's edges
    casts = [n for n in model16.graph.node if n.op_type == "Cast"]
    assert len(casts) == stats["casts"] + 1  # + the output cast
    assert ops["square"].input[0].endswith("_to32")
    assert ops["sqrt"].input[0].endswith("_to16")
    # every node follows its producers
    seen = {i.name for i in model16.graph.input} | set(inits) | {""}
    for n in model16.graph.node:
        assert all(i in seen for i in n.input), n.name
        seen.update(n.output)


def test_to_fp16_runs_on_onnxruntime_like_the_fp32_graph():
    ort = pytest.importorskip("onnxruntime")
    model = _tiny_model()
    x = (np.random.default_rng(1).standard_normal((1, 2, 4, 4)) * 300).astype(np.float32)
    ref = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, {"x": x}
    )[0]
    blocked = sm.blocked_nodes(
        model, [{"tensor": "sq", "node": "square"}, {"tensor": "var", "node": "mean"}]
    )
    model16, _ = sm.to_fp16(model, blocked=blocked)
    out = ort.InferenceSession(model16.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, {"x": x}
    )[0]
    assert out.dtype == np.float32
    assert np.allclose(out, ref, rtol=2e-3, atol=2e-3)


def test_toposort_orders_an_appended_producer():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    a = helper.make_node("Relu", ["t"], ["y"], name="second")
    b = helper.make_node("Neg", ["x"], ["t"], name="first")
    graph = helper.make_graph(
        [a, b],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    sm.toposort(graph)
    assert [n.name for n in graph.node] == ["first", "second"]
    onnx.checker.check_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))


# --- the engine's contract -------------------------------------------------------------


def test_satisfies_the_avatar_engine_protocol():
    assert isinstance(SwapEngine(), AvatarEngine)


def test_only_swap_avatars_are_supported(sample_descriptor):
    engine = SwapEngine()
    assert engine.engine_id == ENGINE_ID
    assert not engine.supports(sample_descriptor)
    swap = sample_descriptor.model_copy(
        update={
            "manifest": sample_descriptor.manifest.model_copy(
                update={
                    "engine": sample_descriptor.manifest.engine.model_copy(
                        update={"engine_id": ENGINE_ID}
                    )
                }
            )
        }
    )
    assert engine.supports(swap)
    engine.reset_anchor("anything")
    assert engine.rest_status("anything") is None
    assert engine.loaded_avatars == ()


# --- the lips from the photograph, the shave ------------------------------------------


def test_warp_triangles_moves_the_source_colour_onto_the_target_band():
    src = np.full((512, 512, 3), (10, 200, 30), dtype=np.uint8)
    src_pts = _open_mouth(_face_points(), 20.0)
    dst_pts = _open_mouth(_face_points(centre=(600.0, 380.0), scale=1.2), 40.0)
    kps = sm.five_points(dst_pts)
    m = sm.umeyama(kps, sm.FFHQ_DST_512)
    layer, cover, box = sm.warp_triangles(
        src, sm.points_in_crop(src_pts, m), sm.points_in_crop(dst_pts, m), sm.LIP_TRIANGLES, 512
    )
    y0, y1, x0, x1 = box
    assert y1 > y0 and x1 > x0
    assert cover[y0:y1, x0:x1].max() == 1.0 and cover[:y0].max() == 0.0
    inside = cover > 0.5
    assert np.all(layer[inside] == (10, 200, 30))
    # shut lips: the opening is gone, the band (the lips themselves) is smaller
    _, cover_shut, box_shut = sm.warp_triangles(
        src,
        sm.points_in_crop(src_pts, m),
        sm.points_in_crop(_open_mouth(_face_points(), 0.0), m),
        sm.LIP_TRIANGLES,
        512,
    )
    assert box_shut != (0, 0, 0, 0) and cover_shut.sum() < cover.sum()


def test_lips_from_portrait_stay_inside_the_mouth_and_off_the_eyes():
    pts = _open_mouth(_face_points(), 30.0)
    kps = sm.five_points(pts)
    m = sm.umeyama(kps, sm.FFHQ_DST_512)
    in_crop = sm.points_in_crop(pts, m)
    portrait = np.full((512, 512, 3), 90, dtype=np.uint8)
    result = sm.lips_from_portrait(portrait, in_crop, in_crop, 512)
    assert result is not None
    layer, band, _box = result
    eye = in_crop[sm.IRIS_A]
    assert band[int(eye[1]), int(eye[0])] == 0.0
    lip = in_crop[0]  # the upper lip's middle (outer ring)
    inner = in_crop[13]  # the upper lip's middle (inner ring)
    mid = (lip + inner) / 2
    assert band[int(mid[1]), int(mid[0])] > 0.5
    assert layer[int(mid[1]), int(mid[0])].tolist() == [90, 90, 90]


def test_lab_shift_moves_the_median_and_touches_only_the_box():
    image = np.full((64, 64, 3), (80, 80, 80), dtype=np.uint8)
    darker = np.full((64, 64, 3), (40, 40, 40), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.float32)
    src_median = sm.lab_median(image, mask)
    out = sm.lab_shift(image, src_median, darker, mask, box=(0, 32, 0, 64))
    assert out[16, 16].mean() < 50
    assert out[48, 16].tolist() == [80, 80, 80]


def test_shave_smooths_the_zone_and_leaves_the_rest():
    rng = np.random.default_rng(3)
    face = np.full((512, 512, 3), 150, dtype=np.uint8)
    stubble = rng.integers(0, 2, size=(512, 512), dtype=np.uint8) * 90
    face[:, :, 0] -= stubble
    face[:, :, 1] -= stubble
    face[:, :, 2] -= stubble
    pts = _open_mouth(_face_points(), 0.0)
    kps = sm.five_points(pts)
    m = sm.umeyama(kps, sm.FFHQ_DST_512)
    hull = sm.crop_mask(512, mode="hull", points_px=pts, m=m)
    zone = sm.shave_zone(512, pts, m, hull)
    chin = sm.points_in_crop(pts, m)[sm.CHIN]
    eye = sm.points_in_crop(pts, m)[sm.IRIS_A]
    cy, cx = min(int(chin[1]), 511) - 24, int(chin[0])
    assert zone[cy, cx] > 0.9
    assert zone[int(eye[1]), int(eye[0])] == 0.0
    out = sm.shave(face, zone, hull)
    before = face[cy - 8 : cy + 8, cx - 8 : cx + 8, 0].astype(np.float32).std()
    after = out[cy - 8 : cy + 8, cx - 8 : cx + 8, 0].astype(np.float32).std()
    assert after < before * 0.5
    ey, ex = int(eye[1]), int(eye[0])
    assert np.array_equal(
        out[ey - 4 : ey + 4, ex - 4 : ex + 4], face[ey - 4 : ey + 4, ex - 4 : ex + 4]
    )


# --- the skin toward the portrait's (his look 2026-10-18: "a tan") -------------------------


def test_lab_median_reads_the_strided_pixels_like_the_whole_picture():
    import cv2

    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.float32)
    whole = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)[::4, ::4]
    expected = np.median(whole.reshape(-1, 3), axis=0)
    assert np.allclose(sm.lab_median(image, mask, stride=4), expected)


def test_skin_goal_is_the_skin_moved_part_way_to_the_portrait():
    median = np.array([150.0, 134.0, 146.0], dtype=np.float32)
    portrait = median + np.array([10.0, 4.0, -2.0], dtype=np.float32)
    assert np.allclose(sm.skin_goal(portrait, median, 1.0), portrait)
    halfway = median + np.array([5.0, 2.0, -1.0], dtype=np.float32)
    assert np.allclose(sm.skin_goal(portrait, median, 0.5), halfway)
    assert sm.skin_goal(portrait, np.zeros(3, dtype=np.float32), 1.0) is None


def test_skin_shift_lands_on_the_goal_clipped_and_eased():
    median = np.array([150.0, 134.0, 146.0], dtype=np.float32)
    goal = median + np.array([10.0, 4.0, -2.0], dtype=np.float32)
    full = sm.skin_shift(goal, median, None)
    assert np.allclose(full, [10.0, 4.0, -2.0])
    far = median + np.array([100.0, 60.0, -60.0], dtype=np.float32)
    assert np.allclose(sm.skin_shift(far, median, None), [40.0, 20.0, -20.0])
    eased = sm.skin_shift(goal, median, np.zeros(3, dtype=np.float32))
    assert np.allclose(eased, [2.0, 0.8, -0.4])  # a fifth of the way per frame
    assert sm.skin_shift(goal, np.zeros(3, dtype=np.float32), full) is full


def test_gain_lut_moves_the_skin_colour_and_keeps_the_shadows_dark():
    import cv2

    image = np.full((16, 16, 3), (110, 130, 160), dtype=np.uint8)
    mask = np.ones((16, 16), dtype=np.float32)
    median = sm.lab_median(image, mask, stride=1)
    goal = median + np.array([12.0, 3.0, 2.0], dtype=np.float32)
    lut = sm.gain_lut(median, goal)
    assert lut.shape == (256, 1, 3) and lut.dtype == np.uint8
    moved = cv2.LUT(image, lut)
    assert np.allclose(sm.lab_median(moved, mask, stride=1), goal, atol=2.0)
    dark = cv2.LUT(np.full((2, 2, 3), 8, dtype=np.uint8), lut)
    assert dark.max() <= 11  # a gain, not an offset: the shadows stay shadows
    assert cv2.LUT(np.full((2, 2, 3), 255, dtype=np.uint8), lut).min() == 255  # saturates
    # the gains are held within SKIN_GAIN_LIMIT
    wild = sm.gain_lut(median, median + np.array([120.0, 0.0, 0.0], dtype=np.float32))
    assert wild[200, 0].max() <= round(200 * sm.SKIN_GAIN_LIMIT[1])


def test_fp16_source_prefers_the_fp16_copy_and_names_what_is_missing(tmp_path):
    from pathlib import Path

    relative = Path("buffalo_l") / "w600k_r50.onnx"
    (tmp_path / "buffalo_l").mkdir()
    with pytest.raises(FileNotFoundError, match=r"w600k_r50-fp16.onnx"):
        sm.fp16_source(tmp_path, relative)
    original = tmp_path / relative
    original.write_bytes(b"fp32")
    assert sm.fp16_source(tmp_path, relative) == (sm.fp16_onnx_path(original), original)
    sm.fp16_onnx_path(original).write_bytes(b"fp16")
    assert sm.fp16_source(tmp_path, relative) == (sm.fp16_onnx_path(original), None)
    original.unlink()  # a package: the fp16 copy alone is enough
    assert sm.fp16_source(tmp_path, relative) == (sm.fp16_onnx_path(original), None)


def test_skin_match_and_forehead_fade_defaults_and_the_off_switch():
    from sha256.avatar.engine_swap import DEFAULT_OPTIONS, _skin_match

    assert DEFAULT_OPTIONS["skin_match"] == 0.0
    assert DEFAULT_OPTIONS["forehead_fade"] == 0.8
    assert _skin_match(0.5) == 0.5
    assert _skin_match(0) == 0.0  # still active: the driver's own skin colour
    assert _skin_match("off") is None
    assert _skin_match(None) is None
    assert _skin_match(False) is None


def _oval_points(size: int, brow_y: float) -> np.ndarray:
    """478 points at the crop's centre, the face oval on an ellipse and the
    brow line at ``brow_y`` -- in the crop's own space (``m`` = identity)."""
    pts = np.full((478, 2), size / 2.0, dtype=np.float32)
    n = len(sm.FACE_OVAL)
    for i, idx in enumerate(sm.FACE_OVAL):
        angle = 2.0 * np.pi * i / n
        x = size / 2.0 + 0.3 * size * np.sin(angle)
        pts[idx] = (x, 0.55 * size - 0.4 * size * np.cos(angle))
    for idx in sm.BROW_LINE:
        pts[idx] = (size / 2.0, brow_y)
    return pts


def test_forehead_fade_lowers_the_mask_above_the_brows_only():
    size = 128
    eye = np.eye(2, 3, dtype=np.float64)
    pts = _oval_points(size, brow_y=45.0)  # the oval's top sits at 0.15 * size = 19
    plain = sm.crop_mask(size, mode="hull", points_px=pts, m=eye)
    faded = sm.crop_mask(size, mode="hull", points_px=pts, m=eye, forehead_fade=1.0)
    x = size // 2
    assert np.allclose(faded[60:100, x], plain[60:100, x])  # below the brows: untouched
    assert faded[45, x] == pytest.approx(plain[45, x], abs=0.02)  # at the brow line: whole
    # halfway up the forehead about half, at the oval's top next to nothing
    assert faded[32, x] == pytest.approx(plain[32, x] * 0.5, abs=0.04)
    assert faded[20, x] < 0.06 * plain[20, x] + 0.01
    half = sm.crop_mask(size, mode="hull", points_px=pts, m=eye, forehead_fade=0.5)
    assert half[20, x] == pytest.approx(plain[20, x] * 0.5, abs=0.04)
