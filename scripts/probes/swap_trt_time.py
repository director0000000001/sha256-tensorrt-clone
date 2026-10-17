"""Road C, step C1/C3 preview: the swap's three networks as TensorRT engines on this GPU,
each timed alone (CUDA events around one execute_async_v3, median and p99 of --runs).

Builds (once; cached under models/tensorrt/) ArcFace w600k_r50, inswapper_128 and
GFPGAN 1.4 from the ONNX files in models/downloads/swap, FP16 where the network allows
the flag, and prints the per-call milliseconds. The answer to "does the enhancer fit in
the 33 ms frame" before any pipeline work.

Usage (from the project root):
  .venv\\Scripts\\python.exe -I scripts/probes/swap_trt_time.py [--precision fp16|fp32] [--runs 200]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = PROJECT_ROOT / "models" / "downloads" / "swap"
PLANS = PROJECT_ROOT / "models" / "tensorrt"
WORKSPACE_BYTES = 4 << 30

NETS = {
    "arcface": MODELS / "buffalo_l" / "w600k_r50.onnx",
    "inswapper": MODELS / "inswapper_128.onnx",
    "gfpgan": MODELS / "gfpgan_1.4.onnx",
}


def plan_name(net: str, precision: str) -> Path:
    import tensorrt as trt
    import torch

    cc = torch.cuda.get_device_capability(0)
    return PLANS / f"swap-{net}-{precision}-sm{cc[0]}{cc[1]}-trt{trt.__version__}.plan"


def onnx_for(net: str, precision: str) -> Path:
    """TensorRT 11 builds strongly typed networks only: fp16 means the fp16 ONNX
    (scripts/probes/swap_onnx_fp16.py writes <name>-fp16.onnx beside the fp32 file)."""
    path = NETS[net]
    if precision == "fp16":
        path16 = path.with_name(path.stem + "-fp16.onnx")
        if not path16.exists():
            raise SystemExit(f"{path16} missing: run scripts/probes/swap_onnx_fp16.py first")
        return path16
    return path


# ----------------------------------------------------------------------------- real inputs
def blob_swap(bgr128: np.ndarray) -> np.ndarray:
    rgb = bgr128[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def unblob_swap(pred: np.ndarray) -> np.ndarray:
    out = pred.transpose(0, 2, 3, 1)[0]
    return np.clip(255.0 * out, 0, 255).astype(np.uint8)[:, :, ::-1].copy()


def blob_enh(bgr512: np.ndarray) -> np.ndarray:
    rgb = bgr512[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None])


def unblob_enh(pred: np.ndarray) -> np.ndarray:
    out = (np.clip(pred[0], -1.0, 1.0) + 1.0) / 2.0
    return (out.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)[:, :, ::-1].copy()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else round(float(10.0 * np.log10(255.0**2 / mse)), 2)


def build(onnx_path: Path, target: Path, precision: str) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    network = builder.create_network(flags)
    strongly_typed = False
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        try:
            strongly_typed = bool(
                network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )
        except Exception:
            strongly_typed = False
    parser = trt.OnnxParser(network, logger)
    t0 = time.perf_counter()
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    parse_s = time.perf_counter() - t0
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    fp16_set = False
    if precision == "fp16" and not strongly_typed:
        config.set_flag(trt.BuilderFlag.FP16)
        fp16_set = True
    # a dynamic batch (ArcFace's 'None') needs a profile pinned at 1
    dynamic = []
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        if any(d < 0 for d in shape):
            fixed = tuple(1 if d < 0 else d for d in shape)
            profile = builder.create_optimization_profile()
            profile.set_shape(inp.name, fixed, fixed, fixed)
            config.add_optimization_profile(profile)
            dynamic.append((inp.name, fixed))
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError(f"TensorRT could not build {onnx_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(bytes(serialized))
    partial.replace(target)
    meta = {
        "onnx": onnx_path.name,
        "plan": target.name,
        "tensorrt": str(trt.__version__),
        "strongly_typed_network": strongly_typed,
        "fp16_flag": fp16_set,
        "dynamic_inputs_pinned": dynamic,
        "layers": network.num_layers,
        "parse_s": round(parse_s, 1),
        "build_s": round(build_s, 1),
        "size_bytes": target.stat().st_size,
    }
    Path(str(target) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


class Runner:
    def __init__(self, plan: Path) -> None:
        import tensorrt as trt
        import torch

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialise {plan}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"no execution context for {plan}")
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = dtypes[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = tuple(1 if d < 0 else d for d in shape)
                self.context.set_input_shape(name, shape)
                t = torch.rand(shape, dtype=dtype, device="cuda")
                self.inputs[name] = t
                self.context.set_tensor_address(name, t.data_ptr())
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = dtypes[self.engine.get_tensor_dtype(name)]
                t = torch.empty(shape, dtype=dtype, device="cuda")
                self.outputs[name] = t
                self.context.set_tensor_address(name, t.data_ptr())
        self.device_memory_mb = int(self.engine.device_memory_size_v2) / 1e6

    def run(self, **arrays: np.ndarray) -> dict[str, np.ndarray]:
        """Real inputs in, outputs out (numpy), one synchronous call."""
        import torch

        for name, arr in arrays.items():
            t = self.inputs[name]
            t.copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(t.dtype))
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 failed")
        stream.synchronize()
        return {k: v.float().cpu().numpy() for k, v in self.outputs.items()}

    def time(self, warm: int, runs: int) -> dict:
        import torch

        stream = torch.cuda.current_stream()
        for _ in range(warm):
            self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        ms = []
        for _ in range(runs):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record(stream)
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("execute_async_v3 failed")
            b.record(stream)
            b.synchronize()
            ms.append(a.elapsed_time(b))
        arr = np.asarray(ms)
        return {
            "p50_ms": round(float(np.median(arr)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(arr.min()), 2),
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warm", type=int, default=20)
    p.add_argument(
        "--only", action="append", choices=tuple(NETS), help="net(s) to time; default all"
    )
    p.add_argument("--force", action="store_true", help="rebuild the plans")
    p.add_argument(
        "--real-inputs",
        type=Path,
        default=None,
        help="npz from swap_still.py --dump-inputs: run the real face through the engines, "
        "compare with the fp32 ONNX Runtime outputs (PSNR), save the outputs as PNG beside it",
    )
    args = p.parse_args(argv)

    import torch

    print("gpu:", torch.cuda.get_device_name(0))
    real = np.load(args.real_inputs) if args.real_inputs else None
    rows = []
    for net in NETS:
        if args.only and net not in args.only:
            continue
        onnx_path = onnx_for(net, args.precision)
        target = plan_name(net, args.precision)
        if args.force or not target.exists():
            print(f"building {net} ({onnx_path.name}) -> {target.name} ...", flush=True)
            meta = build(onnx_path, target, args.precision)
            print(
                f"   built in {meta['build_s']} s: {meta['size_bytes'] / 1e6:.0f} MB, "
                f"strongly_typed={meta['strongly_typed_network']} fp16_flag={meta['fp16_flag']} "
                f"layers={meta['layers']}",
                flush=True,
            )
        else:
            meta = json.loads(Path(str(target) + ".json").read_text(encoding="utf-8"))
        runner = Runner(target)
        t = runner.time(args.warm, args.runs)
        row = {
            "net": net,
            "precision": args.precision,
            **t,
            "device_memory_mb": round(runner.device_memory_mb, 1),
            "fp16_flag": meta.get("fp16_flag"),
            "strongly_typed": meta.get("strongly_typed_network"),
            "build_s": meta.get("build_s"),
        }
        rows.append(row)
        print(
            f"{net:10s} p50 {t['p50_ms']:7.2f} ms  p99 {t['p99_ms']:7.2f} ms"
            f"  min {t['min_ms']:7.2f} ms  (activation memory {row['device_memory_mb']} MB)",
            flush=True,
        )
        if real is not None and net in ("inswapper", "gfpgan"):
            import cv2

            if net == "inswapper":
                names = list(runner.inputs)
                target_name = next(n for n in names if runner.inputs[n].dim() == 4)
                source_name = next(n for n in names if runner.inputs[n].dim() == 2)
                out = runner.run(
                    **{target_name: blob_swap(real["in128"]), source_name: real["latent"]}
                )
                img = unblob_swap(next(iter(out.values())))
                ref = real["out128"]
            else:
                name = next(iter(runner.inputs))
                out = runner.run(**{name: blob_enh(real["enh_in"])})
                img = unblob_enh(next(iter(out.values())))
                ref = real["enh_out"]
            row["psnr_vs_ort_fp32_db"] = psnr(img, ref)
            row["max_abs_diff"] = int(np.abs(img.astype(np.int16) - ref.astype(np.int16)).max())
            row["nan"] = bool(np.isnan(next(iter(out.values()))).any())
            png = args.real_inputs.with_name(f"trt_{net}_{args.precision}.png")
            cv2.imwrite(str(png), np.hstack([ref, img]))
            print(
                f"   real face: PSNR vs ORT fp32 {row['psnr_vs_ort_fp32_db']} dB,"
                f" max |diff| {row['max_abs_diff']}, nan={row['nan']} -> {png.name}"
                f" (left ORT fp32, right TensorRT {args.precision})",
                flush=True,
            )
        del runner
        torch.cuda.empty_cache()
    total = sum(r["p50_ms"] for r in rows)
    print(f"sum of p50: {total:.2f} ms")
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
