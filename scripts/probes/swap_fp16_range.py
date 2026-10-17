"""Road C: which nodes of a swap network leave fp16's range on a real face.

Runs the fp32 ONNX on ONNX Runtime (CPU) with every intermediate tensor exposed, in
chunks, and lists the nodes whose outputs overflow fp16 (|x| > 65504), come close
(|x| > 1e4) or underflow (0 < |x| < 6e-8 for most of the tensor). Their names go into
the fp16 converter's node block list (scripts/probes/swap_onnx_fp16.py --block-from).

Usage (from the project root):
  .venv\\Scripts\\python.exe -I scripts/probes/swap_fp16_range.py --net inswapper
      --real-inputs <npz> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = PROJECT_ROOT / "models" / "downloads" / "swap"
NETS = {
    "inswapper": MODELS / "inswapper_128.onnx",
    "gfpgan": MODELS / "gfpgan_1.4.onnx",
}
FP16_MAX = 65504.0
FP16_TINY = 6.1e-5  # the smallest normal fp16; below it precision collapses


def blob_swap(bgr128: np.ndarray) -> np.ndarray:
    rgb = bgr128[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def blob_enh(bgr512: np.ndarray) -> np.ndarray:
    rgb = bgr512[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--net", choices=tuple(NETS), required=True)
    p.add_argument("--real-inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=40)
    p.add_argument("--risk", type=float, default=1e4, help="|x| above this is reported as a risk")
    args = p.parse_args(argv)

    import onnx
    import onnxruntime as ort

    real = np.load(args.real_inputs)
    model = onnx.load(str(NETS[args.net]))
    graph = model.graph
    # every node output becomes a graph output, in chunks (memory)
    producers: dict[str, tuple[str, str]] = {}
    for node in graph.node:
        for o in node.output:
            if o:
                producers[o] = (node.name, node.op_type)
    names = list(producers)
    if args.net == "inswapper":
        feeds = {}
        sess0 = ort.InferenceSession(str(NETS[args.net]), providers=["CPUExecutionProvider"])
        for i in sess0.get_inputs():
            feeds[i.name] = blob_swap(real["in128"]) if len(i.shape) == 4 else real["latent"]
    else:
        sess0 = ort.InferenceSession(str(NETS[args.net]), providers=["CPUExecutionProvider"])
        feeds = {sess0.get_inputs()[0].name: blob_enh(real["enh_in"])}
    del sess0

    report = []
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    for start in range(0, len(names), args.chunk):
        chunk = names[start : start + args.chunk]
        m = onnx.ModelProto()
        m.CopyFrom(model)
        del m.graph.output[:]
        for n in chunk:
            m.graph.output.append(onnx.ValueInfoProto(name=n))
        sess = ort.InferenceSession(m.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        outs = sess.run(chunk, feeds)
        for n, arr in zip(chunk, outs, strict=True):
            if not isinstance(arr, np.ndarray) or arr.dtype.kind != "f" or arr.size == 0:
                continue
            a = np.abs(arr.astype(np.float32))
            amax = float(a.max())
            nz = a[a > 0]
            tiny_share = float((nz < FP16_TINY).mean()) if nz.size else 0.0
            node_name, op = producers[n]
            flag = None
            if amax > FP16_MAX:
                flag = "overflow"
            elif amax > args.risk:
                flag = "risk"
            elif tiny_share > 0.5:
                flag = "underflow"
            if flag:
                report.append(
                    {
                        "tensor": n,
                        "node": node_name,
                        "op": op,
                        "max_abs": round(amax, 1),
                        "tiny_share": round(tiny_share, 3),
                        "flag": flag,
                    }
                )
        del sess, outs
        done = min(start + args.chunk, len(names))
        print(f"  {done}/{len(names)} tensors, flagged so far {len(report)}", flush=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    from collections import Counter

    print(
        f"{args.net}: {len(names)} tensors, {len(report)} flagged:",
        Counter((r["op"], r["flag"]) for r in report).most_common(20),
    )
    worst = sorted(report, key=lambda r: -r["max_abs"])[:12]
    for r in worst:
        print(f"   {r['flag']:9s} {r['op']:16s} {r['node'][:40]:40s} max {r['max_abs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
