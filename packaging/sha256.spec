# -*- mode: python -*-
"""PyInstaller spec for the portable build (day 16, D4).

    uv run pyinstaller packaging/sha256.spec --noconfirm

Produces ``dist/sha256/`` with ``sha256.exe`` and its libraries; the
``scripts/build_exe.py`` wrapper runs this and then lays ``avatars/``,
``models/`` and ``configs/`` beside the executable (the portable layout
``AppPaths.portable_root`` looks for), writes the guide in, and zips it.

What is *not* in the executable, on purpose: the model weights (1.2 GB of
LivePortrait plus the face landmarker; fetched by the scripts, copied by the
wrapper), the avatars (owner data), the configuration (the machine's own).

The fast engine (2026-10-09, ``sha256.avatar.liveportrait_tensorrt``): when
the ``tensorrt`` package is in this environment, its runtime DLL, the plugin
library and the ONNX parser (the three the bindings module links -- without
any one of them ``import tensorrt`` fails) and the builder resources for the
GPU families whose engine is built on the user's machine (RTX 30 and 40:
sm86, sm89) go into ``_internal/tensorrt_libs``, where the package's own
loader finds them. The RTX 50 family (sm120) needs no builder: its engine,
built here, travels under ``models/tensorrt`` (the wrapper). The other
builder resources (data-centre and older families, 2.5 GB that PyInstaller's
TensorRT hook would otherwise collect whole) are filtered out of the
binaries below; such a card runs the torch networks. The ONNX exporter's
packages (``onnx``, ``onnxscript``) stay out too: the export travels as a
file.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 -- PyInstaller's global
SRC = ROOT / "src"

hidden = []
for package in ("sha256", "mediapipe", "cv2", "pyvirtualcam", "torch", "torchvision"):
    with contextlib.suppress(Exception):  # a package not installed in this environment
        hidden += collect_submodules(package)

datas = []
for package in ("mediapipe",):
    with contextlib.suppress(Exception):
        datas += collect_data_files(package)
# The window's own assets (background image, if any) travel next to the exe
# via the wrapper, not inside the bundle.

# The fast engine's DLLs, by name, into the package's own folder.
# The RTX 20 series (sm75, 2026-10-18, his friend's machine unknown), 30 and 40.
TENSORRT_BUILDER_FAMILIES = ("75", "86", "89")
binaries = []
tensorrt_dlls: set[str] = set()
_libs_spec = importlib.util.find_spec("tensorrt_libs")
if _libs_spec is not None and _libs_spec.origin:
    _libs = Path(_libs_spec.origin).resolve().parent
    try:
        _major = importlib.metadata.version("tensorrt_cu12").split(".")[0]
    except importlib.metadata.PackageNotFoundError:
        _major = "11"
    tensorrt_dlls = {
        f"nvinfer_{_major}.dll",
        f"nvinfer_plugin_{_major}.dll",  # the bindings module links it
        f"nvonnxparser_{_major}.dll",
    } | {
        f"nvinfer_builder_resource_sm{family}_{_major}.dll"
        for family in TENSORRT_BUILDER_FAMILIES
    }
    for _name in sorted(tensorrt_dlls):
        if (_libs / _name).is_file():
            binaries.append((str(_libs / _name), "tensorrt_libs"))
    hidden += ["tensorrt", "tensorrt_bindings", "tensorrt_libs"]

block_cipher = None

a = Analysis(  # noqa: F821
    [str(SRC / "sha256" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "IPython", "pytest", "onnx", "onnxscript"],
    noarchive=False,
    optimize=0,
)

# PyInstaller's own TensorRT hook collects every DLL of tensorrt_libs (3.4 GB);
# keep the ones named above and drop the rest.
if tensorrt_dlls:
    a.binaries = [
        entry
        for entry in a.binaries
        if not (
            Path(entry[0]).parts[0] == "tensorrt_libs"
            and Path(entry[0]).name.lower().endswith(".dll")
            and Path(entry[0]).name not in tensorrt_dlls
        )
    ]

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sha256",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="sha256",
)
