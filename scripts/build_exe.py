"""Build the portable folder (day 16, D4): the frozen application with the
models, the avatars' manifests, the configuration and the guide beside it.

    uv run python scripts/build_exe.py            # dist/sha256-portable/ and .zip
    uv run python scripts/build_exe.py --no-freeze  # lay out beside an existing dist/sha256

What the folder holds, and why each part is where it is:

* ``sha256.exe`` + its libraries -- PyInstaller's output for
  ``packaging/sha256.spec`` (torch's CUDA runtime included; ~2-3 GB).
* ``models/`` -- the face landmarker and LivePortrait's weights, copied from
  this checkout's ``models/`` (fetched by ``scripts/fetch_face_landmarker.py``
  and ``scripts/fetch_liveportrait.py``). Never in git, so never in the
  bundle: copied at build time. With them, when this checkout has them, the
  fast engine's files under ``models/tensorrt/`` (2026-10-09): the ONNX
  export every machine builds its engine from at the first start, and every
  engine already built here, so that GPU family skips the build. The
  TensorRT DLLs themselves travel inside ``_internal`` (the spec).
* ``avatars/`` -- the neural avatars named on the command line (``--avatar``),
  each directory whole (portrait, manifest, profile). None by default: a
  portrait is the owner's to hand out.
* ``configs/default.toml`` -- the committed defaults, unchanged -- and
  ``configs/local.toml`` with the shipped backends (the neural engine, the
  camera, the tracker, the virtual camera through the OBS device; the
  development defaults are the null backends), the kill switch URL from
  ``--switch-url`` and the default avatar, layered over the defaults the
  way a machine's own local.toml is.
* No guide: the tester reads nothing, the owner presents (2026-10-18, his word).

The layout is what ``AppPaths.portable_root`` recognises: a frozen
executable with ``models`` beside it keeps everything in its own folder.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST = REPO_ROOT / "dist"
FROZEN = DIST / "sha256"
PORTABLE = DIST / "sha256-portable"

SHIPPED_OVERRIDES = """
# --- shipped build (scripts/build_exe.py) ----------------------------------
# This machine's file, written by the build: the neural engine, the camera,
# the tracker and the virtual camera through the OBS device. Edit freely.
[engine]
backend = "liveportrait"

[virtual_camera]
backend = "pyvirtualcam"

[camera]
backend = "opencv"

[tracking]
backend = "mediapipe"
"""


def freeze() -> None:
    spec = REPO_ROOT / "packaging" / "sha256.spec"
    print(f"PyInstaller: {spec}")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(spec), "--noconfirm", "--clean"],
        check=True,
        cwd=REPO_ROOT,
    )


def lay_out(avatars: list[str], switch_url: str | None, engine: str = "liveportrait") -> Path:
    if not FROZEN.is_dir():
        raise SystemExit(f"{FROZEN} is not there -- freeze first (or drop --no-freeze)")
    if PORTABLE.exists():
        shutil.rmtree(PORTABLE)
    shutil.copytree(FROZEN, PORTABLE)

    models = REPO_ROOT / "models"
    downloads = models / "downloads"
    if not downloads.is_dir():
        raise SystemExit(f"{downloads} is missing -- run the fetch scripts first")
    if engine == "swap":
        # Road C (2026-10-17): the landmarker and the segmenters, then the
        # swap's three networks as fp16 ONNX only (the fp32 originals and the
        # buffalo_l pack stay home: 1.2 GB the package does not need) with
        # inswapper's cached emap; LivePortrait's weights stay out too.
        shutil.copytree(
            downloads, PORTABLE / "models" / "downloads", ignore=shutil.ignore_patterns("swap")
        )
        swap = downloads / "swap"
        target = PORTABLE / "models" / "downloads" / "swap"
        (target / "buffalo_l").mkdir(parents=True)
        for relative in (
            "inswapper_128-fp16.onnx",
            "gfpgan_1.4-fp16.onnx",
            "buffalo_l/w600k_r50-fp16.onnx",
            "inswapper_emap.npy",
            "SHA256SUMS.txt",
        ):
            source = swap / relative
            if not source.is_file():
                raise SystemExit(
                    f"{source} is missing -- run scripts/probes/swap_onnx_fp16.py first"
                )
            shutil.copy2(source, target / relative)
            print(f"swap file: {relative} ({source.stat().st_size / 1e6:.0f} MB)")
        plan_patterns = ("swap-*-fp16-sm*-trt*.plan*",)
    else:
        for name in ("downloads", "liveportrait"):
            source = models / name
            if not source.is_dir():
                raise SystemExit(f"{source} is missing -- run the fetch scripts first")
            shutil.copytree(source, PORTABLE / "models" / name)
        plan_patterns = (
            "liveportrait-pixels-fp16.onnx*",
            "liveportrait-pixels-fp16-sm*-trt*.plan*",
        )

    engines = models / "tensorrt"
    if engines.is_dir():
        target = PORTABLE / "models" / "tensorrt"
        target.mkdir(parents=True, exist_ok=True)
        for pattern in plan_patterns:
            for file in sorted(engines.glob(pattern)):
                if ".failed." in file.name or file.name.endswith(".part"):
                    continue
                shutil.copy2(file, target / file.name)
                print(f"fast engine file: {file.name} ({file.stat().st_size / 1e6:.0f} MB)")
    metadata = models / "metadata"
    if metadata.is_dir():
        shutil.copytree(metadata, PORTABLE / "models" / "metadata", dirs_exist_ok=True)

    library = PORTABLE / "avatars"
    library.mkdir(exist_ok=True)
    shutil.copy2(REPO_ROOT / "avatars" / "README.md", library / "README.md")
    for avatar_id in avatars:
        source = REPO_ROOT / "avatars" / avatar_id
        if not source.is_dir():
            raise SystemExit(f"avatar {avatar_id!r} is not in {REPO_ROOT / 'avatars'}")
        shutil.copytree(
            source,
            library / avatar_id,
            ignore=shutil.ignore_patterns("cache", "frames", "superseded*"),
        )

    configs = PORTABLE / "configs"
    configs.mkdir(exist_ok=True)
    shutil.copy2(REPO_ROOT / "configs" / "default.toml", configs / "default.toml")
    # The shipped backends go in local.toml, layered over the defaults the
    # way a machine's own file is -- a TOML table may be declared once per
    # file, so nothing is appended to default.toml (the first build did,
    # and the frozen application died on "Cannot declare ('engine',) twice").
    local = SHIPPED_OVERRIDES.replace('backend = "liveportrait"', f'backend = "{engine}"', 1)
    if switch_url:
        # Day 17, his ask: the switch is read every five minutes (and at
        # every Start), not hourly, so a stop lands within minutes.
        local += f'\n[kill_switch]\nurl = "{switch_url}"\ncheck_hours = 0.0833\n'
    if avatars:
        local += f'\n[avatars]\ndefault_avatar_id = "{avatars[0]}"\n'
    (configs / "local.toml").write_text(local, encoding="utf-8")
    # 2026-10-16, his ask: the owner's passphrase travels with the package
    # (the salted key only, see sha256.app.owner), so Add avatar and
    # Delete avatar stay behind Login on every copy.
    owner = REPO_ROOT / "configs" / "local.owner.json"
    if owner.is_file():
        shutil.copy2(owner, configs / "local.owner.json")

    # The window's background photograph, when this checkout has one (it is
    # the owner's, gitignored; the window falls back to a plain colour).
    assets = REPO_ROOT / "assets"
    if assets.is_dir():
        shutil.copytree(assets, PORTABLE / "assets", dirs_exist_ok=True)
    for name in ("captures", "recordings", "logs"):
        (PORTABLE / name).mkdir(exist_ok=True)
    return PORTABLE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-freeze", action="store_true", help="Skip PyInstaller; lay out only.")
    parser.add_argument("--avatar", action="append", default=[], help="An avatar id to include.")
    parser.add_argument("--switch-url", default=None, help="The courtesy switch's file URL.")
    parser.add_argument("--zip", action="store_true", help="Also write dist/sha256-portable.zip.")
    parser.add_argument(
        "--engine",
        choices=("liveportrait", "swap"),
        default="liveportrait",
        help="The engine the package starts with (road C, 2026-10-17): liveportrait with its "
        "weights, or swap with its three fp16 networks and no LivePortrait weights.",
    )
    args = parser.parse_args()
    if not args.no_freeze:
        freeze()
    folder = lay_out(args.avatar, args.switch_url, args.engine)
    print(f"portable folder: {folder}")
    if args.zip:
        archive = shutil.make_archive(
            str(folder), "zip", root_dir=folder.parent, base_dir=folder.name
        )
        print(f"archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
