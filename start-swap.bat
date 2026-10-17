@echo off
rem Road C (2026-10-17): the application from this clone's own environment.
rem Never "uv run" / "uv sync" here: uv removes torch, TensorRT, MediaPipe from .venv.
cd /d "%~dp0"
".venv\Scripts\sha256.exe" %*
