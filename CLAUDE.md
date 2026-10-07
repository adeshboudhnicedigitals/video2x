# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current work

See `HANDOFF.md` for the state of the pipelining work, the Colab notebook (`Video2X.ipynb`), measurements so far and the open items.

## Overview

Video2X is a C++ video super-resolution and frame-interpolation framework (v6 rewrite). It builds two things from one CMake project: `libvideo2x` (the library, shared by default) and the `video2x` CLI (`tools/video2x`, toggle with `VIDEO2X_BUILD_CLI`). FFmpeg does decode/encode; ncnn+Vulkan or libplacebo does the ML/shader work. There is no test suite.

## Build

Recipes live in `.justfile` (needs `just`; uses PowerShell on Windows, Ninja + clang++ on Unix):

- `just build` — Release build and install into `build/video2x-install`
- `just debug` — Debug build (no install)
- `just static` — static build with bundled ncnn/spdlog/Boost (Unix)
- `just clean`

Raw CMake works too: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel --target install`.

- Third-party deps are git submodules in `third_party/` (ncnn, spdlog, boost, and the k4yt3x `librealesrgan/librealcugan/librife-ncnn-vulkan` wrappers). Run `git submodule update --init --recursive` before building with bundled deps.
- `VIDEO2X_USE_EXTERNAL_{NCNN,SPDLOG,BOOST}` default to **ON** (system packages). The Windows recipe and static builds set them OFF to use the submodules. FFmpeg is always external (on Windows CI it's unpacked into `third_party/ffmpeg-*-shared`).
- CPU tuning flags: `VIDEO2X_ENABLE_NATIVE`, `VIDEO2X_ENABLE_X86_64_V3` (AVX2), `VIDEO2X_ENABLE_X86_64_V4`.
- Models are shipped in `models/` (libplacebo GLSL, realesrgan, realcugan, rife) and installed alongside the binary; `scripts/download_merge_anime4k_glsl.py` regenerates the Anime4K shaders.

## Conventions (from CONTRIBUTING.md)

- Commits follow Conventional Commits with the module as scope, e.g. `fix(encoder): ...`.
- Every user-visible change goes into `CHANGELOG.md` (Keep a Changelog format).
- Format C++ with `clang-format` using the repo's `.clang-format` before committing.

## Architecture

Public headers are in `include/libvideo2x/`, implementations in `src/`, all in namespace `video2x::*`.

**Pipeline** (`VideoProcessor` in `libvideo2x.h` / `src/libvideo2x.cpp`): `process()` opens a `decoder::Decoder`, builds an `encoder::Encoder` from `EncoderConfig`, creates a `Processor` via `processor_factory`, then `process_frames()` runs the decode → process → encode loop. With `queue_size > 0` (default 4, `--queue-size`) that loop is pipelined: a decode thread and an encode thread run beside the calling thread (the GPU stage), connected by `BoundedQueue`s; `queue_size == 0` runs the original serial loop (`process_frames_serial`). All output muxing goes through `Encoder` (`mux_mutex_`). Pipelining is skipped (serial loop) when hardware decoding (`--hwaccel`) is enabled. Non-video streams (audio/subtitles) are passed through as raw packets (`write_raw_packet`). The processor is controllable from other threads via atomic state (`pause/resume/abort`, `get_processed_frames`) — this is what the GUI/CLI progress uses.

**Processor abstraction** (`processor.h`): `Processor` is the abstract base with `ProcessingMode::Filter` (1 frame in → 1 out, upscaling) or `Interpolate` (frame pairs → intermediate frames, via `RIFE`). `ProcessorConfig` carries common fields plus a `std::variant` of per-backend configs (`LibplaceboConfig`, `RealESRGANConfig`, `RealCUGANConfig`, `RIFEConfig`). `processor_factory` maps `ProcessorType` → concrete class: `filter_libplacebo` (Anime4K/GLSL shaders via libplacebo, lives in `libplacebo.cpp` helper), `filter_realesrgan`, `filter_realcugan`, `interpolator_rife`. Adding a backend means: new config struct + variant entry, new `Processor` subclass, factory registration, CLI args.

**Support modules**: `avutils` (FFmpeg RAII/deleters, e.g. `av_frame_deleter`), `conversions` (AVFrame ↔ ncnn::Mat), `fsutils` (`StringType` is `wstring` on Windows / `string` elsewhere — mind this for paths), `logger_manager` (spdlog).

**CLI** (`tools/video2x/src`): `argparse` (Boost.Program_options → configs), `validators`, `vulkan_utils` (GPU enumeration/selection), `newline_safe_sink` + `timer` (logging/progress), `video2x.cpp` (main; runs processing on a thread and polls the keyboard — space pauses/resumes, `q` calls `VideoProcessor::abort`).

Packaging for distros/AppImage is under `packaging/`; user docs (mdBook) under `docs/book`.
