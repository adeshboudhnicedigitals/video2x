# CPU offload: hypothesis test, swscale cache, NVENC

Status: closed, nothing built. Measured on 2026-10-07 (Colab T4, 1080p x2 to 4K, 147 frames): A 1.35 fps, B 1.31, C 1.26, D failed (`Invalid encoder 'hevc_nvenc'`). B reaches 97% of A, so by the decision rule below neither option is built. The GPU was under its software power cap in 190 of 201 monitor samples. Results are also in `HANDOFF.md` section 4.

Scope: 1080p output only (see `HANDOFF.md`). 1440p, 4K and 8K are out of scope.

## Hypothesis

On Colab (T4, 2 vCPUs) the GPU is idle for part of every frame because CPU work on the GPU-stage thread (input and output colour conversion) and the CPU encoder (x264) compete for two cores. Removing some of that CPU work raises end-to-end fps toward the no-encode ceiling.

Evidence so far: GPU utilisation 60-94% (about 80% average) on the 4K run; the GPU thread was blocked on the encoder 47% of the time in the 8K runs. Neither shows how much of the gap is conversion and how much is encode at 1080p output.

## Step 1: measure (no code change)

Run on Colab with the same 142-frame sample, `realesr-animevideov3`, x2, then the target 1080p output path.

| Run | How | What it tells us |
|---|---|---|
| A. No-encode ceiling | `--benchmark` (skips encoding, still writes a file) | GPU plus conversions only |
| B. Normal | `libx264 veryfast crf 20`, default `--queue-size 4` | Real throughput |
| C. Serial | `--queue-size 0` | Value of the existing pipelining (notebook Step 3C) |
| D. NVENC | `-c h264_nvenc` or `hevc_nvenc` (see below) | Encoder share of the gap |

Read-out: if A is close to B, encode is not the limit and NVENC is pointless. If A is much faster than B, encode is the limit. If A itself is far below the GPU-only speed (raw network 1.13 fps at x4 in the earlier test), the conversions on the GPU thread are the limit. Record GPU utilisation from the Step 2B monitor for each run, and `btop` for per-core CPU use.

Decision rule: build the swscale cache and any conversion offload only if A is clearly below the raw network speed. Build the NVENC path only if D is clearly faster than B.

## Option 1: cache the swscale context

`src/conversions.cpp` calls `sws_getContext` and `sws_freeContext` on every frame in both `convert_avframe_pix_fmt` (line 34) and `ncnn_mat_to_avframe` (line 163). Replace with a cached context (`sws_getCachedContext`, kept per call site and keyed on width, height and formats), freed at shutdown.

- Thread safety: conversions run on the GPU-stage thread only, but the cache must not be a bare global if two processors can run in one process. Prefer a small struct owned by the processor, passed in or held in a `thread_local`.
- Output must stay bit-identical. Verify with frame hashes before and after on the sample.
- Expected gain: unknown. Only worth doing if measurement A shows conversion cost.
- Quality note, separate from speed: the output conversion uses `SWS_BILINEAR` to subsample chroma, and the path is 8-bit BGR24 even for 10-bit sources (the sample decodes as `yuv420p10le`). A better scaler flag (for example `SWS_SPLINE` or `SWS_LANCZOS` with `SWS_ACCURATE_RND`) or a higher bit depth path could improve 1080p quality and banding. Evaluate this separately from the cache.

## Option 2: NVENC

The CLI already passes arbitrary encoder options through (`-c <codec> -e name=value`, applied with `av_opt_set` in `src/encoder.cpp`), so NVENC may need no C++ change. Example to try:

`-c hevc_nvenc -e preset=p7 -e tune=hq -e rc=vbr -e cq=19`

- Requires an FFmpeg build with NVENC. Step 1 of the notebook builds Video2X against the system FFmpeg; the Colab apt FFmpeg may or may not include it. Check `ffmpeg -encoders | grep nvenc` and the headers (`ffnvcodec`). Not verified.
- Unverified: whether `Encoder` picks a compatible pixel format for NVENC and whether `crf`/`preset` defaults it sets would conflict. Test with a short clip before assuming it works.
- Quality: NVENC is usually less efficient than x264 `slow` at the same size. Compare by eye and by size on the sample (same bitrate or same `cq`); the goal is best picture quality, so NVENC is only acceptable if the difference is not visible at the target bitrate.
- No local test is possible here (the MX130 driver is not loaded and the Intel iGPU has no NVENC).

## Out of scope for now

CUDA kernels for conversion, NVDEC (`--hwaccel` disables pipelining), a TensorRT or CUDA network backend, multi-GPU chunking, and the target-size feature. Revisit after the Step 1 numbers.
