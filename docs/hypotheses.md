# Speed hypotheses log

Goal: make this fork use newer GPU technology as far as possible, so that anime upscaled to 1080p runs as fast as the hardware allows without losing picture quality. This file records every hypothesis we have tested (passed or failed) and the ones still to test, so no result is lost and no test is run twice.

## Control setting (do not change when comparing)

Every speed number is compared against this one setting, so results from different ideas and different GPUs stay comparable.

| Item | Value |
|---|---|
| Input | the user's 1908x1080 Bleach episodes, first 10 s (300 frames) as the standard clip |
| Model | Real-ESRGAN `realesr-animevideov3`, scale 2 |
| Output | each upscaled frame resized (Lanczos) to height 1080, so 1908x1080 |
| Encoder | libx264 `slow`, crf 18 (speed tests that must isolate the GPU use `--benchmark`) |
| Tiles | `--realesrgan-tile-size 400` |
| Pipeline | `--queue-size 4` |

**What to record for each run:** GPU model and count, driver, ncnn version, git commit, fps (frames / wall seconds), the GPU clock, power and temperature during the run, and whether the run encoded.

**Quality rule:** a speed gain only counts if the picture is not worse. Compare against the control output side by side on several frames and in motion.

## Tested

"Result" is what was measured; the "Verdict" says what we conclude.

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| T1 | Running decode, GPU and encode on separate threads speeds things up | 1080p to 4K: serial 1.26 fps, pipelined 1.31 fps (+4%); output hashes identical | Passed, small. The CPU was never the main cost at this size |
| T2 | The 2-vCPU CPU is the bottleneck | At 4K it is not: the pipelined run reached 97% of the no-encode ceiling. At 8K (x4) it is: the GPU thread was blocked on the encoder 47% of the time | Rejected at the 1080p goal, confirmed at 8K |
| T3 | Encoding on the GPU (NVENC) removes the encoder cost | The `video2x` build cannot use it: the FFmpeg libraries it links have no NVENC encoders (`ffmpeg` the program does). The ceiling is only 3-10% anyway | Failed, not worth fixing at this size |
| T4 | x264 preset matters for speed | `veryfast` 97% of the ceiling at 4K; `slow` at 1080p output 90% | Passed: `slow` costs about 7-10% |
| T5 | The T4 is limited by its power cap | Throttle reason "software power cap" in 190 of 201 samples; SM clock 840-983 MHz (max 1590); 62-64 W of 70 W; no thermal throttling | Confirmed. This is the main limit on a T4 |
| T6 | Memory (VRAM or RAM) limits speed or leaks | About 151 MB VRAM used per process; RAM plateaued near 1.3 GB | Rejected |
| T7 | The model choice changes speed | animevideov3 1.36 fps; Real-CUGAN conservative 1.32; Real-CUGAN denoise 1 1.32 | Rejected (all about the same) |
| T8 | Bigger Real-ESRGAN tiles are faster | Tile 400 is 8% faster than the automatic 200; 1000 and 1920 are only 1% faster than 400 | Passed, small (the 400 default) |
| T9 | Shrinking to 720p before upscaling is faster | 142 frames x2: 2.93 fps against 1.36 fps (2.15x). One frame looked nearly the same | Passed for speed, quality check was one frame only. Not adopted: the user chose full x2 for quality |
| T10 | The raw network is faster in PyTorch/CUDA fp16 than in ncnn | 1080p frame, x4 weights: ncnn 1.13 fps; PyTorch fp32 0.56; fp16 2.53; fp16 `channels_last` 3.22 (2.85x); ONNX Runtime CUDA fp16 2.39 | Passed (tensor cores), but fp32 PyTorch is slower than ncnn |
| T11 | A CUDA pipeline is faster end to end | At 8K x4 with encoding: ncnn 0.77 fps, CUDA 0.83 fps. Output matched (PSNR 48.2 dB) | Inconclusive: the encoder hid the gain. The `no encoding` mode and a test at x2/1080p were never run |
| T12 | Frames dropped at the end of the stream | The decoder is now flushed: 300 of 300 frames (298 and 147 of 150 before) | Fixed |
| T13 | Real-CUGAN and animevideov3 differ in look | One frame: Real-CUGAN crisper, animevideov3 smoother. The user chose animevideov3 | Closed; do not re-run the comparison |
| T14 | Two T4s on Kaggle, one `video2x` process per GPU, chunked | 10 s clip, 4 chunks: 300 frames in 120 s = 2.49 fps (the long-run one-GPU figure is about 1.3 fps). Both GPUs at 90%+ when busy | Passed, about 1.9x. Needs a proper one-GPU baseline on the same clip |
| T15 | Vulkan works on Kaggle's T4s | Both Tesla T4 are visible, the build runs | Passed |
| T16 | Local Windows iGPU is usable | Intel UHD: 0.34 fps at 1080p (the Linux hang does not occur on Windows) | Works, far too slow for real use |
| T17 | Duplicate frames exist in the episode (H1 step 1, measured on `AnimePahe_Bleach_-_271`, 1908x1080, **29.97 fps, 43,944 frames**) | 20 s windows at 8 points (480x270 gray, mean absolute difference between consecutive frames below 0.5 of 255): 30-46% duplicate in most places, 71% in one static shot; unique frames 29-71%, average 61%. In three 6 s windows the duplicates fall on every 5th frame (3:2 pulldown from 23.976 fps); the rest are held frames in static shots. `mpdecimate` on the first 10 s kept 187 of 300 frames | Passed: about 20% of frames are pure pulldown repeats and about 1.6x fewer frames are unique on average |
| T18 | The pulldown repeats are exact enough at full resolution that `decimate=cycle=5` drops only true repeats (same file, 8 windows of 10 s, 479 blocks of 5 frames, 1908x1080 luma) | Per block, the frame most like its predecessor: mean difference below 0.3 (of 255) in 98% of blocks, 0.3-1.0 in 1%, 1.0 or more in 1% (cuts or cadence breaks, up to 13). Median difference 0.01-0.05; only 0.03-0.24% of pixels differ by more than 2 levels (the repeats are re-encoded, not bit-identical). The split-and-join command gave 300 frames to exactly 240, 23.976 fps, audio kept, duration 10.09 s against 10.01 s | Passed: removing pulldown is safe for this source, with a real frame dropped in about 1% of blocks. Built as `remove_pulldown` in the Kaggle notebook; its speed gain (expected about 1.25x) is not yet measured on a GPU |
| T19 | The T4 can use its tensor cores through Vulkan, and ncnn uses them for this network | ncnn reports for both Kaggle T4s: `fp16-p/s/u/a=1/1/1/1`, `fp16-8x8x16/16x8x8/16x8x16/16x16x16=0/1/1/1` (cooperative matrices 16x8x8 and 16x16x16 supported). In the pinned ncnn (May 2025) cooperative matrices are on by default, and `convolution_vulkan.cpp` picks the tensor-core Winograd path for 3x3 convolutions with at least 16 channels that are multiples of 8, with fp16 storage and no image storage, which matches this network's 64-channel layers. The Real-ESRGAN wrapper turns fp16 arithmetic off (`use_fp16_arithmetic = false`; ncnn's default is on) | Supported by the code, not yet measured: the A/B cell (Kaggle notebook 3.1) switches cooperative matrices, fp16 arithmetic and Winograd with `patches/librealesrgan-ncnn-options.patch`. On the Intel iGPU (no tensor cores) the switches work and Winograd off is 3.6x slower |
| T20 | ncnn settings change T4 speed (Kaggle cell 3.1, one T4, 480-frame clip with pulldown removed, `--benchmark`, tile 400, 1080p) | Round 1 of 5 settings: control 1.556 fps; cooperative matrices off 1.137 (0.73x); fp16 math on 1.603 (1.03x); fp16 math on with cooperative matrices off 1.185 (0.76x); Winograd off 1.638 (1.05x). Confirmation, 2 rounds in alternating order: control 1.570 (1.576 / 1.564); Winograd off 1.626 (1.035x); fp16 math on 1.622 (1.033x); **Winograd off + fp16 math 1.648 (1.05x)**. PSNR against the control: 59.8-69 dB (invisible). With Winograd off the SM clock averaged about 935 MHz against 757 at the same 66 W | Passed. The tensor cores are already used (turning them off costs 27%). Winograd off plus fp16 math is 5% faster with no visible change; the two gains do not fully add up. Winograd off runs at a higher clock for the same power, which suggests the GEMM path spends less power on data movement (inference, not measured) |
| T21 | The tile size matters with the T20 settings (Kaggle cell 3.1, one T4, same 480-frame clip, Winograd off + fp16 math, one round) | Tile 300: 1.556 fps (0.96x, 28 tiles per 1908x1080 frame); 400: 1.626 (reference, 15 tiles); **600: 1.694 (1.04x, 8 tiles)**; 800: 1.643 (1.01x, 6 tiles, with thin leftover strips of 308 and 280 px). PSNR against tile 400: 59-61 dB. Clock, power and temperature the same for all (about 930-955 MHz, 66 W, 78 C) | Passed, small: tile 600 is the new default in Kaggle cell 2.1. Together with T20 one T4 goes from 1.570 to 1.694 fps (about +8%) with no visible change. The whole range moves speed by only about 4% either way, which supports the kernel-efficiency analysis (H14). Tiles that split the frame evenly (for example 540) are untested |
| T22 | The T4's ncnn settings carry over to an RTX PRO 6000 Blackwell Server Edition (96 GB, 188 SMs, 600 W limit, driver 595.58; `scripts/server/bench.py --preset ncnn`, 480-frame clip with pulldown removed, `--benchmark`, tile 600, one process) | ncnn reports cooperative matrices 16x8x8 and 16x16x16 (`0/1/1/1`). Speed: ncnn defaults 6.46 fps (about 4x one T4); cooperative matrices off 5.32 (0.82x); fp16 math on 5.94 (0.92x); Winograd off 7.24 (1.12x); Winograd off + fp16 math 7.23. **Picture: Winograd off gives 12.8 dB against ncnn defaults, every frame corrupted** (scrambled tiles), with cooperative matrices on or off; fp16 math is fine (58.5 dB). Checked by hand on 5 frames: ncnn defaults correct at tile 200, 400 and 600, with and without the 1080p resize, serial and pipelined. During the run the GPU stayed at about 26% utilisation, 190 W of 600 W, about 2400 MHz | **Failed: the T4 settings are not portable.** ncnn's non-Winograd convolution path is broken on this GPU or driver, so Winograd must stay on; fp16 math is slower here. The best ncnn setting on this GPU is ncnn's own. The server scripts now default to ncnn's settings and check every setting's picture against a known-good baseline (ncnn defaults, automatic tile) before trusting its speed. One process uses about a quarter of the GPU, so processes per GPU (H8) is the next test |
| T23 | Tile size and processes per GPU on the RTX PRO 6000 Blackwell (ncnn defaults, same clip, `--benchmark`, one round each; `bench.py --preset tiles` and `--preset procs`) | Tiles, one process: 200 4.53 fps; 400 6.40; 600 5.01; 1000 5.72; 1500 5.30; 1920 5.99; all pictures correct (71-73 dB against the baseline). Tile 600 ran at 6.46 and 6.44 fps in the other two runs, so one process varies by about 25% between runs; utilisation 22-40%. **Processes on the one GPU (tile 600): 1 6.44 fps; 2 9.46 (1.47x); 4 14.16 (2.20x); 8 16.56 (2.57x)**, with utilisation 34 / 43 / 70 / 85%, power 188 / 233 / 318 / 373 W of 600, SM clock steady at about 2410 MHz, temperature up to 79 C. 8 processes are about 10x one T4 (1.65 fps) | Passed: one process leaves most of this GPU idle (latency-bound, not compute-bound), and several processes fill it, confirming the analysis above (H8). Not power-capped even at 8 processes. Single-process tile results are within run-to-run noise. Untested: more than 8 processes, tile 400 with several processes, and the speed with encoding |
| T24 | Full episode on the RTX PRO 6000 Blackwell (`upscale.py --procs-per-gpu 8 --chunk-seconds 60 --tile 600`, ncnn defaults, x2 resized to 1080p, x264 slow crf 18, pulldown removed; Bleach 271, 1466 s) | Picture check before the job: 73.4 dB against the baseline. Split with pulldown removal (whole file, before any GPU work): 391 s. **Upscaling and encoding: 35,156 frames in 1934 s = 18.2 fps** (25 chunks, 8 processes); about 40 min in total. GPU at 100% utilisation, about 446 W, 2407 MHz, 83 C throughout; CPU load about 10 of 32 cores; the other user's process stayed idle. Output 1908x1080, 23.976 fps, 601 MB, duration 1466.3 s, the same as the input, audio included. Two whole frames inspected: clean, no tile artefacts | Passed. The real job with encoding (18.2 fps) is faster than the 20 s benchmark without encoding (16.6), because the benchmark carries proportionally more start-up time. One episode: about 9.2 h on one T4, about 3.7 h on two T4s, about 40 min here. The up-front split is now the largest serial cost (16% of the run); next: overlap it with the GPU work. Not yet checked: side-by-side with the source, motion and flicker |

## Analysis: why more frames in parallel do not speed up one T4

Written 2026-10-08 while the tile sweep was running. The numbers are a back-of-envelope estimate, not a profile; H14 is the test that checks them.

**Frames are already processed in parallel, across GPUs.** The Kaggle notebook runs one `video2x` process per GPU on different chunks (T14). Within one GPU the parallelism is inside the frame: a 1080p frame is about 2 million pixels, and every convolution layer computes 64 channels for each of them, so one frame already spreads over all 40 SMs of a T4. A second frame at the same time shares the same SMs. It only raises throughput if one frame leaves SMs idle.

**VRAM is capacity, not compute.** Holding more frames in VRAM does not add arithmetic units. VRAM only helps when its lack forces waste: tiles so small that the border overlap dominates, or batches too small to fill the GPU. The T4 job uses about 150 MB (T6), so neither applies.

**What the T4 is limited by (estimate).** Assuming `realesr-animevideov3` is the usual compact network (16 convolution layers at 64 channels, 3x3 kernels):

- Work per 1080p frame: about 16 x 9 x 64 x 64 multiply-adds per pixel, about 1.2 million operations per pixel, so about 2.4 trillion per frame (plus about 10% tile overlap).
- At 1.65 fps (T20) that is about 4 TFLOPS achieved.
- The T4's fp16 tensor-core peak is 65 TFLOPS at 1590 MHz, so about 30-38 TFLOPS at the 750-950 MHz it actually runs at: we use roughly 10-13% of it.
- Memory traffic is also far below the card's 320 GB/s.
- Yet the card sits at its 70 W power cap with about 85% "utilisation" (T5, T20).

So neither the arithmetic units nor the memory are full; the power goes into kernel overhead (instructions per useful operation, synchronisation, data shuffling). nvidia-smi's utilisation only says that some kernel was running, not that the SMs were full. This fits two measurements: PyTorch/cuDNN ran the same network 2.85x faster on the same card (T10), and Winograd off ran at a higher clock for the same power (T20). **Working conclusion: on the T4 the bottleneck is kernel efficiency, not hardware capacity.**

**When more frames in flight would pay off:**

1. **Larger GPUs.** An H100 has 132 SMs and Blackwell parts more. One 400-pixel tile may not fill them, so several tiles or frames at once, or larger tiles, can raise throughput, and their VRAM makes that cheap.
2. **Gaps between dispatches.** ncnn processes a frame tile by tile and waits after each tile, so the GPU idles briefly while the CPU prepares the next one. A second process or stream fills those gaps (H8). On a T4 the power cap limits this: more concurrent work means more power and a lower clock.
3. **Memory-bound networks**, where batching reuses loaded weights. This network is small and its weights stay in cache, so it does not apply here.

**What this means for preparing for better GPUs:** make frames in flight per GPU configurable and measure it (H8); pick the tile size per GPU instead of fixing it (H15); and fix kernel efficiency (H3, then H5), which carries over to every NVIDIA GPU. If the tile sweep (H6) changes speed by only a few percent, that supports kernel efficiency as the limit.

## To test

Ordered by expected gain at the control setting. "Gain" values are estimates, not measurements.

| # | Hypothesis | How to test | Expected gain / note |
|---|---|---|---|
| H1 | **Duplicate frames:** old TV anime often repeats a drawing for 2-3 frames, and this episode has 3:2 pulldown (every 5th frame repeats). Upscale each unique frame once and reuse the result | Step 1 done (T17). Next: (a) inverse telecine before upscaling, ffmpeg `decimate=cycle=5` in the split step (built as `remove_pulldown` in the Kaggle notebook, checked in T18), giving true 23.976 fps and about 20% fewer frames; measure the speed on a GPU; (b) skip held frames in the frame loop and compare pictures (a threshold changes the output slightly) | (a) about 1.25x with no picture change if the repeats are exact; (b) up to about 1.6x on average, 1.4-3.4x by scene. Highest value, cheapest first step |
| H2 | **Tile-level skipping:** extend H1 to regions: reuse the upscaled tile where the source tile did not change | Needs H1's difference measurement first | Gain on top of H1 for mostly static shots |
| H3 | **Tensor-core backend at x2:** run the network in PyTorch or TensorRT fp16 `channels_last` and compare it end to end at 1080p output | Write a loader for the ncnn `.param/.bin` x2 weights (the official `.pth` is x4 only), then run notebook cell 4.4 with `mode = no encoding` at x2 | Up to 2.85x on the network; large build job |
| H4 | **Newer GPUs:** L4, A100, H100, RTX 4090/5090, RTX PRO 6000 Blackwell | Run the control clip on each, with ncnn and (once H3 exists) with CUDA. Record fps, clock, power | ncnn speed follows shader FLOPs, not TDP. Tensor-core backends gain much more on newer chips |
| H5 | **Newer low precision:** FP8 (Ada, Hopper) or FP4 (Blackwell) in TensorRT | After H3 works, quantise the network and compare quality | Large on supported GPUs; risk to picture quality |
| H6 | **ncnn settings and version:** Winograd off plus fp16 arithmetic is measured (T20, 1.05x) and is now the default in Kaggle cell 2.1 (`ncnn_options`). Next: re-tune the tile size with these settings, then try a newer ncnn | Kaggle cell 3.1 now runs a tile sweep (300, 400, 600, 800) with the T20 settings by default; then bump the ncnn submodule and repeat T20 | A few percent from tiles; a newer ncnn is unknown |
| H7 | **Several GPUs and platforms at once:** Kaggle 2x T4 plus a Colab T4, different chunks of one episode | Same chunk workers, started from each platform with different chunk ranges, then one join | About 3x one T4. Quotas limit it (Kaggle about 30 GPU hours a week) |
| H8 | **Several processes (frames in flight) per GPU** to fill the idle gaps between tile dispatches. Small on a T4 but the main knob on large GPUs (see the analysis above) | Workers like `devices = [0, 0, 1, 1]` in Kaggle cell 2.1 on the 60 s clip; record fps, clock and power against one process per GPU. Repeat on any larger GPU | At most about 1.2x on a T4 (power-capped); possibly much more on an H100-class GPU |
| H9 | **Smaller or newer network:** newer anime video models (for example AnimeSR, APISR, SPAN) or a pruned or distilled network | Port the weights, run the control clip, compare pictures | Unknown; quality and speed both unproven |
| H10 | **Less work per frame:** shrink to a height between 720p and 1080p (for example 900p), then x2 | Side-by-side on several frames and motion, then speed | Between 1x and 2.15x depending on the height |
| H11 | **Colour conversion and decode on the GPU** (NVDEC, GPU colour conversion, zero-copy to the network) | Measure the CPU share on a fast GPU first | At most about 10% on a T4; matters more once the network gets faster |
| H12 | **Batching frames** | Only possible in a CUDA backend (ncnn has no batch dimension) | Small on a T4 (compute-bound); retest on a large GPU |
| H13 | **Tile sizes 300-700 and x264 `medium`** | Notebook cell 3.1 with tile sizes `[300, 400, 500, 700]` | A few percent |
| H14 | **Kernel efficiency is the T4's real limit** (see the analysis above): ncnn reaches only about 10-13% of the clock-adjusted tensor-core peak | Count the network's operations from the `.param` file instead of assuming the architecture; profile one frame with Nsight Systems or Vulkan timestamps (time per layer, gaps between tiles); compare achieved TFLOPS with cuDNN (T10) | Decides whether to invest in H3 (if the kernels are the limit) or in H8 and H15 (if the gaps are) |
| H15 | **Tile size chosen per GPU** (by VRAM and SM count) instead of a fixed 400 | Run the tile sweep on each new GPU; turn the result into a rule in `src/filter_realesrgan.cpp` (today: 200 on a T4 when the heap budget is above 1900 MB, or `--realesrgan-tile-size`) | A few percent on a T4; more on large GPUs where one tile may not fill the SMs |

## Quality-only items (no speed goal)

- **10-bit path:** the network sees 8-bit RGB, but the user's sources are 10-bit (`yuv420p10le`). Check for banding, then add a 16-bit or float path.
- **Chroma upsampling:** YUV 4:2:0 to RGB uses `SWS_BILINEAR`, which softens colour edges. A different flag changes every frame, so it needs a side-by-side check.

## How to add an entry

Move a hypothesis from "To test" to "Tested" with the measured numbers, the control setting it was compared with, and a verdict (passed, failed, inconclusive). Note what a failed test did and did not show, so the next attempt does not repeat it.
