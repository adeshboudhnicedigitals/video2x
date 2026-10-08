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

## To test

Ordered by expected gain at the control setting. "Gain" values are estimates, not measurements.

| # | Hypothesis | How to test | Expected gain / note |
|---|---|---|---|
| H1 | **Duplicate frames:** old TV anime often repeats a drawing for 2-3 frames. Upscale each unique frame once and reuse the result | First measure the share of near-identical consecutive frames with `ffmpeg -vf mpdecimate` or a frame-difference count. Then add a skip in the frame loop and compare pictures (a threshold changes the output slightly) | 1.5-3x if the share is high; less flicker. Highest value, cheapest first step |
| H2 | **Tile-level skipping:** extend H1 to regions: reuse the upscaled tile where the source tile did not change | Needs H1's difference measurement first | Gain on top of H1 for mostly static shots |
| H3 | **Tensor-core backend at x2:** run the network in PyTorch or TensorRT fp16 `channels_last` and compare it end to end at 1080p output | Write a loader for the ncnn `.param/.bin` x2 weights (the official `.pth` is x4 only), then run notebook cell 4.4 with `mode = no encoding` at x2 | Up to 2.85x on the network; large build job |
| H4 | **Newer GPUs:** L4, A100, H100, RTX 4090/5090, RTX PRO 6000 Blackwell | Run the control clip on each, with ncnn and (once H3 exists) with CUDA. Record fps, clock, power | ncnn speed follows shader FLOPs, not TDP. Tensor-core backends gain much more on newer chips |
| H5 | **Newer low precision:** FP8 (Ada, Hopper) or FP4 (Blackwell) in TensorRT | After H3 works, quantise the network and compare quality | Large on supported GPUs; risk to picture quality |
| H6 | **Newer ncnn and cooperative matrix:** the repo pins an old ncnn; newer ncnn can use Vulkan cooperative matrices (tensor cores through Vulkan) | Check the "cooperative matrix" lines printed by ncnn on the T4; update the submodule and run the control clip | Unknown; cheap to test |
| H7 | **Several GPUs and platforms at once:** Kaggle 2x T4 plus a Colab T4, different chunks of one episode | Same chunk workers, started from each platform with different chunk ranges, then one join | About 3x one T4. Quotas limit it (Kaggle about 30 GPU hours a week) |
| H8 | **Two processes per GPU** to fill the idle gaps between dispatches | `devices = [0, 0, 1, 1]` style workers on the 10 s clip | At most about 1.2x. The GPU is power-capped |
| H9 | **Smaller or newer network:** newer anime video models (for example AnimeSR, APISR, SPAN) or a pruned or distilled network | Port the weights, run the control clip, compare pictures | Unknown; quality and speed both unproven |
| H10 | **Less work per frame:** shrink to a height between 720p and 1080p (for example 900p), then x2 | Side-by-side on several frames and motion, then speed | Between 1x and 2.15x depending on the height |
| H11 | **Colour conversion and decode on the GPU** (NVDEC, GPU colour conversion, zero-copy to the network) | Measure the CPU share on a fast GPU first | At most about 10% on a T4; matters more once the network gets faster |
| H12 | **Batching frames** | Only possible in a CUDA backend (ncnn has no batch dimension) | Small on a T4 (compute-bound); retest on a large GPU |
| H13 | **Tile sizes 300-700 and x264 `medium`** | Notebook cell 3.1 with tile sizes `[300, 400, 500, 700]` | A few percent |

## Quality-only items (no speed goal)

- **10-bit path:** the network sees 8-bit RGB, but the user's sources are 10-bit (`yuv420p10le`). Check for banding, then add a 16-bit or float path.
- **Chroma upsampling:** YUV 4:2:0 to RGB uses `SWS_BILINEAR`, which softens colour edges. A different flag changes every frame, so it needs a side-by-side check.

## How to add an entry

Move a hypothesis from "To test" to "Tested" with the measured numbers, the control setting it was compared with, and a verdict (passed, failed, inconclusive). Note what a failed test did and did not show, so the next attempt does not repeat it.
