# Handoff: Video2X fork (pipelining, Colab notebook, quality tests)

Written for the next Claude Code session (or any developer) picking this work up on another machine.
Read `CLAUDE.md` first for build commands and architecture; this file covers the state of the work and what is next.

## 1. Who and what

- The user owns the fork `https://github.com/adeshboudhnicedigitals/video2x` and **cannot push to upstream** (`k4yt3x/video2x`). Remote `origin` is upstream (never push there); remote `fork` is theirs (push there). Check `git remote -v` first: on the Linux machine the clone has no `fork` remote and `origin` is the user's fork.
- **Goal (current scope): 1080p output only, with the best picture quality we can get.** Clean, sharp and less blurry, "like modern anime", not like old anime. 4K and 8K are out of scope. 1440p ("2K") may be planned later, but nothing should be built or tuned for it now. Judge every change by 1080p output quality first, then by time to finish. The 4K/8K measurements in section 4 are kept as evidence about bottlenecks, not as targets. Their material is old (about 2008) anime episodes, for example Bleach. Their files are labelled 1080p (measured 1908x1080) but the content looks like an upscaled lower-resolution master. Picture quality matters more than raw speed, but runs must finish in reasonable time.
- **Where it runs:** free Google Colab, Tesla T4, **2 vCPUs** (x264 picks 3 threads). Local machine is Windows with **no compiler, CMake, ffmpeg or GPU on PATH**, so nothing can be built locally. All building and testing happens on Colab. A second machine (Linux, i5-10210U, Intel UHD iGPU, NVIDIA MX130 with no working driver) does build it: static build with bundled deps into `build/video2x-install`, run with `LD_LIBRARY_PATH=build/video2x-install/lib`. There libplacebo runs (0.70 fps at 1080p to 4K, GPU-bound) but Real-ESRGAN hangs at frame 0 on the Intel Vulkan driver (cause not found), so model tests still need Colab.
- The user prefers short, direct answers and wants to be asked before outward-facing actions (pushing, publishing).

## 2. Git state

- Work is on `master` of the fork (branch `feat/frame-pipelining` points at the same commits). Local `master` tracks `origin/master`, so git reports it "ahead"; that is expected.
- Never run `git add -A`. These stay **untracked and uncommitted** on purpose:
  - `AnimePahe_Bleach_-_271_BD_1080p_Judas.mp4` (222 MB) and `Bleach - 309.mkv` (349 MB): the user's episodes, large and not ours to publish.
  - `download.png`: a screenshot of the Step 3D comparison (a frame from the show).
  - `.superpowers/` is git-ignored local scratch (task briefs, reports, one-off notebook edit scripts, a progress ledger). It is not part of the repo.
- Commit style: Conventional Commits with the module as scope; end messages with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. `clang-format` was never available locally, so formatting was done by eye.

## 3. What was built

### 3.1 Pipelined frame loop (C++)

Decode, GPU processing and encode now run on separate threads joined by bounded queues, so a slow CPU encoder no longer stalls the GPU.

- `include/libvideo2x/bounded_queue.h`: header-only blocking queue (`push`, `pop`, `close`, `cancel`).
- `src/encoder.cpp`, `include/libvideo2x/encoder.h`: `mux_mutex_` guards every `av_interleaved_write_frame`; `write_raw_packet` moved into `Encoder`.
- `src/libvideo2x.cpp`, `include/libvideo2x/libvideo2x.h`: `process_frames` dispatches to `process_frames_serial` (original loop) or `process_frames_pipelined`; `init_total_frames` helper; `write_frame` pushes a cloned frame into the encode queue when pipelined.
- CLI: `--queue-size N` (default 4, `0` = original serial loop) in `tools/video2x/src/argparse.cpp`; `VideoProcessor` constructor takes a trailing `queue_size` parameter.
- Pipelining is **skipped when `--hwaccel` is used** (decoder surface pool could be exhausted by queued frames).
- Design and plan: `docs/superpowers/specs/2026-10-07-frame-pipelining-design.md` and `docs/superpowers/plans/2026-10-07-frame-pipelining.md`. Built with per-task reviews plus a final whole-branch review: no Critical or Important findings remained.

**Verification status (be precise):**
- It **builds and runs on Colab**: Step 1 of the notebook compiled it from this fork, and the pipelined path (default `queue_size` 4) has processed clips with correct frame counts (300/300, 150/150). The reported version string is 6.4.0.
- **Not yet measured:** output equivalence between `--queue-size 0` and the default; the speed benefit (notebook Step 3C does this, nobody has run it); pause/abort/bad-input behaviour; ThreadSanitizer.
- Deferred minor review findings: worker-thread exceptions would `std::terminate`; no upper bound on `--queue-size`; `PipelineState::fail` does not set `state_ = Failed` (set by `process()` afterwards, spec updated to say so); an `ENOMEM` in pipelined `write_frame` is not logged; audio/subtitle packets may be muxed earlier relative to video than in serial.
- A pre-existing behaviour was preserved on purpose: neither loop flushes the decoder at end of stream, so the last few buffered decoder frames may be dropped.

### 3.2 Colab notebook `Video2X.ipynb` (repo root)

Based on the upstream notebook and edited with Python/JSON scripts. It has no Colab form fields: every setting is a plain Python variable near the top of its cell. Cells, in order:

| Cell | Purpose |
|---|---|
| 1.1 | GPU check (`nvidia-smi`) |
| 1.2 | **Builds Video2X from source** (`repo_url`, `branch`), instead of installing the 6.2.0 `.deb`; ~10-20 min the first time. Prints whether FFmpeg has NVENC encoders (it does on Colab: `h264_nvenc`, `hevc_nvenc`, `av1_nvenc`) |
| 1.3 | Input file from Colab local disk `/content` (no Google Drive, no widgets): `input_path` ("" uses the only video found), `upload_new_file`, `output_dir` (default `/content/output`). **Local files vanish when the runtime resets** |
| 1.4 | Optional background monitor (GPU clocks, temperature, power, throttle reasons, RAM, video2x memory, CPU load) writing `/content/monitor.csv` |
| 2.1 | Upscale with `video2x`. Processors: `realesrgan`, `realcugan` (model `models-se/pro/nose`, noise `conservative/none/1/2/3`), `libplacebo`. Has `output_height` (default 1080; passes `--height`, so the upscaled frame is resized before encoding; 0 = keep the full upscaled size), `shrink_first_height` (0 = off; e.g. 720 makes a temporary Lanczos-shrunk copy, keeps audio/subtitles, deletes it afterwards), `queue_size`, `benchmark` (no encoding, throwaway output) and NVENC support (a `*_nvenc` codec uses `preset` p1-p7 and `crf` as `cq`). Validates scale/noise combinations the models actually have before running. Defaults: `realesr-animevideov3`, scale 2, `libx264 veryfast crf 20`, libplacebo size 1920x1080 |
| 3.1 | **Speed test** (replaces the old serial-vs-pipelined cell): one short clip run four ways, A no encoding (`--benchmark`), B x264 pipelined, C x264 serial (`--queue-size 0`), D NVENC pipelined, after an untimed warm-up. Prints fps, wall time, frame counts, GPU clock/temperature/power per run, the B/C frame-hash comparison and x264 vs NVENC file sizes. This is the hypothesis test in `docs/superpowers/specs/2026-10-07-cpu-offload-hypothesis-design.md` |
| 3.2 | **Model comparison**: runs animevideov3, Real-CUGAN (conservative / denoise 1 / optional 3) and animevideov3-from-720p on a short sample and shows the same frame side by side with timings; saves `/content/output/model_comparison.png` |
| 3.3 | Results and plots for the 1.4 monitor |
| 4.1 | Frame interpolation (RIFE), marked broken on Colab upstream |
| 4.2-4.4 | Experimental: PyTorch/CUDA feasibility test of the same network at x4 (raw speed, then end-to-end with ffmpeg); 4.4 has `mode`: `with encoding` / `no encoding` |

Older notes in this file use the previous step names: Step 0/1/2/2B are now 1.1-1.4, Step 3A is 2.1, Step 3C is 3.1, Step 3D is 3.2, Step 4 is 3.3, Step 3B is 4.1 and Steps 5A-5C are 4.2-4.4.

Gotchas learned: Colab cells run one at a time, so Step 4 can only run after a job finishes; opening an updated notebook starts a fresh runtime (re-runs the whole build), so changed cells are best pasted into the live session; use `--noise-level=-1` (with `=`) because Boost can read a bare `-1` as an option; the summary line with average FPS is only printed at `--log-level info` or lower; clip cuts with `-c copy` snap to the previous keyframe.

## 4. Measurements so far (Colab T4, 2 vCPU)

Source: user's sample, 1920x1080, 30 fps. Model `realesr-animevideov3`.

| Test | Result |
|---|---|
| 1080p -> 4K (x2), `veryfast` | 1.41-1.43 fps (300 frames in ~3:30) |
| 1080p -> 4K (x2), `slow` preset | ~1.15 fps (encoder-limited) |
| 1080p -> 8K (x4), with encoding | 0.77 fps (ncnn) and 0.83 fps (PyTorch/CUDA): both limited by the CPU encode and colour conversion; GPU thread was blocked on the encoder 47% of the time |
| ncnn `--benchmark` (no encode), x4 | 1.13 fps |
| Monitor during the 4K run | GPU utilization 60-94% (~80% avg); SM clock fell from ~1500 to ~700 MHz as temperature rose 44 -> 81 C; throttle reason "software power cap" (70 W limit), no thermal throttle; video2x memory plateaued ~1.3 GB (no leak); only ~151 MB of VRAM used |
| Raw network, 1080p frame, official x4 weights | ncnn baseline 1.13 fps; PyTorch fp32 0.56; fp16 2.53; fp16 + channels_last **3.22 (2.85x)**; ONNX Runtime CUDA fp16 2.39 |
| PSNR, ncnn output vs CUDA-pipeline output | 48.2 dB (pipeline is correct) |
| Model comparison, 142 frames, x2 | animevideov3 104.1 s (1.36 fps); Real-CUGAN conservative 107.6 s (1.32); Real-CUGAN denoise 1 107.6 s (1.32); **animevideov3 from 720p: 48.4 s (2.93 fps)** |
| Speed test (cell 3.1), 147 frames, 1080p x2 to 4K, 2026-10-07 | A no encoding 1.35 fps; B x264 veryfast pipelined 1.31; C serial 1.26; D `hevc_nvenc` failed (`Invalid encoder`). B is 97% of A, pipelining gives 1.04x, hashes identical |
| Full 10 s sample (298 of 300 frames), 1080p x2 to 4K, x264 veryfast crf 20 | 1.35 fps, 223 s wall |
| Resource monitor over those runs | GPU utilisation 78-81%; throttle reason "software power cap" in 190 of 201 samples; SM clock 840-983 MHz (maximum 1590); power 62-64 W |

Reading: the 8K tests are CPU-bound and say little about the GPU backends. At 4K the GPU is the limit (power-capped T4). Model choice barely changes speed. Shrinking to 720p first is ~2.15x faster (it cuts both input and output pixels by 2.25x; this test cannot say which of those mattered).

Visual impressions (single frame, from the Step 3D image): all models are clearly cleaner than a plain Lanczos resize. Real-CUGAN (conservative or denoise 1) looked crisper (thinner, darker outlines); animevideov3 looked smoother and slightly softer; animevideov3-from-720p looked nearly the same as direct animevideov3. The user said animevideov3 quality is also good. Motion/flicker was not checked.

**Conclusion of the CPU-offload hypothesis (2026-10-08): rejected at this job size.** The encoder costs 3% and pipelining adds 4%, so neither the swscale cache nor NVENC is worth building (see `docs/superpowers/specs/2026-10-07-cpu-offload-hypothesis-design.md`). The T4 is held back by its power cap, not by the CPU. NVENC also does not work as built: the `ffmpeg` program lists the NVENC encoders, but the libavcodec that `video2x` links does not have them (`avcodec_find_encoder_by_name` fails in `tools/video2x/src/argparse.cpp`). Not investigated further. What is left for speed is a faster GPU or a faster network backend (items 4 and 5 below).

**Decision (2026-10-08): the model is `realesr-animevideov3`.** The model question is closed; do not re-run the comparison to choose a model.

**Decision (2026-10-08): the 1080p path is x2 then downscale to 1080p** (not shrink-first). Built: `--width`/`--height` now work for Real-ESRGAN and Real-CUGAN. The filter takes the target size from the encoder context and `conversions::ncnn_mat_to_avframe` resizes with `SWS_LANCZOS | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT` in the same swscale call that converts to the output pixel format, so there is one encode and no intermediate file. Without the options the conversion is unchanged (`SWS_BILINEAR`, same size). Checked on the Linux machine with a 320x180 clip: x2 with `--height 270` gives 480x270, `--width 400` gives 400x226, both together give the exact size; the result matches an ffmpeg Lanczos downscale of the plain x2 output at 47 dB PSNR. **Not yet run on Colab or at 1080p.** The scale factor is not auto-picked (item 3 below proposed that); the user sets scale 2. Side finding: Real-ESRGAN does run on the Intel iGPU at 320x180, so the hang at 1080p there depends on frame size.

Estimate: a 24-minute episode is ~35,000 frames: ~7 h at 1.35 fps, ~3.4 h at 2.9 fps.

## 5. Open items, roughly in priority order

Scope note: with the 1080p-only goal, the useful paths are (a) a 720p source upscaled x2 with Real-ESRGAN/Real-CUGAN, and (b) a source already near 1080p (the user's files measure about 1908x1080) cleaned up and sharpened at 1080p, for example by a x2 model and a downscale back to 1080p. Items about 1440p, 4K or 8K below are deferred.

1. **Step 3A was updated** with Real-CUGAN and shrink-first (done, **not yet run on Colab**; its validation and command building were checked offline with a stub). The 1080p final output is now built (`output_height`, see section 4). Still not added, only proposed: a `benchmark` checkbox (`--benchmark`, throwaway output path) to measure the no-encode ceiling, and a Real-CUGAN-from-720p run in Step 3D.
2. **Finish the evidence:** run Step 3C (serial vs pipelined) and Step 5C with `mode = "no encoding"`; ask for a btop screenshot to see whether both CPU cores saturate (the user noticed GPU ~80% and only 2 cores).
3. **Target-size feature (design agreed in principle, not specced or built):** `video2x --height 1080` (or `--width`, or both) for Real-ESRGAN/Real-CUGAN: auto-pick the smallest valid integer scale for the model, then Lanczos-resize inside the existing `ncnn_mat_to_avframe` swscale call, with get_output_dimensions reporting the target. Valid factors: animevideov3 {2,3,4}; other Real-ESRGAN {4}; Real-CUGAN se {2,3,4}, pro {2,3}, nose {2}; if the target needs more than the max, use the max and Lanczos-upscale the rest with a warning; if the input is already >= target, run the smallest factor and shrink, with a log line. Resolve on a local copy of the config inside `VideoProcessor::process` (not the member). Currently `--width/--height` are ignored for Real-ESRGAN/Real-CUGAN. **Recommendation made to the user:** defer it, because shrink-to-720p-first then x2 gives exactly 1440p and an ffmpeg downscale gives 1080p; build it only if differently sized inputs make that awkward.
4. **Speed ideas, untested:** raise the Real-ESRGAN tile cap (hard-coded 200 in `src/filter_realesrgan.cpp` when the heap budget > 1900 MB; only ~151 MB of VRAM was used; tile borders add ~20% overlap work); move RGB/YUV conversions off the GPU thread; try `h264_nvenc` (needs `cq`/`p1-p7` options instead of `crf`/`preset`, and an NVENC-capable FFmpeg build); a faster Colab GPU (L4/A100) is the largest single lever.
5. **CUDA/TensorRT backend:** the raw network is ~2.85x faster in PyTorch fp16 `channels_last`, but the only end-to-end test was at 8K where the encoder hid any gain. The official `.pth` is x4 only; the repo ships the x2 model only as ncnn `.param/.bin` (a loader would have to read those). Not worth building before items 1-2 show the GPU path is what limits real jobs.

## 6. Practical notes

- Running a notebook edit script from Git Bash: long heredocs with some characters failed, so write the script to a file and run it; set `PYTHONIOENCODING=utf-8` (the default console encoding rejects emoji in cell titles).
- Cells are validated by `ast.parse` after stubbing out `!`/`%` lines; none of them has been run locally.
- Real-CUGAN tile/prepadding settings live in `src/filter_realcugan.cpp`; Real-ESRGAN loads `models/realesrgan/<model>-x<scale>.param`.
- Upstream facts worth knowing: `video2x --help` documents flags; the CLI prints the average-FPS summary only at `info` log level; `--benchmark` skips encoding but still creates an output file.
