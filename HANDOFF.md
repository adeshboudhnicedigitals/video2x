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
  - On the Linux machine: `[AniDL] Bleach S14 - 06 - [1080P][BD][D-A][ZR][X265].mkv` (the user's episode) and `sample_10s.mp4` (its first 10 s, 1920x1080, 29.97 fps, 300 frames; the sample used for the Colab tests). Local test clips and outputs go under `data/`, which is git-ignored.
  - `.superpowers/` is git-ignored local scratch (task briefs, reports, one-off notebook edit scripts, a progress ledger). It is not part of the repo.
- Commit style: Conventional Commits with the module as scope; end messages with the `Co-Authored-By` line the session provides. `clang-format` is not available on the Windows machine (formatting was done by eye there); it is on the Linux machine and was used for the changes of 2026-10-08.

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
| Speed test (cell 3.1), 149 frames, 1080p x2 resized to 1080p, x264 slow crf 18, 2026-10-08 | A no encoding 1.37 fps; B x264 slow 1.23 (90% of A); tile 400 1.48 (1.08x A); tile 1000 1.38; tile 1920 1.39. 149 frames against 147 before the decoder flush fix |

Reading: the 8K tests are CPU-bound and say little about the GPU backends. At 4K the GPU is the limit (power-capped T4). Model choice barely changes speed. Shrinking to 720p first is ~2.15x faster (it cuts both input and output pixels by 2.25x; this test cannot say which of those mattered).

Visual impressions (single frame, from the Step 3D image): all models are clearly cleaner than a plain Lanczos resize. Real-CUGAN (conservative or denoise 1) looked crisper (thinner, darker outlines); animevideov3 looked smoother and slightly softer; animevideov3-from-720p looked nearly the same as direct animevideov3. The user said animevideov3 quality is also good. Motion/flicker was not checked.

**Conclusion of the CPU-offload hypothesis (2026-10-08): rejected at this job size.** The encoder costs 3% and pipelining adds 4%, so neither the swscale cache nor NVENC is worth building (see `docs/superpowers/specs/2026-10-07-cpu-offload-hypothesis-design.md`). The T4 is held back by its power cap, not by the CPU. NVENC also does not work as built: the `ffmpeg` program lists the NVENC encoders, but the libavcodec that `video2x` links does not have them (`avcodec_find_encoder_by_name` fails in `tools/video2x/src/argparse.cpp`). Not investigated further. What is left for speed is a faster GPU or a faster network backend (items 4 and 5 below).

**Changes of 2026-10-08 after the speed test (built and checked on the Linux machine, not yet run on Colab):**

- Decoder flush: both loops in `src/libvideo2x.cpp` now send a null packet at end of file and drain the decoder, so the last frames are no longer dropped (the 298-of-300 and 147-of-150 counts above). A 12-frame clip with B-frames now gives 12 of 12 frames in both the pipelined and the serial loop, with identical hashes.
- `--realesrgan-tile-size N` (0 = automatic, as before; minimum 32). The tile size used is logged at `info` level. On a T4 the automatic size is 200, so a 1080p frame is 60 tiles, each with a 10-pixel border: about 21% extra pixels. Tile 400 against 200 on a 320x180 frame differs slightly (58 dB PSNR), from the tile seams. Measured on the T4 (single runs): tile 400 is 8% faster than 200, while 1000 and 1920 are only 1% faster (cause not known). Cell 2.1 now defaults to `tile_size = 400`; cell 3.1 lists `[300, 400, 500, 700]` to narrow it down.
- Notebook defaults: cell 2.1 uses x264 `slow`, `crf` 18 (was `veryfast`, 20) and has `tile_size`; cell 3.1 runs at 1080p output with the same x264 settings, with the serial and NVENC runs off by default. Measured: `slow` at 1080p output reaches 90% of the no-encode ceiling (`veryfast` at 4K reached 97%), so it costs roughly 7-10% of speed. `medium` is not measured.

**Decision (2026-10-08): the model is `realesr-animevideov3`.** The model question is closed; do not re-run the comparison to choose a model.

**Decision (2026-10-08): the 1080p path is x2 then downscale to 1080p** (not shrink-first). Built: `--width`/`--height` now work for Real-ESRGAN and Real-CUGAN. The filter takes the target size from the encoder context and `conversions::ncnn_mat_to_avframe` resizes with `SWS_LANCZOS | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT` in the same swscale call that converts to the output pixel format, so there is one encode and no intermediate file. Without the options the conversion is unchanged (`SWS_BILINEAR`, same size). Checked on the Linux machine with a 320x180 clip: x2 with `--height 270` gives 480x270, `--width 400` gives 400x226, both together give the exact size; the result matches an ffmpeg Lanczos downscale of the plain x2 output at 47 dB PSNR. **Not yet run on Colab or at 1080p.** The scale factor is not auto-picked (item 3 below proposed that); the user sets scale 2. Side finding: Real-ESRGAN does run on the Intel iGPU at 320x180, so the hang at 1080p there depends on frame size.

Estimate: a 24-minute episode at 23.976 fps is ~35,000 frames: ~7 h at 1.35 fps, ~3.4 h at 2.9 fps. **Correction (2026-10-08):** the user's `AnimePahe_Bleach_-_271` file is 29.97 fps with 3:2 pulldown (43,944 frames in 1466 s), so it is ~9.2 h at 1.33 fps (one T4) or ~4.9 h at 2.49 fps (two T4); see T17 and H1 in `docs/hypotheses.md`.

## 5. Where we stopped (2026-10-08) and what is next

**Speed hypotheses:** every speed idea tested so far (passed or failed) and the backlog of ideas still to test are in `docs/hypotheses.md`, with the fixed control setting every comparison must use. Add new results there. A Kaggle notebook (`Video2X-Kaggle.ipynb`) runs the same 1080p job on two T4s with chunked parallel processing; its first test gave 2.49 fps for 300 frames on 2 GPUs.

### Settled, do not reopen

- **Goal:** 1080p output, best picture quality. Model `realesr-animevideov3`, scale 2, then resize to 1080p (`output_height = 1080`, `--height 1080`).
- **CPU is not the limit on Colab.** The encoder costs 3% (`veryfast`) to 10% (`slow`), pipelining adds 4%. The T4 is held back by its power cap. The swscale cache and NVENC were not built and should not be.
- **NVENC does not work** with this build (the FFmpeg libraries `video2x` links have no NVENC encoders). Not worth fixing for a 3-10% ceiling.
- **Notebook defaults (cell 2.1):** `realesrgan`, `realesr-animevideov3`, scale 2, `output_height = 1080`, `libx264`, `preset = slow`, `crf = 18`, `tile_size = 400`, `queue_size = 4`. Expected speed about 1.33 fps on a T4, which is about 7.2 hours for a 24-minute episode at 23.976 fps (the 271 file is 29.97 fps, so about 9.2 hours) (the Colab free limit is 12 hours).

### Not yet verified

- **Cell 2.1 has not been run on Colab since `output_height`, `tile_size` and the `slow` preset were added.** Cell 3.1 used the same options (`--height 1080`, `--realesrgan-tile-size`) successfully, so the binary side works at 1080p; the cell's own command building was only checked offline. First thing to do next session: run 1.2 (rebuild), 1.3 and 2.1 on the 10 s sample, confirm the output is 1920x1080 with 300 of 300 frames, and look at the picture.
- The picture of the 1080p result has not been looked at by anyone: not against the source, not in motion (flicker), not at tile seams with tile 400.
- Tile sizes between 300 and 700 (cell 3.1 lists `[300, 400, 500, 700]`; optional rerun). All speed numbers are single runs.
- x264 `medium` (between `veryfast` at 97% and `slow` at 90% of the ceiling).

### Next, in the order recommended to the user

1. **Verify the real job** (the three checks above) and let the user judge the 1080p picture against the source.
2. **10-bit path (quality).** The user's sources decode as `yuv420p10le`. `conversions.cpp` converts every frame to 8-bit `BGR24` for the network and back, and the encoder then writes a 10-bit file that holds 8-bit data. This risks banding in gradients (sky, glow). A real fix needs a 16-bit or float path into `ncnn::Mat` and back; medium-sized work. A cheaper first check is whether banding is visible in the output at all.
3. **Input chroma upsampling (quality).** `convert_avframe_pix_fmt` uses `SWS_BILINEAR` for YUV 4:2:0 to BGR, which softens colour edges before the network sees them. Changing the flag is small but changes every output frame, so it needs a side-by-side check.
4. **Speed, only large levers are left:** a faster Colab GPU (L4/A100); two Colab sessions with the episode split in half and joined afterwards (frames are independent for this model, so this scales nearly linearly); shrink to 720p first, x2, resize to 1080p (2.93 fps measured against 1.35, looked nearly the same on one frame, but throws away source detail and the user chose full x2 for quality); a TensorRT/CUDA backend (2.85x on the raw network in PyTorch fp16 `channels_last`; large build; the official `.pth` is x4 only and the x2 model exists only as ncnn `.param/.bin`).

### Known problems, not being worked on

- On the Linux machine, Real-ESRGAN hangs at frame 0 on the Intel UHD Vulkan driver with a 1080p input but runs at 320x180, so it depends on frame size (GPU memory or a driver limit; not investigated). Use small clips for local checks.
- The same machine's NVIDIA MX130 has no working driver.
- RIFE is broken on Colab (notebook cell 4.1). Cells 4.1-4.4 are experiments kept for reference; the user was asked whether to delete them and has not answered.
- The proposed auto-pick of the scale factor for a target size was not built; the user sets the scale and the output height themselves.

## 6. Practical notes

- Running a notebook edit script from Git Bash: long heredocs with some characters failed, so write the script to a file and run it; set `PYTHONIOENCODING=utf-8` (the default console encoding rejects emoji in cell titles).
- Cells are validated by `ast.parse` after stubbing out `!`/`%` lines; none of them has been run locally.
- Real-CUGAN tile/prepadding settings live in `src/filter_realcugan.cpp`; Real-ESRGAN loads `models/realesrgan/<model>-x<scale>.param`.
- Upstream facts worth knowing: `video2x --help` documents flags; the CLI prints the average-FPS summary only at `info` log level; `--benchmark` skips encoding but still creates an output file.
