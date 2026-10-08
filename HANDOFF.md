# Handoff: moving from Video2X (ncnn) to a TensorRT upscaling stack

Written 2026-10-09 for the next Claude Code session or developer. The old handoff (pipelining, Colab/Kaggle notebooks, GPU server scripts, ncnn measurements) is archived unchanged in `docs/legacy-video2x-handoff.md`. Every measurement is in `docs/hypotheses.md`.

## 1. Direction

**Goal:** the fastest, best-looking anime video upscale on modern NVIDIA GPUs.

**Where we are:** the Video2X C++ code in this repo (ncnn + Vulkan) is now the **baseline and is being phased out**. We will not extend it. Work moves to TensorRT/ONNX tools; the first candidate is the AnimeJaNai family (AnimeJaNaiConverterGui / VideoJaNai, plus the models in VSGAN-tensorrt-docker). The work stays in this repo for now.

**Why:** ncnn is limited by kernel efficiency, not by hardware (`docs/hypotheses.md`, T10, T20, analysis section). On a T4, PyTorch fp16 `channels_last` ran the same network 2.85x faster (3.22 vs 1.13 fps). On the RTX PRO 6000 Blackwell one ncnn process used about a quarter of the GPU, and 8 processes were needed to reach 18.2 fps (T23, T24). Community benchmarks for AnimeJaNai compact models on TensorRT report 60-130 fps at 1080p on an RTX 5090. Those are web numbers for different models, not our measurements.

**Stages, each with a gate:**

1. **Spike on Kaggle** (next). Run `spike_tensorrt_kaggle.ipynb` on a T4: TensorRT fp16 vs PyTorch fp16 vs the earlier ncnn figure, same network (`realesr-animevideov3` x4, one 1080p frame), plus PSNR. It has not been run yet. Gate: TensorRT is clearly faster and the picture matches (PSNR high, frames inspected).
2. **Pick the stack.** Try AnimeJaNai models through its TensorRT path; compare with our model on a real clip, end to end (decode, encode, resize included). Decide whether the multi-pass workflow in `docs/research/discord-4k-upscale-guide.md` is worth it. Gate: a named stack and a command line that upscales an episode.
3. **Move over.** Port what is worth keeping from `scripts/server/` (chunking, resume, picture check against a baseline, join with audio). Then delete the Video2X C++ code and its build files and detach from the fork so this becomes a normal repo. Exit criteria for deleting: stage 2 passed and one full episode done with the new stack. The new repo name and layout are not decided.

## 2. What the Video2X work established (kept as evidence)

- ncnn baselines, `realesr-animevideov3`, 1080p source: Kaggle T4 about 1.65 fps per GPU (T21); two T4s about 2.49 fps (T14); RTX PRO 6000 Blackwell 6.4 fps with one process, 18.2 fps with 8 processes on a full episode, about 40 min for 24 min of video (T23, T24).
- ncnn settings do not carry between GPUs: Winograd off helps a T4 and corrupts every frame on the Blackwell (T22). Always check the picture of a new setting against a known-good baseline before trusting its speed.
- 3:2 pulldown removal is worth 20% fewer frames when the source has it (T17, T18). Do it on the whole file, not per chunk (drift against audio, T25).
- The encoder hides GPU gains: at 8K, ncnn 0.77 fps vs CUDA 0.83 fps (T11). Judge any new backend end to end, not only on the raw network.
- PSNR can be high while two images are both wrong. Look at a frame.

**8K tests on the Blackwell server (2026-10-08/09), x4 to 7680x4320, x264 slow crf 18, tile 600:**

| Clip | Processes | Result |
|---|---|---|
| 10 s (300 frames) | 1 | 227 s, 1.32 fps, 61 MB |
| 62 s (1854 frames) | 2 | 762 s, 2.43 fps, 264 MB |

GPU utilisation stayed low (0-31%) in the 10 s run. The CPU encode of 8K is the likely limit; this was not checked.

**Playback lesson:** the 8K files (H.264 High 10, 10-bit) stutter on the user's laptop (Intel UHD CometLake, NVIDIA MX130, 8 threads). Its hardware decoder does 8-bit H.264 up to 4K. A 3840x2160 8-bit H.264 version plays; make a 4K or 1080p copy for viewing. 8K output would be better as HEVC or AV1.

## 3. Current state

- **GPU server** (RTX PRO 6000 Blackwell, `administrator@173.208.247.35`, key `~/.ssh/aivastra_gpu_dev`, no root): the `~/v2x` folder was **deleted on 2026-10-09**, so nothing of ours is installed there. `scripts/server/setup.sh` rebuilds it in 15-30 min. `~/bench` and `~/gpu` in the home directory belong to something else and were left alone. Another user's process held about 37 GB of the GPU.
- **Kaggle:** nothing running. `spike_tensorrt_kaggle.ipynb` is written but unrun. It needs the 10 s clip added as a Dataset, GPU T4 x2, Internet on. Nothing in it has been executed.
- **Local files (untracked, large):** the user's episode `[AniDL] Bleach S14 - 06 ...mkv`, `sample_10s.mp4`, `sample_60s.mkv`, the 8K samples, `sample_60s.4k.mp4` and the 1080p episode result. Do not add them to git. Never run `git add -A`.
- **Git:** `origin` on this machine is the user's fork (`adeshboudhnicedigitals/video2x`); on the Windows machine the roles may be reversed (check `git remote -v`). Push only to the fork. Ask before pushing or publishing.

## 4. Next steps

1. Run the Kaggle spike and paste the cell 5 table into `docs/hypotheses.md` as new T-rows.
2. If it passes: install AnimeJaNaiConverterGui or VideoJaNai (ONNX + TensorRT) on a test GPU, take a model from the VSGAN-tensorrt-docker release list, and time a 10 s clip end to end. Compare with the ncnn numbers above.
3. Decide the target resolution. The old scope was 1080p only; the 4K/8K tests since then were exploratory, and the Discord guide is about 4K. This is open.
4. Open questions from `docs/research/discord-4k-upscale-guide.md`: whether multi-pass (2-4 models, resizing down between passes) beats one pass, and whether the AnimeJaNai models look better than `realesr-animevideov3` on the user's old anime episodes.

## 5. Practical notes

- The user wants short, direct answers and wants to be asked before outward-facing actions.
- Commits use Conventional Commits with the module as scope; end messages with the `Co-Authored-By` line the session provides.
- Server commands: use `ssh -i ~/.ssh/aivastra_gpu_dev -o IdentitiesOnly=yes -o BatchMode=yes administrator@173.208.247.35`. Run long jobs detached (`setsid nohup ... < /dev/null &`) so the ssh session does not stay attached. The key is not in `~/.ssh/config` for this IP.
- The Kaggle and Colab notebooks (`Video2X-Kaggle.ipynb`, `Video2X.ipynb`) and `scripts/server/` still work for the ncnn baseline until stage 3.
