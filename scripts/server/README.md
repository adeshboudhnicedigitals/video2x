# Running Video2X on a Linux GPU server (no root, one folder)

Everything lives in `~/v2x` (or `$V2X_HOME`): a private package manager, the build tools and
libraries, the build, the installed program, every cache and the work files. Nothing is installed
system-wide. The only requirement on the host is an NVIDIA driver with its Vulkan part
(`/etc/vulkan/icd.d/nvidia_icd.json`).

## Set up (about 15-30 minutes, about 4-5 GB)

```bash
git clone https://github.com/adeshboudhnicedigitals/video2x.git ~/v2x/src
bash ~/v2x/src/scripts/server/setup.sh
. ~/v2x/env.sh            # in every new shell
```

`setup.sh` ends with a small test upscale. It prints the GPU's cooperative-matrix line
(`fp16-8x8x16/...`) and the ncnn options line from the fork's patch. After a successful build,
`rm -rf ~/v2x/build` frees about 1-2 GB.

## Benchmark one GPU

```bash
python3 ~/v2x/src/scripts/server/bench.py episode.mkv --preset ncnn    # ncnn settings (T20 on this GPU)
python3 ~/v2x/src/scripts/server/bench.py episode.mkv --preset tiles   # tile 400-1920
python3 ~/v2x/src/scripts/server/bench.py episode.mkv --preset procs   # 1, 2, 4, 8 processes on the GPU
```

Each run uses the first 20 s of the input (pulldown removed), times only decoding and the GPU
work (`--benchmark`), checks that every process used the requested ncnn settings, and compares
48 frames losslessly against the first setting (PSNR). Results are printed and saved as CSV in
`~/v2x/work/bench`. Add `--repeats 2` to run each setting twice in alternating order.

## Upscale an episode

```bash
python3 ~/v2x/src/scripts/server/upscale.py episode.mkv --procs-per-gpu 4
python3 ~/v2x/src/scripts/server/upscale.py episode.mkv --limit-seconds 60 --chunk-seconds 15   # quick test
```

Defaults: `realesr-animevideov3` x2 resized to 1080p, tile 600, Winograd off and fp16 math,
x264 `slow` crf 18, 3:2 pulldown removed (`--no-remove-pulldown` for native 29.97/30 fps video),
60 s chunks. Finished chunks are kept in `~/v2x/work/<name>/out`, so rerunning resumes. The result
is written to `~/v2x/work/<name>/<name>.1080p.mkv` (or `-o`).

## Clean up

```bash
bash ~/v2x/src/scripts/server/cleanup.sh
```

It shows what is in `~/v2x`, asks, and deletes it. Copy your results out first. It also reports
(without deleting) any cache folders in your home directory that tools may have created when
`env.sh` was not sourced.
