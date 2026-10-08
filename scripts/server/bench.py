#!/usr/bin/env python3
"""Speed and picture benchmarks for one GPU: ncnn settings, tile sizes and processes per GPU.

Every setting runs the same clip (with pulldown removed, like the real job) with `--benchmark`
(no encoding), so only decoding and the GPU work are timed. A setting with "procs": N starts N
video2x processes on the same GPU at once and reports their combined fps.

Picture check: every setting also upscales a few frames losslessly, and they are compared with a
known-good baseline (ncnn defaults, automatic tile size). Below 40 dB the setting is marked
BROKEN: some settings give wrong output on some GPUs (Winograd off on an RTX PRO 6000 Blackwell,
docs/hypotheses.md T22). Above about 50 dB the difference is invisible.

Presets (all on top of --ncnn, default ncnn's own settings):
    ncnn   ncnn defaults vs cooperative matrices off, fp16 math, Winograd off, Winograd off + fp16
    tiles  tile 600, 200, 400, 1000, 1500 and 1920
    procs  1, 2, 4 and 8 processes on the GPU (tile 600)

Examples (after `. ~/v2x/env.sh`):
    python3 ~/v2x/src/scripts/server/bench.py episode.mkv --preset procs
    python3 ~/v2x/src/scripts/server/bench.py episode.mkv --preset tiles --ncnn '{"VIDEO2X_NCNN_FP16_ARITH": "1"}'
    python3 ~/v2x/src/scripts/server/bench.py episode.mkv --configs my_configs.json --repeats 2

A configs file is a JSON object of name -> {"ncnn": {...}, "tile": N, "procs": N}; the first entry
is the speed reference.
"""

import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import v2x_common as common  # noqa: E402


def presets(base):
    return {
        "ncnn": {
            "ncnn defaults": {"ncnn": {}, "tile": 600},
            "coopmat off": {"ncnn": {"VIDEO2X_NCNN_COOPMAT": "0"}, "tile": 600},
            "fp16 math on": {"ncnn": {"VIDEO2X_NCNN_FP16_ARITH": "1"}, "tile": 600},
            "winograd off": {"ncnn": {"VIDEO2X_NCNN_WINOGRAD": "0"}, "tile": 600},
            "winograd off + fp16 math": {"ncnn": common.T4_NCNN, "tile": 600},
        },
        "tiles": {f"tile {tile}": {"ncnn": base, "tile": tile} for tile in (600, 200, 400, 1000, 1500, 1920)},
        "procs": {f"{procs} process(es)": {"ncnn": base, "tile": 600, "procs": procs} for procs in (1, 2, 4, 8)},
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--preset", choices=("ncnn", "tiles", "procs"), default="procs")
    parser.add_argument("--configs", type=pathlib.Path, help="JSON file of settings (overrides --preset)")
    parser.add_argument("--ncnn", default=json.dumps(common.DEFAULT_NCNN),
                        help="base ncnn options as JSON for the tiles and procs presets (default: ncnn defaults)")
    parser.add_argument("--device", type=int, default=0, help="Vulkan device index")
    parser.add_argument("--nvidia-index", type=int, default=None,
                        help="nvidia-smi index of the same GPU for clock/power samples (default: same as --device)")
    parser.add_argument("--clip-seconds", type=float, default=20, help="taken from the start of the input")
    parser.add_argument("--repeats", type=int, default=1, help="2 runs every setting twice, the second round reversed")
    parser.add_argument("--quality-frames", type=int, default=24, help="0 skips the picture check")
    parser.add_argument("--quality-start", type=float, default=30, help="seconds into the input for the picture check")
    parser.add_argument("--no-remove-pulldown", action="store_true")
    parser.add_argument("--model", default="realesr-animevideov3")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--work", type=pathlib.Path, help="default: $V2X_HOME/work/bench")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.input.resolve()
    configs = json.loads(args.configs.read_text()) if args.configs else presets(json.loads(args.ncnn))[args.preset]
    reference = next(iter(configs))
    remove_pulldown = not args.no_remove_pulldown
    nvidia_index = args.device if args.nvidia_index is None else args.nvidia_index
    home = pathlib.Path(os.environ.get("V2X_HOME", pathlib.Path.home() / "v2x"))
    work = (args.work or home / "work" / "bench").resolve()
    work.mkdir(parents=True, exist_ok=True)

    names = {index: name for index, name, _ in common.gpu_devices()}
    print(f"GPU {args.device}: {names.get(args.device, '?')}")

    # --- Speed clip ---
    clip = work / "clip.mkv"
    vf = ["-vf", "decimate=cycle=5", "-fps_mode", "passthrough"] if remove_pulldown else []
    common.run(["ffmpeg", "-y", "-loglevel", "error", "-t", str(args.clip_seconds), "-i", str(source), "-map", "0:v:0",
                *vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-an", "-sn", str(clip)], check=True)
    clip_frames = common.count_frames(clip)
    print(f"Speed clip: {clip_frames} frames")

    # --- Picture check against the known-good baseline ---
    picture = {}
    if args.quality_frames:
        qclip = work / "qclip.mkv"
        start = common.make_check_clip(source, qclip, args.quality_start, args.quality_frames, remove_pulldown)
        safe = common.lossless_run(qclip, work / "q_safe.mkv", args.device, common.SAFE_SETTINGS,
                                   args.scale, args.height, args.model)
        checked = {}
        for index, (name, cfg) in enumerate(configs.items()):
            key = json.dumps({"ncnn": cfg.get("ncnn", {}), "tile": int(cfg.get("tile", common.DEFAULT_TILE))},
                             sort_keys=True)
            if key not in checked:  # settings that differ only in "procs" give the same picture
                out = common.lossless_run(qclip, work / f"q_{index}.mkv", args.device,
                                          {"ncnn": cfg.get("ncnn", {}), "tile": cfg.get("tile", common.DEFAULT_TILE)},
                                          args.scale, args.height, args.model)
                checked[key] = common.psnr_db(safe, out)
            picture[name] = checked[key]
        print(f"Picture check done ({args.quality_frames} frames from {start:.0f} s).")
        broken = [name for name, db in picture.items() if db is None or db < common.BROKEN_BELOW_DB]
        if broken:
            print(f"WARNING: wrong output on this GPU with: {', '.join(broken)}. Their speed is meaningless.")

    # --- Speed ---
    rows = []
    for round_index in range(args.repeats):
        order = list(configs.items())
        if round_index % 2 == 1:
            order.reverse()
        for name, cfg in order:
            procs = int(cfg.get("procs", 1))
            env = common.env_with(cfg.get("ncnn", {}))
            outs = [work / f"bench_{i}.mkv" for i in range(procs)]
            logs = [work / f"bench_{i}.log" for i in range(procs)]
            for path in outs:
                path.unlink(missing_ok=True)
            command = [common.video2x_command(clip, out, args.device, args.model, args.scale, args.height,
                                              int(cfg.get("tile", common.DEFAULT_TILE)), codec=None,
                                              extra=["--benchmark"]) for out in outs]
            sampler = common.GpuSampler(nvidia_index)
            started = time.time()
            running = []
            for cmd, log in zip(command, logs):
                handle = open(log, "w")
                running.append((subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env), handle))
            for process, handle in running:
                process.wait()
                handle.close()
            seconds = time.time() - started
            stats = sampler.stop()
            processed = 0
            for log in logs:
                text = log.read_text(errors="replace")
                common.check_ncnn_options(text, cfg.get("ncnn", {}), name)
                match = re.search(r"Total frames processed: (\d+)", text)
                count = int(match.group(1)) if match else 0
                if count != clip_frames:
                    print(f"Warning: {name}: a process handled {count} of {clip_frames} frames; see {log}")
                processed += count
            row = {"setting": name, "round": round_index + 1, "procs": procs, "fps": processed / seconds,
                   "seconds": seconds, **stats}
            rows.append(row)
            print(f"{name:28s} round {round_index + 1}: {row['fps']:.2f} fps total ({seconds:.0f} s)")
            for path in outs:
                path.unlink(missing_ok=True)

    # --- Table ---
    by_name = {}
    for row in rows:
        by_name.setdefault(row["setting"], []).append(row)
    ref_fps = sum(r["fps"] for r in by_name[reference]) / len(by_name[reference])
    keys = ["fps", "seconds", "sm_mhz", "watts", "util_%", "temp_c"]
    table = []
    for name, group in by_name.items():
        entry = {"setting": name, "procs": group[0]["procs"]}
        for key in keys:
            values = [r[key] for r in group if key in r]
            entry[key] = round(sum(values) / len(values), 2) if values else ""
        entry["speed_vs_reference"] = round(entry["fps"] / ref_fps, 3)
        db = picture.get(name)
        entry["psnr_vs_baseline_db"] = "" if name not in picture else ("?" if db is None else round(db, 1))
        entry["picture"] = "" if name not in picture else (
            "BROKEN" if db is None or db < common.BROKEN_BELOW_DB else "ok")
        table.append(entry)
    columns = list(table[0].keys())
    widths = {c: max(len(c), *(len(str(e[c])) for e in table)) for c in columns}
    print()
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for entry in table:
        print("  ".join(str(entry[c]).ljust(widths[c]) for c in columns))
    results = work / f"results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(results, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    print(f"\nSpeed reference: {reference}. Picture baseline: ncnn defaults, automatic tile. Saved {results}")


if __name__ == "__main__":
    main()
