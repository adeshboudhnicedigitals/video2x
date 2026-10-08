#!/usr/bin/env python3
"""Upscale a video in chunks with several video2x processes per GPU, then join the chunks.

The same approach as the Kaggle notebook (cells 2.1 and 2.2), as a command-line tool:
1. Cut the video stream into chunks (removing 3:2 pulldown by default).
2. Run `--procs-per-gpu` video2x processes on each GPU; each takes the next unfinished chunk and
   its output frame count is checked against the chunk.
3. Join the finished chunks with the original audio and subtitles.

Finished chunks are kept in the work folder, so running the same command again resumes.

Example (on the server, after `. ~/v2x/env.sh`):
    python3 ~/v2x/src/scripts/server/upscale.py episode.mkv --procs-per-gpu 2
    python3 ~/v2x/src/scripts/server/upscale.py episode.mkv --limit-seconds 60 --chunk-seconds 15
"""

import argparse
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import v2x_common as common  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, help="default: <work>/<input name>.1080p.mkv")
    parser.add_argument("--work", type=pathlib.Path, help="work folder (default: $V2X_HOME/work/<input name>)")
    parser.add_argument("--devices", help="Vulkan device indices, e.g. 0 or 0,1 (default: every discrete GPU)")
    parser.add_argument("--procs-per-gpu", type=int, default=1)
    parser.add_argument("--chunk-seconds", type=float, default=60)
    parser.add_argument("--limit-seconds", type=float, default=0, help="only the first N seconds (0 = whole file)")
    parser.add_argument("--no-remove-pulldown", action="store_true",
                        help="keep every frame (use for video that is natively 29.97/30 fps)")
    parser.add_argument("--precise-split", action="store_true",
                        help="cut at exact times by re-encoding (always on when removing pulldown)")
    parser.add_argument("--model", default="realesr-animevideov3")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--height", type=int, default=1080, help="final height (0 = keep the upscaled size)")
    parser.add_argument("--tile", type=int, default=common.DEFAULT_TILE)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--queue-size", type=int, default=4)
    parser.add_argument("--ncnn", default=json.dumps(common.BEST_NCNN),
                        help='ncnn options as JSON, e.g. \'{"VIDEO2X_NCNN_WINOGRAD": "0"}\'; "{}" for ncnn defaults')
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        sys.exit(f"Input not found: {source}")
    ncnn = json.loads(args.ncnn)
    remove_pulldown = not args.no_remove_pulldown

    if args.devices:
        devices = [int(part) for part in args.devices.split(",")]
    else:
        devices = [index for index, _ in common.discrete_devices()]
    if not devices:
        sys.exit("video2x sees no discrete GPU. Check `video2x --list-devices` and VK_ICD_FILENAMES.")
    names = {index: name for index, name, _ in common.gpu_devices()}
    print("GPUs:", ", ".join(f"{d}: {names.get(d, '?')}" for d in devices), f"x {args.procs_per_gpu} process(es) each")
    print("ncnn options:", ncnn or "ncnn defaults")

    home = pathlib.Path(os.environ.get("V2X_HOME", pathlib.Path.home() / "v2x"))
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)[:40]
    work = (args.work or home / "work" / (safe_stem + ("_test" if args.limit_seconds else ""))).resolve()
    src_dir, out_dir = work / "src", work / "out"
    src_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Split ---
    marker = src_dir / "split.done"
    settings = json.dumps({"input": str(source), "limit": args.limit_seconds, "chunk": args.chunk_seconds,
                           "pulldown": remove_pulldown, "precise": args.precise_split})
    if not marker.exists() or marker.read_text() != settings:
        for old in list(src_dir.glob("chunk_*.mkv")) + list(out_dir.glob("chunk_*")):
            old.unlink()
        marker.unlink(missing_ok=True)
        print("Splitting" + (" and removing pulldown" if remove_pulldown else "") + "...")
        started = time.time()
        result = common.run(common.split_command(source, src_dir / "chunk_%03d.mkv", args.chunk_seconds,
                                                 args.limit_seconds, remove_pulldown, args.precise_split))
        if result.returncode != 0:
            sys.exit("Splitting failed:\n" + result.stderr[-1500:])
        marker.write_text(settings)
        print(f"  done in {time.time() - started:.0f} s")
    chunks = sorted(src_dir.glob("chunk_*.mkv"))
    if not chunks:
        sys.exit("The split produced no chunks.")
    lengths = {chunk: common.duration_of(chunk) for chunk in chunks}
    total_seconds = sum(lengths.values())
    print(f"{len(chunks)} chunks, {total_seconds / 60:.1f} min of video")

    # --- 2. Workers ---
    todo = queue.Queue()
    for chunk in chunks:
        if not (out_dir / chunk.name).exists():
            todo.put(chunk)
    if len(chunks) - todo.qsize():
        print(f"{len(chunks) - todo.qsize()} chunk(s) already finished; skipping them.")

    lock = threading.Lock()
    processes = []
    state = {"done_frames": 0, "done_seconds": sum(lengths[c] for c in chunks if (out_dir / c.name).exists()),
             "failed": [], "running": {}}
    env = common.env_with(ncnn)

    def worker(device, slot):
        while not state["failed"]:
            try:
                chunk = todo.get_nowait()
            except queue.Empty:
                return
            part = out_dir / (chunk.stem + ".partial.mkv")
            log = out_dir / (chunk.stem + ".log")
            part.unlink(missing_ok=True)
            with lock:
                state["running"][f"{device}.{slot}"] = chunk.name
            try:
                command = common.video2x_command(chunk, part, device, args.model, args.scale, args.height, args.tile,
                                                 args.codec, args.preset, args.crf, args.queue_size)
                with open(log, "w") as handle:
                    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
                    with lock:
                        processes.append(process)
                    code = process.wait()
                common.check_ncnn_options(log.read_text(errors="replace"), ncnn, chunk.name)
                expected = common.count_frames(chunk)
                produced = common.count_frames(part) if part.exists() else -1
                if produced != expected:
                    raise RuntimeError(f"{chunk.name}: {produced} frames written, {expected} expected "
                                       f"(exit code {code}); see {log}")
                part.rename(out_dir / chunk.name)
                with lock:
                    state["done_seconds"] += lengths[chunk]
                    state["done_frames"] += expected
            except Exception as error:  # noqa: BLE001 - report every failure and stop the others
                with lock:
                    state["failed"].append(str(error))
            finally:
                with lock:
                    state["running"].pop(f"{device}.{slot}", None)

    threads = [threading.Thread(target=worker, args=(device, slot), daemon=True)
               for device in devices for slot in range(args.procs_per_gpu)]
    started = time.time()
    try:
        for thread in threads:
            thread.start()
        last = 0.0
        while any(thread.is_alive() for thread in threads):
            time.sleep(2)
            if time.time() - last >= 30:
                last = time.time()
                with lock:
                    done, frames, running = state["done_seconds"], state["done_frames"], dict(state["running"])
                elapsed = time.time() - started
                finished = sum(1 for c in chunks if (out_dir / c.name).exists())
                rate = f", {frames / elapsed:.2f} fps" if frames else ""
                eta = f", about {(total_seconds - done) / (done / elapsed) / 60:.0f} min left" if done and frames else ""
                print(f"[{elapsed / 60:6.1f} min] {finished}/{len(chunks)} chunks{rate}{eta}; running {running}")
                if shutil.which("nvidia-smi"):
                    gpus = common.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw,clocks.sm,"
                                       "temperature.gpu", "--format=csv,noheader,nounits"]).stdout.strip()
                    print(f"             GPU (index, util %, W, MHz, C): {gpus.replace(chr(10), ' | ')}")
    except BaseException as error:  # Ctrl+C or a bug in the status loop: stop the workers too
        state["failed"].append(f"stopped by {type(error).__name__}: {error}")
    finally:
        if state["failed"]:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
    if state["failed"]:
        sys.exit("Stopped:\n" + "\n".join(state["failed"]))
    elapsed = time.time() - started
    if state["done_frames"]:
        print(f"Upscaled {state['done_frames']} frames in {elapsed:.0f} s: {state['done_frames'] / elapsed:.2f} fps "
              f"on {len(devices)} GPU(s) x {args.procs_per_gpu} process(es)")

    # --- 3. Join ---
    finished = [out_dir / chunk.name for chunk in chunks]
    list_file = work / "concat.txt"
    list_file.write_text("".join(f"file '{path}'\n" for path in finished))
    output = (args.output or work / f"{source.stem}.{args.height or 'upscaled'}p"
              f"{'.test' if args.limit_seconds else ''}.mkv").resolve()
    output.unlink(missing_ok=True)

    def mux(with_subtitles):
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
                   "-i", str(source), "-map", "0:v:0", "-map", "1:a?"]
        if with_subtitles:
            command += ["-map", "1:s?"]
        command += ["-c", "copy"]
        if args.limit_seconds:
            command += ["-t", str(args.limit_seconds)]
        return common.run(command + [str(output)])

    result = mux(True)
    if result.returncode != 0:
        print("Copying the subtitles failed; retrying without subtitles.")
        output.unlink(missing_ok=True)
        result = mux(False)
    if result.returncode != 0:
        sys.exit("Joining failed:\n" + result.stderr[-1500:])
    expected = min(common.duration_of(source), args.limit_seconds) if args.limit_seconds else common.duration_of(source)
    actual = common.duration_of(output)
    width, height = common.video_size(output)
    print(f"Written: {output}")
    print(f"  {width}x{height}, {output.stat().st_size / 1e6:.0f} MB, {actual:.1f} s (input {expected:.1f} s)")
    if abs(actual - expected) > 1.0:
        print("Warning: the duration differs from the input by more than a second.")


if __name__ == "__main__":
    main()
