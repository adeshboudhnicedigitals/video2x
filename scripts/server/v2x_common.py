"""Helpers shared by upscale.py and bench.py (video2x and ffmpeg command lines, probing, GPU stats)."""

import json
import os
import re
import shutil
import subprocess
import threading

# ncnn options read by the fork's patched Real-ESRGAN wrapper (patches/librealesrgan-ncnn-options.patch)
NCNN_SWITCHES = ("VIDEO2X_NCNN_COOPMAT", "VIDEO2X_NCNN_FP16_ARITH", "VIDEO2X_NCNN_WINOGRAD")

# Fastest settings measured on a T4 (docs/hypotheses.md, T20 and T21)
BEST_NCNN = {"VIDEO2X_NCNN_WINOGRAD": "0", "VIDEO2X_NCNN_FP16_ARITH": "1"}
DEFAULT_TILE = 600


def run(cmd, check=False, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, check=check, **kwargs)


def video2x_binary():
    return os.environ.get("VIDEO2X", "video2x")


def count_frames(path):
    out = run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
               "stream=nb_read_frames", "-of", "csv=p=0", str(path)], check=True).stdout.strip()
    return int(out)


def duration_of(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", str(path)], check=True).stdout.strip())


def video_size(path):
    info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "json", str(path)], check=True).stdout)["streams"][0]
    return int(info["width"]), int(info["height"])


def gpu_devices():
    """Vulkan devices that video2x can use, as (index, name, type)."""
    result = run([video2x_binary(), "--list-devices"])
    found, current = [], None
    for line in (result.stdout + result.stderr).splitlines():
        match = re.match(r"^(\d+)\.\s+(.+)$", line)
        if match:
            current = (int(match.group(1)), match.group(2).strip())
        elif current and "Type:" in line:
            found.append((current[0], current[1], line.split("Type:", 1)[1].strip()))
            current = None
    return found


def discrete_devices():
    return [(index, name) for index, name, kind in gpu_devices() if "Discrete" in kind or "Virtual" in kind]


def env_with(ncnn_options):
    env = os.environ.copy()
    for key in NCNN_SWITCHES:
        env.pop(key, None)
    env.update({key: str(value) for key, value in ncnn_options.items() if key in NCNN_SWITCHES})
    return env


def video2x_command(source, destination, device, model="realesr-animevideov3", scale=2, height=1080,
                    tile=DEFAULT_TILE, codec="libx264", preset="slow", crf=18, queue_size=4, extra=()):
    command = [video2x_binary(), "-i", str(source), "-o", str(destination), "-p", "realesrgan",
               "--realesrgan-model", model, "-s", str(scale), "--device", str(device),
               "--log-level", "info", "--no-progress", "--queue-size", str(queue_size)]
    if tile > 0:
        command += ["--realesrgan-tile-size", str(tile)]
    if height > 0:
        command += ["--height", str(height)]
    if codec:
        command += ["--codec", codec, "-e", f"preset={preset}", "-e", f"crf={crf}"]
    return command + list(extra)


def used_ncnn_options(log_text):
    match = re.search(r"ncnn options: fp16_arithmetic=(\d) cooperative_matrix=(\d) winograd=(\d)", log_text)
    if not match:
        return None
    return {"VIDEO2X_NCNN_FP16_ARITH": match.group(1), "VIDEO2X_NCNN_COOPMAT": match.group(2),
            "VIDEO2X_NCNN_WINOGRAD": match.group(3)}


def check_ncnn_options(log_text, requested, label):
    used = used_ncnn_options(log_text)
    if used is None:
        raise RuntimeError(f"{label}: the log has no '[librealesrgan] ncnn options' line; was the patch applied?")
    for key, value in requested.items():
        if key in NCNN_SWITCHES and used[key] != str(value):
            raise RuntimeError(f"{label}: asked for {key}={value}, the run used {used[key]}")


def split_command(source, destination_pattern, chunk_seconds, limit_seconds=0, remove_pulldown=True,
                  reencode=False):
    """ffmpeg command that cuts the video stream into chunks (video only; audio is added back at the join)."""
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    if limit_seconds:
        command += ["-t", str(limit_seconds)]
    command += ["-i", str(source), "-map", "0:v:0"]
    if remove_pulldown or reencode:
        if remove_pulldown:
            command += ["-vf", "decimate=cycle=5", "-fps_mode", "passthrough"]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "10",
                    "-force_key_frames", f"expr:gte(t,n_forced*{chunk_seconds})"]
    else:
        command += ["-c", "copy"]
    return command + ["-an", "-sn", "-f", "segment", "-segment_time", str(chunk_seconds),
                      "-reset_timestamps", "1", str(destination_pattern)]


class GpuSampler:
    """Samples clock, power, utilisation and temperature of one GPU with nvidia-smi once a second."""

    def __init__(self, nvidia_index):
        self.samples = []
        self.process = None
        if shutil.which("nvidia-smi") and nvidia_index is not None:
            self.process = subprocess.Popen(
                ["nvidia-smi", "-i", str(nvidia_index), "--query-gpu=clocks.sm,power.draw,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits", "-lms", "1000"], stdout=subprocess.PIPE, text=True)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.samples.append([float(part) for part in line.split(",")])
            except ValueError:
                pass

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.thread.join(timeout=2)
        if not self.samples:
            return {}
        count = len(self.samples)
        return {"sm_mhz": sum(s[0] for s in self.samples) / count,
                "watts": sum(s[1] for s in self.samples) / count,
                "util_%": sum(s[2] for s in self.samples) / count,
                "temp_c": max(s[3] for s in self.samples)}
