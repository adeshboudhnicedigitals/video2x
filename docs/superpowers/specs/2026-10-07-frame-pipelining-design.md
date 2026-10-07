# Frame Pipelining Design

Date: 2026-10-07

## Goal

Overlap decoding, GPU processing and encoding in `VideoProcessor::process_frames`
so that a slow CPU encoder (e.g. `libx264 -preset slow` at 4K on few cores) no
longer stalls the GPU, and vice versa. Today the three stages run strictly one
after another for every frame.

## Non-goals

- No change to what is computed: output frames must be identical to the serial path.
- No new GPU backend, no CUDA/TensorRT work, no GPU-side colour conversion.
- No change to the `Processor` / `Filter` / `Interpolator` interfaces.

## Architecture

Three stages connected by bounded blocking queues:

```
decode thread          calling thread (GPU stage)         encode thread
-------------          --------------------------         -------------
av_read_frame
 video pkt -> decode -> [Queue A] -> pop frame
 audio/sub -> mux (locked)          recalc PTS
                                    filter() / interpolate()
                                    push (frame, idx) -> [Queue B] -> pop
                                                                      encoder.write_frame
                                    on EOF: processor->flush()        encoder.flush()
```

- **Decode thread**: reads packets, decodes video frames into Queue A. Audio and
  subtitle packets are muxed directly via `Encoder::write_raw_packet`.
- **GPU stage** runs on the thread that called `process()`, so Vulkan/ncnn use
  stays on one thread as today. It pops from Queue A, performs the work of
  `process_filtering` / `process_interpolation`, and pushes results to Queue B
  instead of calling `write_frame` directly.
- **Encode thread**: pops from Queue B, calls `Encoder::write_frame`, and after
  the queue is drained calls `Encoder::flush()`.

### Queues

- Bounded, blocking on push when full and on pop when empty, closable.
- Closing wakes all blocked callers: push fails, pop returns empty.
- Items own their `AVFrame` via the existing `av_frame_deleter` `unique_ptr`.
  Queue B items also carry the output frame index required by
  `Encoder::write_frame(frame, frame_idx)`.
- Capacity is `queue_size` (default 4). A 4K yuv420p frame is about 12 MB, so
  the default is roughly 50 MB per queue.

### Behaviour that moves

- **PTS recalculation** (`recalculate_pts`) currently runs in the decode loop
  using `frame_idx_`, which also counts interpolated frames. It moves into the
  GPU stage so the value is unchanged.
- **`write_raw_packet`** moves from `VideoProcessor` into `Encoder` and takes a
  `mux_mutex_`, which also guards the `av_interleaved_write_frame` calls inside
  `Encoder::write_frame` and `Encoder::flush`. The decode thread (audio and
  subtitles) and the encode thread (video) otherwise write to the same
  `AVFormatContext` concurrently.

## Configuration

- New `queue_size` constructor parameter (default 4) on `VideoProcessor`
  (there is no general configuration struct), and a `--queue-size` CLI option in
  `tools/video2x/src/argparse.cpp`.
- `queue_size = 0` disables the pipeline and runs the existing serial code path.
  This is the fallback and the baseline for equivalence testing.
- Pipelining is also skipped when hardware decoding (`--hwaccel`) is enabled.

## Pause, abort, errors, progress

- **Pause** is handled in the GPU stage (stops popping while `Paused`, same
  100 ms sleep). Queue A fills and the decode thread blocks on push; the encode
  thread drains Queue B and idles.
- **Abort** stops the decode thread (Queue A is cancelled; the decode thread
  also checks the state between packets). The GPU stage then flushes the
  processor and the encode thread drains Queue B and flushes the encoder, as the
  serial path does, so the truncated output is valid. The trailer is written
  after joins, as today.
- **Errors**: one shared first-error slot (mutex-protected `int`). The failing
  thread stores its error if the slot is empty and closes both queues. Errors
  caused by the shutdown are ignored, so the reported error is the root cause.
  `process_frames` returns the stored error and `process()` sets state `Failed`
  after the threads are joined (a worker setting it could race with a user's
  pause/resume).
- **Joins**: both worker threads are joined on every exit path by an RAII guard,
  so an early return cannot leave a thread running against destroyed objects.
- **Progress**: `frame_idx_` remains the number of frames produced by the GPU
  stage, so the CLI progress bar works unchanged. It may run a few frames ahead
  of what has been encoded. `total_frames_` is unchanged.
- **Benchmark mode**: frames are not pushed to Queue B at all (as the serial
  path skips `Encoder::write_frame`); the encode thread only waits for the
  queue to close and then calls `Encoder::flush()`, as the serial path does.
- **Flush ordering**: decode thread closes Queue A at EOF; GPU stage drains it,
  calls `processor->flush()`, pushes flushed frames, closes Queue B; encode
  thread drains Queue B, then calls `Encoder::flush()`. Threads are joined
  before `av_write_trailer`.

## Testing and verification

The repository has no test suite, and none is added. Verification:

1. Build Debug and Release with the `just` recipes; check warnings.
2. Output equivalence on a short clip: run with `--queue-size 0` and with the
   default, compare `ffmpeg -f framemd5` output (frame count, timestamps,
   hashes) in both filter and interpolation modes. They must match.
3. Pause/resume mid-run, abort mid-run, and a bad input file: no hang, no
   leaked threads. Run once under ThreadSanitizer if available.
4. Compare fps between `--queue-size 0` and the default on the same clip.

Steps 2 to 4 require a working Vulkan GPU.

## Risks

- Muxer locking is the main correctness risk; all writes to the output
  `AVFormatContext` must go through `Encoder` and the mutex.
- Hardware decode (`hw_device_type`): frames queued between stages would keep decoder pool surfaces referenced and can exhaust a fixed-size pool. Resolution: the dispatcher uses the serial loop whenever a hardware device type is set, and logs it.
- Memory grows by up to `queue_size` frames per queue.
