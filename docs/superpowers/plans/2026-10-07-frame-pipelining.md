# Frame Pipelining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run decoding, GPU processing and encoding on separate threads connected by bounded queues, so a slow CPU encoder no longer stalls the GPU.

**Architecture:** `VideoProcessor::process_frames` becomes a dispatcher. `queue_size_ == 0` runs the existing serial loop unchanged; otherwise a decode thread and an encode thread run beside the calling thread (the GPU stage). The GPU stage reuses `process_filtering` / `process_interpolation` unchanged; their `write_frame` call pushes a frame reference into the encode queue instead of encoding. All output-muxer writes go through `Encoder` behind a mutex.

**Tech Stack:** C++17, FFmpeg (libavcodec/libavformat), `std::thread` / `std::mutex` / `std::condition_variable`, Boost.Program_options, CMake.

**Design spec:** `docs/superpowers/specs/2026-10-07-frame-pipelining-design.md`

## Global Constraints

- Output must be identical to the serial path (same frame count, timestamps and hashes) for filter and interpolation modes.
- `--queue-size 0` must keep the existing serial behaviour.
- Default queue size is 4; it is a `VideoProcessor` constructor parameter and a `--queue-size` CLI option (non-negative integer).
- Vulkan/ncnn processing stays on the thread that called `VideoProcessor::process()`.
- All writes to the output `AVFormatContext` go through `Encoder` and `mux_mutex_`.
- No test framework or tracked test files are added (the repo has none). Verification uses a throwaway test under the git-ignored `build/` directory plus manual runs.
- Commit messages follow Conventional Commits with the module as scope, end with the `Co-Authored-By` trailer, and C++ is formatted with `clang-format` (repo `.clang-format`).
- Do not change decoder end-of-stream behaviour: the serial loop never sends a flush packet to the decoder, and the pipelined path must match it exactly so outputs stay comparable.

## Environment note

This Windows machine had no C++ compiler, CMake or Ninja on `PATH` when the plan was written, and no `ffmpeg`. Builds and runs in this plan must happen in an environment with the toolchain (Linux/Colab with `just build`, or a Windows Visual Studio developer prompt). GPU runs need a working Vulkan device. Where a step needs a compiler, the command is given for both clang/g++ and MSVC.

## File Structure

- Create `include/libvideo2x/bounded_queue.h`: header-only blocking bounded queue (generic, no FFmpeg dependency).
- Modify `include/libvideo2x/encoder.h`, `src/encoder.cpp`: `mux_mutex_`, `Encoder::write_raw_packet`.
- Modify `include/libvideo2x/libvideo2x.h`, `src/libvideo2x.cpp`: queue_size, dispatcher, serial rename, pipelined path.
- Modify `tools/video2x/include/argparse.h`, `tools/video2x/src/argparse.cpp`, `tools/video2x/src/video2x.cpp`: CLI option and wiring.
- Modify `CHANGELOG.md`, `CLAUDE.md`.

---

### Task 1: BoundedQueue

**Files:**
- Create: `include/libvideo2x/bounded_queue.h`
- Create (not committed): `build/bounded_queue_test.cpp`

**Interfaces:**
- Produces: `video2x::BoundedQueue<T>` with
  - `explicit BoundedQueue(size_t capacity)` (capacity 0 is treated as 1)
  - `bool push(T&& item)`: blocks while full; returns `false` and leaves `item` untouched if the queue is closed or cancelled
  - `std::optional<T> pop()`: blocks while empty; returns remaining items after `close()`, then `std::nullopt`
  - `void close()`: no more pushes; consumers drain what is left
  - `void cancel()`: close and discard all queued items; wakes everyone

- [ ] **Step 1: Write the failing test**

Create `build/bounded_queue_test.cpp` (create the `build/` directory if it does not exist; it is git-ignored):

```cpp
#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <thread>

#include "bounded_queue.h"

using video2x::BoundedQueue;

int main() {
    // FIFO order with move-only items
    {
        BoundedQueue<std::unique_ptr<int>> q(2);
        assert(q.push(std::make_unique<int>(1)));
        assert(q.push(std::make_unique<int>(2)));
        assert(**q.pop() == 1);
        assert(**q.pop() == 2);
    }

    // close() lets consumers drain remaining items, then pop() returns nullopt
    {
        BoundedQueue<int> q(4);
        assert(q.push(1));
        assert(q.push(2));
        q.close();
        assert(!q.push(3));
        assert(*q.pop() == 1);
        assert(*q.pop() == 2);
        assert(!q.pop().has_value());
    }

    // cancel() discards queued items
    {
        BoundedQueue<int> q(4);
        assert(q.push(1));
        q.cancel();
        assert(!q.pop().has_value());
        assert(!q.push(2));
    }

    // push() blocks while full and resumes after a pop()
    {
        BoundedQueue<int> q(1);
        assert(q.push(1));
        std::atomic<bool> pushed{false};
        std::thread t([&] {
            assert(q.push(2));
            pushed = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        assert(!pushed);
        assert(*q.pop() == 1);
        t.join();
        assert(pushed);
        assert(*q.pop() == 2);
    }

    // cancel() wakes a blocked pusher, which then reports failure
    {
        BoundedQueue<int> q(1);
        assert(q.push(1));
        std::atomic<bool> result{true};
        std::thread t([&] { result = q.push(2); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        q.cancel();
        t.join();
        assert(!result);
    }

    // cancel() wakes a blocked popper
    {
        BoundedQueue<int> q(1);
        std::atomic<bool> got_value{true};
        std::thread t([&] { got_value = q.pop().has_value(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        q.cancel();
        t.join();
        assert(!got_value);
    }

    // Producer/consumer: every item arrives exactly once, in order
    {
        BoundedQueue<int> q(3);
        long long sum = 0;
        int last = -1;
        std::thread consumer([&] {
            while (auto item = q.pop()) {
                assert(*item == last + 1);
                last = *item;
                sum += *item;
            }
        });
        for (int i = 0; i < 1000; i++) {
            assert(q.push(int(i)));
        }
        q.close();
        consumer.join();
        assert(last == 999);
        assert(sum == 999LL * 1000 / 2);
    }

    return 0;
}
```

- [ ] **Step 2: Run it to verify it fails**

Run (from the repo root, clang/g++): `clang++ -std=c++17 -pthread -I include/libvideo2x build/bounded_queue_test.cpp -o build/bounded_queue_test`
MSVC (Developer prompt): `cl /std:c++17 /EHsc /I include\libvideo2x build\bounded_queue_test.cpp /Fe:build\bounded_queue_test.exe`
Expected: compile FAIL with `'bounded_queue.h' file not found`.

- [ ] **Step 3: Write the implementation**

Create `include/libvideo2x/bounded_queue.h`:

```cpp
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace video2x {

// A bounded, blocking, closable FIFO queue for handing items between threads.
template <typename T>
class BoundedQueue {
   public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    // Blocks while the queue is full. Returns false, leaving `item` untouched,
    // if the queue was closed or cancelled.
    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return closed_ || items_.size() < capacity_; });
        if (closed_) {
            return false;
        }
        items_.push_back(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // Blocks while the queue is empty. After close(), remaining items are still
    // returned; std::nullopt means the queue is closed and drained (or cancelled).
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
        if (items_.empty()) {
            return std::nullopt;
        }
        std::optional<T> item(std::move(items_.front()));
        items_.pop_front();
        not_full_.notify_one();
        return item;
    }

    // No more pushes are accepted; consumers drain what is left.
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    // Like close(), but also discards everything queued so every waiter exits now.
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        items_.clear();
        not_empty_.notify_all();
        not_full_.notify_all();
    }

   private:
    const size_t capacity_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> items_;
    bool closed_ = false;
};

}  // namespace video2x
```

- [ ] **Step 4: Run the test to verify it passes**

Run the compile command from Step 2, then `build/bounded_queue_test` (`build\bounded_queue_test.exe` on Windows).
Expected: compiles with no warnings and exits with code 0 (`echo $?` prints `0`).

- [ ] **Step 5: Commit**

Only the header is committed (the test file lives in git-ignored `build/`).

```bash
git add include/libvideo2x/bounded_queue.h
git commit -m "feat(libvideo2x): add bounded blocking queue

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Mutex-guarded muxing in Encoder

**Files:**
- Modify: `include/libvideo2x/encoder.h`
- Modify: `src/encoder.cpp` (write_frame ~line 347, flush ~line 396, new method at end)
- Modify: `src/libvideo2x.cpp` (remove `write_raw_packet`, update call site at ~line 269)
- Modify: `include/libvideo2x/libvideo2x.h` (remove `write_raw_packet` declaration, lines 61-66)

**Interfaces:**
- Produces: `int Encoder::write_raw_packet(AVPacket* packet, AVFormatContext* ifmt_ctx)`: rescales the packet to the mapped output stream and muxes it under `mux_mutex_`. Returns `< 0` on error. After this task every `av_interleaved_write_frame` call on the output context holds `mux_mutex_`.

This task keeps serial behaviour identical; it only moves code and adds locking.

- [ ] **Step 1: Add the mutex and method to `encoder.h`**

Add `#include <mutex>` after `#include <filesystem>`, add the declaration after `int flush();`, and the member after `int* stream_map_;`:

```cpp
    int write_frame(AVFrame* frame, int64_t frame_idx);
    int flush();

    // Rescales and muxes an audio/subtitle packet; safe to call while another
    // thread is encoding video frames
    int write_raw_packet(AVPacket* packet, AVFormatContext* ifmt_ctx);
```

```cpp
    int* stream_map_;
    std::mutex mux_mutex_;
```

- [ ] **Step 2: Lock the muxer calls in `src/encoder.cpp`**

In `Encoder::write_frame`, replace

```cpp
        // Write the packet
        ret = av_interleaved_write_frame(ofmt_ctx_, enc_pkt);
        av_packet_unref(enc_pkt);
        if (ret < 0) {
            logger()->error("Error muxing packet");
```

with

```cpp
        // Write the packet
        {
            std::lock_guard<std::mutex> lock(mux_mutex_);
            ret = av_interleaved_write_frame(ofmt_ctx_, enc_pkt);
        }
        av_packet_unref(enc_pkt);
        if (ret < 0) {
            logger()->error("Error muxing packet");
```

In `Encoder::flush`, replace

```cpp
        // Write the packet
        ret = av_interleaved_write_frame(ofmt_ctx_, enc_pkt);
        av_packet_unref(enc_pkt);
        if (ret < 0) {
            logger()->error("Error muxing packet during flush");
```

with

```cpp
        // Write the packet
        {
            std::lock_guard<std::mutex> lock(mux_mutex_);
            ret = av_interleaved_write_frame(ofmt_ctx_, enc_pkt);
        }
        av_packet_unref(enc_pkt);
        if (ret < 0) {
            logger()->error("Error muxing packet during flush");
```

Add the include `#include <mutex>` is already provided by `encoder.h`. Then add this method before `Encoder::get_encoder_context()`:

```cpp
int Encoder::write_raw_packet(AVPacket* packet, AVFormatContext* ifmt_ctx) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];

    AVStream* in_stream = ifmt_ctx->streams[packet->stream_index];
    int out_stream_idx = stream_map_[packet->stream_index];
    AVStream* out_stream = ofmt_ctx_->streams[out_stream_idx];

    av_packet_rescale_ts(packet, in_stream->time_base, out_stream->time_base);
    packet->stream_index = out_stream_idx;

    int ret;
    {
        std::lock_guard<std::mutex> lock(mux_mutex_);
        ret = av_interleaved_write_frame(ofmt_ctx_, packet);
    }
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        logger()->critical("Error muxing audio/subtitle packet: {}", errbuf);
    }
    return ret;
}
```

- [ ] **Step 3: Remove `VideoProcessor::write_raw_packet` and use the Encoder's**

In `include/libvideo2x/libvideo2x.h` delete:

```cpp
    [[nodiscard]] inline int write_raw_packet(
        AVPacket* packet,
        AVFormatContext* ifmt_ctx,
        AVFormatContext* ofmt_ctx,
        int* stream_map
    );
```

In `src/libvideo2x.cpp` delete the whole `int VideoProcessor::write_raw_packet(...) { ... }` definition (from `int VideoProcessor::write_raw_packet(` through its closing brace, just before `int VideoProcessor::process_filtering(`). In `process_frames`, change

```cpp
            ret = write_raw_packet(packet.get(), ifmt_ctx, ofmt_ctx, stream_map);
```

to

```cpp
            ret = encoder.write_raw_packet(packet.get(), ifmt_ctx);
```

and delete the now unused local `AVFormatContext* ofmt_ctx = encoder.get_format_context();` (keep `stream_map`, still used in the condition).

- [ ] **Step 4: Build to verify it compiles**

Run: `just build` (Linux), or `cmake --build build --config Release --parallel`.
Expected: build succeeds with no new warnings.

- [ ] **Step 5: Verify behaviour is unchanged (requires GPU machine)**

Run the same short clip with audio before and after this commit, e.g. `video2x -i clip.mp4 -o out.mkv -p realesrgan --realesrgan-model realesr-animevideov3 -s 2`, then compare:
`ffmpeg -i out.mkv -map 0 -f framemd5 -` and `ffprobe -show_streams out.mkv`
Expected: identical video hashes and the same audio/subtitle streams as the output from the commit before this task.

- [ ] **Step 6: Commit**

```bash
git add include/libvideo2x/encoder.h src/encoder.cpp include/libvideo2x/libvideo2x.h src/libvideo2x.cpp
git commit -m "refactor(encoder): guard output muxing with a mutex

Move write_raw_packet into Encoder so audio/subtitle packets and encoded
video packets share one lock when written to the output context.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: queue_size plumbing and serial/pipelined dispatcher

**Files:**
- Modify: `include/libvideo2x/libvideo2x.h`
- Modify: `src/libvideo2x.cpp`
- Modify: `tools/video2x/include/argparse.h`
- Modify: `tools/video2x/src/argparse.cpp` (after the `benchmark` option, ~line 91)
- Modify: `tools/video2x/src/video2x.cpp` (~line 85)

**Interfaces:**
- Consumes: nothing from earlier tasks except the Task 2 `encoder.write_raw_packet` call already in `process_frames`.
- Produces:
  - `VideoProcessor(proc_cfg, enc_cfg, vk_device_idx = 0, hw_device_type = AV_HWDEVICE_TYPE_NONE, benchmark = false, queue_size = 4)`
  - private `int process_frames_serial(decoder::Decoder&, encoder::Encoder&, std::unique_ptr<processors::Processor>&)` (the existing loop)
  - private `int process_frames_pipelined(...)` with the same parameters (a stub in this task)
  - private `void init_total_frames(AVFormatContext* ifmt_ctx, int in_vstream_idx, const processors::Processor& processor)`
  - `Arguments::queue_size` (int, default 4)

- [ ] **Step 1: Header changes in `libvideo2x.h`**

Add `#include "bounded_queue.h"` after `#include "avutils.h"`. After the `VideoProcessorState` enum add:

```cpp
using FramePtr = std::unique_ptr<AVFrame, decltype(&avutils::av_frame_deleter)>;

// A frame waiting to be encoded, with the output index the encoder needs
struct EncodeItem {
    FramePtr frame;
    int64_t idx;
};
```

Change the constructor to take the new parameter:

```cpp
    VideoProcessor(
        const processors::ProcessorConfig proc_cfg,
        const encoder::EncoderConfig enc_cfg,
        const uint32_t vk_device_idx = 0,
        const AVHWDeviceType hw_device_type = AV_HWDEVICE_TYPE_NONE,
        const bool benchmark = false,
        const int queue_size = 4
    );
```

In the private section, replace the `process_frames` declaration with these four declarations:

```cpp
    [[nodiscard]] int process_frames(
        decoder::Decoder& decoder,
        encoder::Encoder& encoder,
        std::unique_ptr<processors::Processor>& processor
    );

    [[nodiscard]] int process_frames_serial(
        decoder::Decoder& decoder,
        encoder::Encoder& encoder,
        std::unique_ptr<processors::Processor>& processor
    );

    [[nodiscard]] int process_frames_pipelined(
        decoder::Decoder& decoder,
        encoder::Encoder& encoder,
        std::unique_ptr<processors::Processor>& processor
    );

    void init_total_frames(
        AVFormatContext* ifmt_ctx,
        int in_vstream_idx,
        const processors::Processor& processor
    );
```

and add members after `bool benchmark_ = false;`:

```cpp
    int queue_size_ = 4;
    BoundedQueue<EncodeItem>* out_queue_ = nullptr;
```

- [ ] **Step 2: Constructor, dispatcher and helper in `libvideo2x.cpp`**

Update the constructor:

```cpp
VideoProcessor::VideoProcessor(
    const processors::ProcessorConfig proc_cfg,
    const encoder::EncoderConfig enc_cfg,
    const uint32_t vk_device_idx,
    const AVHWDeviceType hw_device_type,
    const bool benchmark,
    const int queue_size
)
    : proc_cfg_(proc_cfg),
      enc_cfg_(enc_cfg),
      vk_device_idx_(vk_device_idx),
      hw_device_type_(hw_device_type),
      benchmark_(benchmark),
      queue_size_(queue_size) {}
```

Rename the existing `int VideoProcessor::process_frames(` definition to `int VideoProcessor::process_frames_serial(`. In it, replace the block

```cpp
    // Set the total number of frames in the VideoProcessingContext
    logger()->debug("Estimating the total number of frames to process");
    total_frames_ = avutils::get_video_frame_count(ifmt_ctx, in_vstream_idx);

    if (total_frames_ <= 0) {
        logger()->warn("Unable to determine the total number of frames");
        total_frames_ = 0;
    } else {
        logger()->debug("{} frames to process", total_frames_.load());
    }

    // Set total frames for interpolation
    if (processor->get_processing_mode() == processors::ProcessingMode::Interpolate) {
        total_frames_.store(total_frames_.load() * proc_cfg_.frm_rate_mul);
    }
```

with

```cpp
    init_total_frames(ifmt_ctx, in_vstream_idx, *processor);
```

Insert before `process_frames_serial` the helper, dispatcher and a stub:

```cpp
void VideoProcessor::init_total_frames(
    AVFormatContext* ifmt_ctx,
    int in_vstream_idx,
    const processors::Processor& processor
) {
    // Set the total number of frames in the VideoProcessingContext
    logger()->debug("Estimating the total number of frames to process");
    total_frames_ = avutils::get_video_frame_count(ifmt_ctx, in_vstream_idx);

    if (total_frames_ <= 0) {
        logger()->warn("Unable to determine the total number of frames");
        total_frames_ = 0;
    } else {
        logger()->debug("{} frames to process", total_frames_.load());
    }

    // Set total frames for interpolation
    if (processor.get_processing_mode() == processors::ProcessingMode::Interpolate) {
        total_frames_.store(total_frames_.load() * proc_cfg_.frm_rate_mul);
    }
}

// Process frames, pipelined across threads unless queue_size_ is 0.
int VideoProcessor::process_frames(
    decoder::Decoder& decoder,
    encoder::Encoder& encoder,
    std::unique_ptr<processors::Processor>& processor
) {
    if (queue_size_ > 0) {
        return process_frames_pipelined(decoder, encoder, processor);
    }
    return process_frames_serial(decoder, encoder, processor);
}

int VideoProcessor::process_frames_pipelined(
    decoder::Decoder& decoder,
    encoder::Encoder& encoder,
    std::unique_ptr<processors::Processor>& processor
) {
    // Implemented in the next task; run serially until then
    return process_frames_serial(decoder, encoder, processor);
}
```

- [ ] **Step 3: CLI option**

`tools/video2x/include/argparse.h`: add `int queue_size = 4;` after `bool benchmark = false;`.

`tools/video2x/src/argparse.cpp`: after the `benchmark,b` option (before the `;`) add:

```cpp
            ("queue-size", po::value<int>(&arguments.queue_size)->default_value(4)
                ->notifier([](int v) { validate_min(v, "queue-size", 0); }),
                "Frames buffered between decode, processing and encode stages; "
                "0 disables pipelining")
```

`tools/video2x/src/video2x.cpp`: change the constructor call to

```cpp
    video2x::VideoProcessor video_processor = video2x::VideoProcessor(
        proc_cfg,
        enc_cfg,
        arguments.vk_device_index,
        arguments.hw_device_type,
        arguments.benchmark,
        arguments.queue_size
    );
```

- [ ] **Step 4: Build and check the option**

Run: `just build`, then `build/video2x-install/bin/video2x --help | grep queue-size` (adjust path for your install layout) and `video2x -i a -o b --queue-size -1`.
Expected: build succeeds; help shows `--queue-size`; `-1` is rejected with "queue-size must be at least 0".

- [ ] **Step 5: Commit**

```bash
git add include/libvideo2x/libvideo2x.h src/libvideo2x.cpp tools/video2x/include/argparse.h tools/video2x/src/argparse.cpp tools/video2x/src/video2x.cpp
git commit -m "feat(libvideo2x): add queue-size option and frame loop dispatcher

Split process_frames into serial and pipelined variants chosen by
queue_size. The pipelined variant falls back to the serial one for now.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Pipelined frame loop

**Files:**
- Modify: `src/libvideo2x.cpp` (`write_frame` and `process_frames_pipelined`, plus includes and an anonymous-namespace helper)

**Interfaces:**
- Consumes: `BoundedQueue<T>` (Task 1), `Encoder::write_raw_packet` (Task 2), `FramePtr`, `EncodeItem`, `out_queue_`, `init_total_frames`, `process_frames_pipelined` stub (Task 3).
- Produces: a working `process_frames_pipelined`. Behaviour: when `out_queue_ != nullptr`, `write_frame` pushes `EncodeItem{av_frame_clone(frame), frame_idx_}`; the encode thread calls `encoder.write_frame(item.frame.get(), item.idx)`.

- [ ] **Step 1: Includes and shared state**

At the top of `src/libvideo2x.cpp`, add after `#include "libvideo2x.h"`:

```cpp
#include <chrono>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
```

(Keep the existing includes; `<thread>`/`<chrono>` may already be included transitively, listing them is harmless.) Inside `namespace video2x {`, before the constructor, add:

```cpp
namespace {

// State shared by the decode, GPU and encode stages of the pipelined loop
struct PipelineState {
    explicit PipelineState(size_t capacity) : decoded(capacity), encoded(capacity) {}

    BoundedQueue<FramePtr> decoded;
    BoundedQueue<EncodeItem> encoded;
    std::atomic<bool> failed{false};

    // Records the root-cause error (later ones are consequences of the shutdown
    // and are ignored) and wakes every stage so it can unwind
    void fail(int error_code) {
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (first_error_ == 0) {
                first_error_ = error_code;
            }
        }
        failed.store(true);
        decoded.cancel();
        encoded.cancel();
    }

    int get_error() {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return first_error_;
    }

   private:
    std::mutex error_mutex_;
    int first_error_ = 0;
};

}  // namespace
```

- [ ] **Step 2: Route `write_frame` through the queue when pipelined**

Replace the whole `VideoProcessor::write_frame` definition with:

```cpp
int VideoProcessor::write_frame(AVFrame* frame, encoder::Encoder& encoder) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    int ret = 0;

    if (!benchmark_) {
        if (out_queue_ != nullptr) {
            // Pipelined: hand a reference to the encode thread
            FramePtr frame_ref(av_frame_clone(frame), &avutils::av_frame_deleter);
            if (frame_ref == nullptr) {
                return AVERROR(ENOMEM);
            }
            if (!out_queue_->push(EncodeItem{std::move(frame_ref), frame_idx_.load()})) {
                return AVERROR_EXIT;
            }
            return 0;
        }

        ret = encoder.write_frame(frame, frame_idx_.load());
        if (ret < 0) {
            av_strerror(ret, errbuf, sizeof(errbuf));
            logger()->critical("Error encoding/writing frame: {}", errbuf);
        }
    }
    return ret;
}
```

- [ ] **Step 3: Implement `process_frames_pipelined`**

Replace the stub with:

```cpp
int VideoProcessor::process_frames_pipelined(
    decoder::Decoder& decoder,
    encoder::Encoder& encoder,
    std::unique_ptr<processors::Processor>& processor
) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    int ret = 0;

    AVFormatContext* ifmt_ctx = decoder.get_format_context();
    AVCodecContext* dec_ctx = decoder.get_codec_context();
    int in_vstream_idx = decoder.get_video_stream_index();
    AVCodecContext* enc_ctx = encoder.get_encoder_context();
    int* stream_map = encoder.get_stream_map();

    init_total_frames(ifmt_ctx, in_vstream_idx, *processor);

    PipelineState ps(static_cast<size_t>(queue_size_));

    // Decode stage: reads packets, decodes video into the first queue and muxes
    // audio/subtitle packets directly
    auto decode_stage = [&]() {
        char err[AV_ERROR_MAX_STRING_SIZE];
        std::unique_ptr<AVFrame, decltype(&avutils::av_frame_deleter)> frame(
            av_frame_alloc(), &avutils::av_frame_deleter
        );
        std::unique_ptr<AVPacket, decltype(&avutils::av_packet_deleter)> packet(
            av_packet_alloc(), &avutils::av_packet_deleter
        );
        if (frame == nullptr || packet == nullptr) {
            logger()->critical("Error allocating frame or packet");
            ps.fail(AVERROR(ENOMEM));
            ps.decoded.close();
            return;
        }

        while (state_.load() != VideoProcessorState::Aborted && !ps.failed.load()) {
            int dec_ret = av_read_frame(ifmt_ctx, packet.get());
            if (dec_ret < 0) {
                if (dec_ret == AVERROR_EOF) {
                    logger()->debug("Reached end of file");
                    break;
                }
                av_strerror(dec_ret, err, sizeof(err));
                logger()->critical("Error reading packet: {}", err);
                ps.fail(dec_ret);
                break;
            }

            if (packet->stream_index == in_vstream_idx) {
                dec_ret = avcodec_send_packet(dec_ctx, packet.get());
                if (dec_ret < 0) {
                    av_strerror(dec_ret, err, sizeof(err));
                    logger()->critical("Error sending packet to decoder: {}", err);
                    ps.fail(dec_ret);
                    break;
                }

                bool queue_open = true;
                while (queue_open) {
                    dec_ret = avcodec_receive_frame(dec_ctx, frame.get());
                    if (dec_ret == AVERROR(EAGAIN)) {
                        break;
                    } else if (dec_ret < 0) {
                        av_strerror(dec_ret, err, sizeof(err));
                        logger()->critical("Error decoding video frame: {}", err);
                        ps.fail(dec_ret);
                        queue_open = false;
                        break;
                    }

                    // Move the decoded frame into its own AVFrame for the queue
                    FramePtr decoded(av_frame_alloc(), &avutils::av_frame_deleter);
                    if (decoded == nullptr) {
                        ps.fail(AVERROR(ENOMEM));
                        queue_open = false;
                        break;
                    }
                    av_frame_move_ref(decoded.get(), frame.get());
                    queue_open = ps.decoded.push(std::move(decoded));
                }
                if (!queue_open) {
                    break;
                }
            } else if ((enc_cfg_.copy_audio_streams || enc_cfg_.copy_subtitle_streams) &&
                       stream_map[packet->stream_index] >= 0) {
                dec_ret = encoder.write_raw_packet(packet.get(), ifmt_ctx);
                if (dec_ret < 0) {
                    ps.fail(dec_ret);
                    break;
                }
            }
            av_packet_unref(packet.get());
        }

        // No more frames will be produced
        ps.decoded.close();
    };

    // Encode stage: encodes and muxes frames from the second queue
    auto encode_stage = [&]() {
        while (true) {
            std::optional<EncodeItem> item = ps.encoded.pop();
            if (!item.has_value()) {
                break;
            }
            int enc_ret = encoder.write_frame(item->frame.get(), item->idx);
            if (enc_ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(enc_ret, err, sizeof(err));
                logger()->critical("Error encoding/writing frame: {}", err);
                ps.fail(enc_ret);
                return;
            }
        }
        if (ps.failed.load()) {
            return;
        }

        int enc_ret = encoder.flush();
        if (enc_ret < 0) {
            char err[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(enc_ret, err, sizeof(err));
            logger()->critical("Error flushing encoder: {}", err);
            ps.fail(enc_ret);
        }
    };

    // Always unblock and join the worker threads, whatever path we leave by
    struct StageGuard {
        PipelineState& ps;
        std::thread& decode_thread;
        std::thread& encode_thread;
        BoundedQueue<EncodeItem>*& out_queue;

        ~StageGuard() {
            ps.decoded.cancel();
            ps.encoded.cancel();
            if (decode_thread.joinable()) {
                decode_thread.join();
            }
            if (encode_thread.joinable()) {
                encode_thread.join();
            }
            out_queue = nullptr;
        }
    };

    std::thread decode_thread;
    std::thread encode_thread;
    StageGuard guard{ps, decode_thread, encode_thread, out_queue_};
    out_queue_ = &ps.encoded;
    decode_thread = std::thread(decode_stage);
    encode_thread = std::thread(encode_stage);

    // GPU stage: runs on the calling thread
    std::unique_ptr<AVFrame, decltype(&avutils::av_frame_deleter)> prev_frame(
        nullptr, &avutils::av_frame_deleter
    );

    while (true) {
        // Sleep for 100 ms if processing is paused
        while (state_.load() == VideoProcessorState::Paused && !ps.failed.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (state_.load() == VideoProcessorState::Aborted || ps.failed.load()) {
            break;
        }

        std::optional<FramePtr> item = ps.decoded.pop();
        if (!item.has_value()) {
            break;
        }
        AVFrame* frame = item->get();

        // Calculate this frame's presentation timestamp (PTS)
        if (enc_cfg_.recalculate_pts) {
            frame->pts =
                av_rescale_q(frame_idx_, av_inv_q(enc_ctx->framerate), enc_ctx->time_base);
        }

        // Process the frame based on the selected processing mode
        AVFrame* proc_frame = nullptr;
        switch (processor->get_processing_mode()) {
            case processors::ProcessingMode::Filter: {
                ret = process_filtering(processor, encoder, frame, proc_frame);
                break;
            }
            case processors::ProcessingMode::Interpolate: {
                ret = process_interpolation(processor, encoder, prev_frame, frame, proc_frame);
                break;
            }
            default:
                logger()->critical("Unknown processing mode");
                ps.fail(-1);
                return ps.get_error();
        }
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            ps.fail(ret);
            break;
        }
        frame_idx_.fetch_add(1);
        logger()->debug("Processed frame {}/{}", frame_idx_.load(), total_frames_.load());
    }

    // The decode stage is no longer needed; unblock it if it is waiting on a full queue
    ps.decoded.cancel();
    if (decode_thread.joinable()) {
        decode_thread.join();
    }
    if (ps.failed.load()) {
        return ps.get_error();
    }

    // Flush the processor
    std::vector<AVFrame*> raw_flushed_frames;
    ret = processor->flush(raw_flushed_frames);
    if (ret < 0) {
        av_strerror(ret, errbuf, sizeof(errbuf));
        logger()->critical("Error flushing processor: {}", errbuf);
        ps.fail(ret);
        return ps.get_error();
    }

    // Wrap flushed frames in unique_ptrs
    std::vector<FramePtr> flushed_frames;
    for (AVFrame* raw_frame : raw_flushed_frames) {
        flushed_frames.emplace_back(raw_frame, &avutils::av_frame_deleter);
    }

    // Queue all flushed frames for encoding
    for (auto& flushed_frame : flushed_frames) {
        ret = write_frame(flushed_frame.get(), encoder);
        if (ret < 0) {
            ps.fail(ret);
            return ps.get_error();
        }
        frame_idx_.fetch_add(1);
    }

    // Let the encode stage drain its queue, flush the encoder and finish
    ps.encoded.close();
    encode_thread.join();
    return ps.get_error();
}
```

- [ ] **Step 4: Format and build**

Run: `clang-format -i src/libvideo2x.cpp include/libvideo2x/libvideo2x.h include/libvideo2x/bounded_queue.h include/libvideo2x/encoder.h src/encoder.cpp`, then `just build`.
Expected: build succeeds with no warnings; `git diff --stat` shows only the intended files.

- [ ] **Step 5: Verify output equivalence (requires GPU machine)**

Use a 20-30 second 1080p clip with audio. Run each pair and compare:

```bash
video2x -i clip.mp4 -o serial.mkv    -p realesrgan --realesrgan-model realesr-animevideov3 -s 2 --queue-size 0
video2x -i clip.mp4 -o pipelined.mkv -p realesrgan --realesrgan-model realesr-animevideov3 -s 2
ffmpeg -i serial.mkv    -map 0:v -f framemd5 serial.md5
ffmpeg -i pipelined.mkv -map 0:v -f framemd5 pipelined.md5
diff serial.md5 pipelined.md5 && echo IDENTICAL
```

Repeat with `-p rife --rife-model rife-v4.26 --frame-rate-mul 2` (adjust flag names to `video2x --help`) and with `--no-copy-audio-streams`.
Expected: `IDENTICAL` for each pair, and `ffprobe` shows the same stream list. If the encoder is non-deterministic across runs (threads), compare with `-e threads=1` on both runs. If hashes differ only by the encoder's own multithreading, compare frame counts and pts lists (`ffprobe -show_frames -show_entries frame=pts_time`) and decoded pixel hashes of the video stream instead.

- [ ] **Step 6: Verify pause, abort, error and speed (requires GPU machine)**

1. Start a run, press space to pause for 10 s, press space to resume. Expected: processing resumes, the output is complete.
2. Start a run, press `q`. Expected: the process exits within a couple of seconds, the output file plays and ends early.
3. `video2x -i /nonexistent.mp4 -o out.mkv -p realesrgan ...` and a corrupt file. Expected: an error message and non-zero exit, no hang.
4. If ThreadSanitizer is available (`-fsanitize=thread` Debug build), run steps 1-2 once under it. Expected: no data race reports.
5. Speed: run the same clip with `--queue-size 0` and the default, note `fps=`. Expected: the default is faster when the encoder is a bottleneck (e.g. libx264 with `-e preset=slow`). Report both numbers; a pipelined run that is not faster means the GPU is the limit.

- [ ] **Step 7: Commit**

```bash
git add src/libvideo2x.cpp
git commit -m "feat(libvideo2x): pipeline decoding, processing and encoding

Run decoding and encoding on their own threads, connected to the GPU
stage by bounded queues, so a slow CPU encoder no longer stalls frame
processing. --queue-size 0 keeps the previous serial behaviour.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CHANGELOG.md` (under `## [Unreleased]` → `### Added`)
- Modify: `CLAUDE.md` (Architecture → Pipeline paragraph)

- [ ] **Step 1: Changelog**

In `CHANGELOG.md`, add as the first bullet under `## [Unreleased]` → `### Added`:

```markdown
- Pipelined decoding, processing, and encoding on separate threads, with the `--queue-size` option (`0` disables it).
```

- [ ] **Step 2: CLAUDE.md**

In `CLAUDE.md`, replace the sentence in the **Pipeline** paragraph beginning `` `process()` opens a `decoder::Decoder` `` through `` ...then `process_frames()` loops decode → process → encode.`` with:

```markdown
`process()` opens a `decoder::Decoder`, builds an `encoder::Encoder` from `EncoderConfig`, creates a `Processor` via `processor_factory`, then `process_frames()` runs the decode → process → encode loop. With `queue_size > 0` (default 4, `--queue-size`) that loop is pipelined: a decode thread and an encode thread run beside the calling thread (the GPU stage), connected by `BoundedQueue`s; `queue_size == 0` runs the original serial loop (`process_frames_serial`). All output muxing goes through `Encoder` (`mux_mutex_`).
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md CLAUDE.md
git commit -m "docs(libvideo2x): document the pipelined frame loop

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** three stages and queues (Tasks 1, 4); PTS recalculation stays in the GPU stage (Task 4, GPU loop); `write_raw_packet` and the mux mutex in `Encoder` (Task 2); `queue_size` parameter, `--queue-size`, `0` = serial (Task 3); pause in the GPU stage, abort, first-error slot, RAII joins, progress via `frame_idx_`, benchmark skip, flush ordering (Task 4); testing steps (Tasks 1 and 4); docs and changelog (Task 5). The spec's hardware-decode frame-pool risk is not exercised by any test here; if you use `--hwaccel`, test it explicitly in Task 4 Step 5.
- **Placeholders:** none; the Task 3 pipelined stub is replaced in Task 4.
- **Type consistency:** `FramePtr`, `EncodeItem`, `BoundedQueue<T>::push/pop/close/cancel`, `Encoder::write_raw_packet(AVPacket*, AVFormatContext*)`, `init_total_frames(AVFormatContext*, int, const processors::Processor&)` and `process_frames_{serial,pipelined}` are spelled the same in every task.
- **Known limits:** nothing in this plan could be compiled on the machine it was written on (no toolchain); the code was written against the current sources but has not been built.
