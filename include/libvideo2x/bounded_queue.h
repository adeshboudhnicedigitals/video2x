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
