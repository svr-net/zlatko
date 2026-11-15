#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
#define CCR_NO_THREADS 1  // WebAssembly without pthreads: the pool runs everything on the caller
#else
#include <chrono>
#include <thread>
#endif

namespace ccr {

/// A lock-free fork-join pool for data-parallel loops.
///
/// No mutex, no condition variable: workers wait for work by spinning on an atomic epoch
/// (pause, then yield, then short sleeps when idle for long), claim chunks of the index range
/// with compare-exchange on a shared cursor, and report completion by decrementing an atomic
/// counter the caller spins on. The caller works too, so a pool of n workers runs n + 1 ways.
///
/// One loop runs at a time: a second caller (another thread, or a nested loop inside a body)
/// that fails the compare-exchange on the busy flag simply runs its loop serially, so the pool
/// never blocks and never deadlocks. Bodies write disjoint outputs; the pool adds no ordering
/// of its own, so a loop whose iterations are independent gives the same bits on any number of
/// threads. The first exception thrown by a body is rethrown to the caller.
class SpinPool {
 public:
  /// A pool with `workers` background threads (0: everything runs on the caller).
  explicit SpinPool(std::size_t workers) {
#ifndef CCR_NO_THREADS
    threads_.reserve(workers);
    for (std::size_t i = 0; i < workers; ++i) threads_.emplace_back([this] { workerLoop(); });
#else
    (void)workers;
#endif
  }

  ~SpinPool() {
    stop_.store(true, std::memory_order_release);
#ifndef CCR_NO_THREADS
    for (auto& t : threads_) t.join();
#endif
  }

  SpinPool(const SpinPool&) = delete;
  SpinPool& operator=(const SpinPool&) = delete;

  /// Threads a loop runs on: the workers and the caller.
  std::size_t concurrency() const { return threads_.size() + 1; }

  /// Calls body(begin, end) for consecutive chunks covering [0, n), at most `chunk` indices each
  /// (0: about four chunks per thread), and returns when every chunk has run.
  template <class F>
  void parallelFor(std::size_t n, std::size_t chunk, F&& body) {
    if (n == 0) return;
    if (chunk == 0) chunk = std::max<std::size_t>(1, n / (4 * concurrency()));
    bool idle = false;
    if (threads_.empty() || n <= chunk || !busy_.compare_exchange_strong(idle, true, std::memory_order_acquire)) {
      for (std::size_t b = 0; b < n; b += chunk) body(b, std::min(n, b + chunk));
      return;
    }
    using Body = std::remove_reference_t<F>;
    task_.ctx = &body;
    task_.run = [](void* ctx, std::size_t b, std::size_t e) { (*static_cast<Body*>(ctx))(b, e); };
    task_.n = n;
    task_.chunk = chunk;
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    cursor_.store(0, std::memory_order_relaxed);
    pending_.store(threads_.size(), std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_release);  // publishes the task to the workers

    work();
    Backoff wait;
    while (pending_.load(std::memory_order_acquire) != 0) wait.pause();  // join: every worker is done
    const bool failed = failed_.load(std::memory_order_acquire);
    std::exception_ptr error = error_;
    busy_.store(false, std::memory_order_release);
    if (failed) std::rethrow_exception(error);
  }

 private:
  struct Task {
    void* ctx = nullptr;
    void (*run)(void*, std::size_t, std::size_t) = nullptr;
    std::size_t n = 0, chunk = 1;
  };

  // Spin with a CPU pause, then yield the core, then (only when idle for long) sleep briefly.
  struct Backoff {
    std::uint32_t spins = 0;
    void pause() {
      ++spins;
#ifndef CCR_NO_THREADS
      if (spins < 64) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
      } else if (spins < 4096) {
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(spins < 65536 ? 50 : 500));
      }
#endif
    }
  };

  // Claims the next chunk with compare-exchange on the shared cursor.
  bool claim(std::size_t& begin, std::size_t& end) {
    std::size_t cur = cursor_.load(std::memory_order_relaxed);
    do {
      if (cur >= task_.n) return false;
      end = std::min(task_.n, cur + task_.chunk);
    } while (!cursor_.compare_exchange_weak(cur, end, std::memory_order_acq_rel, std::memory_order_relaxed));
    begin = cur;
    return true;
  }

  void work() {
    std::size_t b = 0, e = 0;
    while (claim(b, e)) {
      if (failed_.load(std::memory_order_relaxed)) continue;  // drain the range after a failure
      try {
        task_.run(task_.ctx, b, e);
      } catch (...) {
        bool expected = false;
        if (failed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) error_ = std::current_exception();
      }
    }
  }

  void workerLoop() {
    std::uint64_t seen = 0;
    for (;;) {
      Backoff wait;
      std::uint64_t epoch;
      while ((epoch = epoch_.load(std::memory_order_acquire)) == seen) {
        if (stop_.load(std::memory_order_acquire)) return;
        wait.pause();
      }
      seen = epoch;
      work();
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

#ifndef CCR_NO_THREADS
  std::vector<std::thread> threads_;
#else
  std::vector<int> threads_;  // always empty
#endif
  Task task_;
  std::exception_ptr error_;
  alignas(64) std::atomic<std::uint64_t> epoch_{0};
  alignas(64) std::atomic<std::size_t> cursor_{0};
  alignas(64) std::atomic<std::size_t> pending_{0};
  alignas(64) std::atomic<bool> busy_{false};
  std::atomic<bool> failed_{false};
  std::atomic<bool> stop_{false};
};

/// The process-wide pool: hardware threads - 1 workers (the caller is the last one), or
/// CCR_THREADS - 1 when that environment variable is set; no workers in WebAssembly without
/// pthreads. Created on first use, installed with compare-exchange (no lock), never destroyed.
SpinPool& defaultPool();

}  // namespace ccr
