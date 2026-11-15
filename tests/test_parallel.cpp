#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ccr/core/parallel.hpp"
#include "test_framework.hpp"

using namespace ccr;

TEST(spin_pool_runs_every_index_exactly_once) {
  for (const std::size_t workers : {std::size_t(0), std::size_t(1), std::size_t(7)}) {
    SpinPool pool(workers);
    CHECK(pool.concurrency() == workers + 1);
    for (const std::size_t n : {std::size_t(1), std::size_t(5), std::size_t(1000), std::size_t(100003)})
      for (const std::size_t chunk : {std::size_t(0), std::size_t(1), std::size_t(64)}) {
        std::vector<std::atomic<int>> hits(n);
        pool.parallelFor(n, chunk, [&](std::size_t b, std::size_t e) {
          CHECK(b < e && e <= n);
          for (std::size_t i = b; i < e; ++i) hits[i].fetch_add(1, std::memory_order_relaxed);
        });
        bool once = true;
        for (const auto& h : hits) once = once && h.load() == 1;
        CHECK(once);
      }
  }
}

TEST(spin_pool_uses_its_workers_and_reuses_them) {
  // Many short loops back to back: workers must pick up every epoch (no lost wake-up, no
  // deadlock), and with blocking bodies more than one thread must take part.
  SpinPool pool(3);
  std::atomic<long> total{0};
  for (int round = 0; round < 2000; ++round)
    pool.parallelFor(64, 1, [&](std::size_t b, std::size_t e) { total.fetch_add(static_cast<long>(e - b)); });
  CHECK(total.load() == 2000 * 64);
  std::vector<std::atomic<bool>> seen(16);
  std::atomic<int> distinct{0};
  std::atomic<std::size_t> started{0};
  pool.parallelFor(4, 1, [&](std::size_t, std::size_t) {
    started.fetch_add(1);
    // Hold each chunk until all four have started: only possible if four threads run at once.
    for (int spin = 0; started.load() < 4 && spin < 20000000; ++spin) std::this_thread::yield();
    const auto id = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 16;
    bool expected = false;
    if (seen[id].compare_exchange_strong(expected, true)) distinct.fetch_add(1);
  });
  CHECK(started.load() == 4);
  CHECK(distinct.load() >= 2);
}

TEST(spin_pool_rethrows_and_stays_usable) {
  SpinPool pool(3);
  CHECK_THROWS(pool.parallelFor(1000, 1, [](std::size_t b, std::size_t) {
    if (b == 517) throw std::runtime_error("body failed");
  }));
  std::atomic<std::size_t> count{0};
  pool.parallelFor(1000, 1, [&](std::size_t b, std::size_t e) { count.fetch_add(e - b); });
  CHECK(count.load() == 1000);
}

TEST(spin_pool_nested_and_concurrent_loops_run_without_blocking) {
  // A loop started inside a body, or by another thread while the pool is busy, loses the
  // compare-exchange on the busy flag and runs serially on its caller: never a deadlock.
  SpinPool pool(3);
  std::atomic<std::size_t> inner{0};
  pool.parallelFor(8, 1, [&](std::size_t, std::size_t) {
    pool.parallelFor(10, 1, [&](std::size_t b, std::size_t e) { inner.fetch_add(e - b); });
  });
  CHECK(inner.load() == 80);
  std::atomic<std::size_t> sum{0};
  std::vector<std::thread> callers;
  for (int t = 0; t < 4; ++t)
    callers.emplace_back([&] {
      for (int r = 0; r < 200; ++r) pool.parallelFor(100, 7, [&](std::size_t b, std::size_t e) { sum.fetch_add(e - b); });
    });
  for (auto& c : callers) c.join();
  CHECK(sum.load() == 4u * 200u * 100u);
  CHECK(defaultPool().concurrency() >= 1 && &defaultPool() == &defaultPool());
}
