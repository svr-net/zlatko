#include "ccr/core/parallel.hpp"

#include <cstdlib>
#include <string>

namespace ccr {

namespace {

std::size_t defaultWorkers() {
#ifdef CCR_NO_THREADS
  return 0;
#else
  std::size_t threads = std::thread::hardware_concurrency();
  if (const char* env = std::getenv("CCR_THREADS")) {
    try {
      threads = std::stoul(env);
    } catch (...) {
    }
  }
  return threads > 1 ? threads - 1 : 0;
#endif
}

}  // namespace

SpinPool& defaultPool() {
  static std::atomic<SpinPool*> pool{nullptr};
  SpinPool* p = pool.load(std::memory_order_acquire);
  if (p) return *p;
  auto* fresh = new SpinPool(defaultWorkers());
  if (pool.compare_exchange_strong(p, fresh, std::memory_order_acq_rel)) return *fresh;
  delete fresh;  // another thread installed its pool first
  return *p;
}

}  // namespace ccr
