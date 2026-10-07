#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

namespace ccr {

/// Seeded standard normal generator (Mersenne Twister 64).
///
/// A fixed seed gives reproducible scenarios, which is what makes
/// bump-and-revalue sensitivities with common random numbers stable.
class NormalGenerator {
 public:
  explicit NormalGenerator(std::uint64_t seed) : engine_(seed) {}

  double next() { return distribution_(engine_); }

  void fill(double* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = distribution_(engine_);
  }

 private:
  std::mt19937_64 engine_;
  std::normal_distribution<double> distribution_{0.0, 1.0};
};

}  // namespace ccr
