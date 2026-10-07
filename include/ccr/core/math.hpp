#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ccr {

enum class OptionType { Call, Put };

inline double normalCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline double normalPdf(double x) {
  constexpr double kInvSqrt2Pi = 0.39894228040143267794;
  return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

/// Undiscounted-forward Black formula times a discount factor:
/// df * E[(F_T - K)^+] (call) with log-normal F_T and total standard deviation `stdDev`.
inline double blackFormula(OptionType type, double forward, double strike, double stdDev, double discount) {
  const double sign = type == OptionType::Call ? 1.0 : -1.0;
  if (stdDev <= 1e-14 || forward <= 0.0 || strike <= 0.0)
    return discount * std::max(sign * (forward - strike), 0.0);
  const double d1 = std::log(forward / strike) / stdDev + 0.5 * stdDev;
  const double d2 = d1 - stdDev;
  return discount * sign * (forward * normalCdf(sign * d1) - strike * normalCdf(sign * d2));
}

inline double mean(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

/// Empirical quantile with linear interpolation between order statistics.
inline double quantile(std::vector<double> values, double p) {
  if (values.empty()) throw std::invalid_argument("quantile: empty sample");
  if (p < 0.0 || p > 1.0) throw std::invalid_argument("quantile: level must be in [0, 1]");
  std::sort(values.begin(), values.end());
  const double h = p * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(h));
  const std::size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (h - static_cast<double>(lo)) * (values[hi] - values[lo]);
}

/// Brent's root finder on a bracketing interval [a, b].
template <class F>
double solveBrent(F&& f, double a, double b, double tol = 1e-12, int maxIterations = 200) {
  double fa = f(a);
  double fb = f(b);
  if (fa == 0.0) return a;
  if (fb == 0.0) return b;
  if ((fa > 0.0) == (fb > 0.0)) throw std::invalid_argument("solveBrent: root is not bracketed");
  double c = b, fc = fb, d = b - a, e = d;
  for (int iter = 0; iter < maxIterations; ++iter) {
    if ((fb > 0.0) == (fc > 0.0)) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
    if (std::fabs(fc) < std::fabs(fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    const double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::fabs(b) + 0.5 * tol;
    const double xm = 0.5 * (c - b);
    if (std::fabs(xm) <= tol1 || fb == 0.0) return b;
    if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
      double p, q;
      const double s = fb / fa;
      if (a == c) {
        p = 2.0 * xm * s;
        q = 1.0 - s;
      } else {
        const double qq = fa / fc;
        const double r = fb / fc;
        p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
        q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
      }
      if (p > 0.0) q = -q;
      p = std::fabs(p);
      const double min1 = 3.0 * xm * q - std::fabs(tol1 * q);
      const double min2 = std::fabs(e * q);
      if (2.0 * p < std::min(min1, min2)) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    a = b;
    fa = fb;
    b += std::fabs(d) > tol1 ? d : (xm >= 0.0 ? tol1 : -tol1);
    fb = f(b);
  }
  throw std::runtime_error("solveBrent: maximum number of iterations exceeded");
}

}  // namespace ccr
