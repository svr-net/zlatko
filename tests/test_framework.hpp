#pragma once

// Minimal self-registering test framework (no external dependencies).

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ccrtest {

struct TestCase {
  std::string name;
  std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void fail(const std::string& message, const char* file, int line) {
  std::ostringstream os;
  os << file << ":" << line << ": " << message;
  throw Failure(os.str());
}

}  // namespace ccrtest

#define CCR_CONCAT_INNER(a, b) a##b
#define CCR_CONCAT(a, b) CCR_CONCAT_INNER(a, b)

#define TEST(name)                                                                             \
  static void CCR_CONCAT(test_, name)();                                                       \
  static ::ccrtest::Registrar CCR_CONCAT(registrar_, name)(#name, &CCR_CONCAT(test_, name));   \
  static void CCR_CONCAT(test_, name)()

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) ::ccrtest::fail("CHECK(" #condition ") failed", __FILE__, __LINE__); \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                   \
  do {                                                                                            \
    const double ccr_a = (actual);                                                                \
    const double ccr_e = (expected);                                                              \
    if (!(std::fabs(ccr_a - ccr_e) <= (tolerance))) {                                             \
      std::ostringstream ccr_os;                                                                  \
      ccr_os.precision(12);                                                                       \
      ccr_os << "CHECK_NEAR(" #actual ", " #expected ") failed: " << ccr_a << " vs " << ccr_e      \
             << " (tolerance " << (tolerance) << ")";                                             \
      ::ccrtest::fail(ccr_os.str(), __FILE__, __LINE__);                                          \
    }                                                                                             \
  } while (0)

#define CHECK_THROWS(statement)                                                       \
  do {                                                                                \
    bool ccr_threw = false;                                                           \
    try {                                                                             \
      statement;                                                                      \
    } catch (...) {                                                                   \
      ccr_threw = true;                                                               \
    }                                                                                 \
    if (!ccr_threw) ::ccrtest::fail("expected exception: " #statement, __FILE__, __LINE__); \
  } while (0)
