#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ccr::app {

/// A JSON-like value with typed numeric arrays: what the analyses read (specifications) and
/// return (results). It lets every analysis live in the C++ library, testable natively, while
/// the WebAssembly binding only converts Value <-> JavaScript. Objects and arrays have
/// reference semantics, like JavaScript values: copies share their contents.
class Value {
 public:
  enum class Type { Undefined, Null, Bool, Number, String, Array, Object, F64, F32, U32 };

  Value() = default;
  Value(bool b) : type_(Type::Bool), number_(b ? 1.0 : 0.0) {}  // NOLINT(google-explicit-constructor)
  template <class T, std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, int> = 0>
  Value(T n) : type_(Type::Number), number_(static_cast<double>(n)) {}  // NOLINT(google-explicit-constructor)
  Value(std::string s) : type_(Type::String), string_(std::make_shared<std::string>(std::move(s))) {}  // NOLINT
  Value(const char* s) : Value(std::string(s)) {}  // NOLINT(google-explicit-constructor)

  static Value object() { return Value(Type::Object); }
  static Value array() { return Value(Type::Array); }
  static Value null() { return Value(Type::Null); }
  static Value undefined() { return Value(); }
  static Value f64(std::vector<double> v) { Value r(Type::F64); *r.f64_ = std::move(v); return r; }
  static Value f32(std::vector<float> v) { Value r(Type::F32); *r.f32_ = std::move(v); return r; }
  static Value u32(std::vector<std::uint32_t> v) { Value r(Type::U32); *r.u32_ = std::move(v); return r; }

  Type type() const { return type_; }
  bool isUndefined() const { return type_ == Type::Undefined; }
  bool isNull() const { return type_ == Type::Null; }

  // Objects (insertion-ordered keys).
  void set(const std::string& key, Value v) {
    if (type_ != Type::Object) throw std::logic_error("Value::set on a non-object");
    for (auto& kv : *object_)
      if (kv.first == key) {
        kv.second = std::move(v);
        return;
      }
    object_->emplace_back(key, std::move(v));
  }
  const Value& operator[](const std::string& key) const {
    if (type_ == Type::Object)
      for (const auto& kv : *object_)
        if (kv.first == key) return kv.second;
    return undefinedValue();
  }
  const Value& operator[](const char* key) const { return (*this)[std::string(key)]; }
  const std::vector<std::pair<std::string, Value>>& items() const { return *object_; }

  // Arrays.
  void push(Value v) {
    if (type_ != Type::Array) throw std::logic_error("Value::push on a non-array");
    array_->push_back(std::move(v));
  }
  const Value& operator[](std::size_t i) const {
    if (type_ == Type::Array && i < array_->size()) return (*array_)[i];
    return undefinedValue();
  }
  template <class I, std::enable_if_t<std::is_integral_v<I> && !std::is_same_v<I, std::size_t>, int> = 0>
  const Value& operator[](I i) const { return (*this)[static_cast<std::size_t>(i)]; }
  const std::vector<Value>& elements() const { return *array_; }
  std::size_t size() const {
    switch (type_) {
      case Type::Array: return array_->size();
      case Type::Object: return object_->size();
      case Type::F64: return f64_->size();
      case Type::F32: return f32_->size();
      case Type::U32: return u32_->size();
      default: return 0;
    }
  }

  // Scalars and numeric arrays.
  template <class T>
  T as() const {
    if constexpr (std::is_same_v<T, std::string>) {
      if (type_ != Type::String) throw std::invalid_argument("expected a string");
      return *string_;
    } else if constexpr (std::is_same_v<T, bool>) {
      if (type_ != Type::Bool && type_ != Type::Number) throw std::invalid_argument("expected a boolean");
      return number_ != 0.0;
    } else {
      if (type_ != Type::Number && type_ != Type::Bool) throw std::invalid_argument("expected a number");
      return static_cast<T>(number_);
    }
  }
  /// The elements of a numeric array (plain or typed) as doubles / floats.
  std::vector<double> numbers() const { return convert<double>(); }
  std::vector<float> floats() const { return convert<float>(); }
  const std::vector<double>& f64Data() const { return *f64_; }
  const std::vector<float>& f32Data() const { return *f32_; }
  const std::vector<std::uint32_t>& u32Data() const { return *u32_; }
  const std::string& stringData() const { return *string_; }
  double numberData() const { return number_; }

 private:
  explicit Value(Type t) : type_(t) {
    switch (t) {
      case Type::Object: object_ = std::make_shared<std::vector<std::pair<std::string, Value>>>(); break;
      case Type::Array: array_ = std::make_shared<std::vector<Value>>(); break;
      case Type::F64: f64_ = std::make_shared<std::vector<double>>(); break;
      case Type::F32: f32_ = std::make_shared<std::vector<float>>(); break;
      case Type::U32: u32_ = std::make_shared<std::vector<std::uint32_t>>(); break;
      default: break;
    }
  }
  static const Value& undefinedValue() {
    static const Value u;
    return u;
  }
  template <class T>
  std::vector<T> convert() const {
    switch (type_) {
      case Type::F64: return std::vector<T>(f64_->begin(), f64_->end());
      case Type::F32: return std::vector<T>(f32_->begin(), f32_->end());
      case Type::U32: return std::vector<T>(u32_->begin(), u32_->end());
      case Type::Array: {
        std::vector<T> out;
        out.reserve(array_->size());
        for (const auto& v : *array_) out.push_back(v.as<T>());
        return out;
      }
      default: throw std::invalid_argument("expected a numeric array");
    }
  }

  Type type_ = Type::Undefined;
  double number_ = 0.0;
  std::shared_ptr<std::string> string_;
  std::shared_ptr<std::vector<std::pair<std::string, Value>>> object_;
  std::shared_ptr<std::vector<Value>> array_;
  std::shared_ptr<std::vector<double>> f64_;
  std::shared_ptr<std::vector<float>> f32_;
  std::shared_ptr<std::vector<std::uint32_t>> u32_;
};

/// Runs a library entry point (see analyses.hpp) on a specification.
Value call(const std::string& function, const Value& spec);
/// Names of the entry points `call` accepts.
const std::vector<std::string>& functions();

}  // namespace ccr::app
