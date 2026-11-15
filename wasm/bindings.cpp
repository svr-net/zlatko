// Embind wrapper: converts JavaScript values to ccr::app::Value and back, and exposes every
// library entry point (ccr::app::call) under its name. No numerics live here.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <string>
#include <vector>

#include "ccr/app/value.hpp"

using emscripten::val;
using ccr::app::Value;

namespace {

template <class T>
val typed(const std::vector<T>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

val toJs(const Value& v) {
  switch (v.type()) {
    case Value::Type::Undefined: return val::undefined();
    case Value::Type::Null: return val::null();
    case Value::Type::Bool: return val(v.as<bool>());
    case Value::Type::Number: return val(v.numberData());
    case Value::Type::String: return val(v.stringData());
    case Value::Type::F64: return typed(v.f64Data());
    case Value::Type::F32: return typed(v.f32Data());
    case Value::Type::U32: return typed(v.u32Data());
    case Value::Type::Array: {
      val a = val::array();
      for (const auto& e : v.elements()) a.call<void>("push", toJs(e));
      return a;
    }
    case Value::Type::Object: {
      val o = val::object();
      for (const auto& [k, e] : v.items()) o.set(k, toJs(e));
      return o;
    }
  }
  return val::undefined();
}

Value fromJs(const val& v) {
  const std::string t = v.typeOf().as<std::string>();
  if (t == "undefined") return Value();
  if (t == "number") return Value(v.as<double>());
  if (t == "boolean") return Value(v.as<bool>());
  if (t == "string") return Value(v.as<std::string>());
  if (v.isNull()) return Value::null();
  if (v.instanceof(val::global("Float32Array"))) return Value::f32(emscripten::convertJSArrayToNumberVector<float>(v));
  if (v.instanceof(val::global("Float64Array"))) return Value::f64(emscripten::convertJSArrayToNumberVector<double>(v));
  if (v.instanceof(val::global("Uint32Array"))) return Value::u32(emscripten::convertJSArrayToNumberVector<std::uint32_t>(v));
  if (val::global("Array").call<bool>("isArray", v)) {
    Value a = Value::array();
    const auto n = v["length"].as<std::size_t>();
    for (std::size_t i = 0; i < n; ++i) a.push(fromJs(v[i]));
    return a;
  }
  Value o = Value::object();
  const val keys = val::global("Object").call<val>("keys", v);
  const auto n = keys["length"].as<std::size_t>();
  for (std::size_t i = 0; i < n; ++i) {
    const std::string k = keys[i].as<std::string>();
    o.set(k, fromJs(v[k]));
  }
  return o;
}

// One JS function per library entry point: call(name, spec).
val callLibrary(std::string name, val spec) { return toJs(ccr::app::call(name, fromJs(spec))); }
val listFunctions() {
  val a = val::array();
  for (const auto& n : ccr::app::functions()) a.call<void>("push", val(n));
  return a;
}

}  // namespace

EMSCRIPTEN_BINDINGS(ccr) {
  emscripten::function("call", &callLibrary);
  emscripten::function("functions", &listFunctions);
}
