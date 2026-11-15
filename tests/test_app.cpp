#include <cmath>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "ccr/app/value.hpp"
#include "test_framework.hpp"

using ccr::app::Value;

namespace {

Value obj(std::initializer_list<std::pair<const char*, Value>> kv) {
  Value o = Value::object();
  for (const auto& [k, v] : kv) o.set(k, v);
  return o;
}
Value list(std::initializer_list<Value> xs) {
  Value a = Value::array();
  for (const auto& x : xs) a.push(x);
  return a;
}
Value credit(const char* name, double spread) {
  return obj({{"name", name}, {"recovery", 0.4},
              {"quotes", list({obj({{"maturity", 1}, {"spread", spread}}), obj({{"maturity", 5}, {"spread", spread * 1.4}})})},
              {"cir", obj({{"kappa", 0.4}, {"theta", 0.02}, {"xi", 0.1}, {"y0", spread * 1.5}, {"substeps", 4}})}});
}
Value swap(const char* id, double notional, double tenor, const char* direction) {
  return obj({{"type", "swap"}, {"id", id}, {"notional", notional}, {"fixedRate", 0.034}, {"start", 0}, {"tenor", tenor}, {"freq", 2}, {"direction", direction}});
}
Value forward(const char* id, double notional, double maturity) {
  return obj({{"type", "forward"}, {"id", id}, {"asset", "EURUSD"}, {"notional", notional}, {"strike", 1.12}, {"maturity", maturity}});
}
Value nettingSet(const char* id, const char* cpty, Value trades, bool csa = false) {
  return obj({{"id", id}, {"counterparty", cpty}, {"trades", std::move(trades)},
              {"csa", obj({{"enabled", csa}, {"thresholdCounterparty", 1e5}, {"thresholdOwn", 1e5}, {"mta", 1e4}, {"mpr", 10.0 / 250.0}})}});
}

// Two counterparties, one FX asset; correlation order [rates, EURUSD, A, B].
Value baseSpec() {
  return obj({
      {"domestic", obj({{"times", list({1, 5, 10})}, {"rates", list({0.03, 0.033, 0.035})}})},
      {"hw", obj({{"a", 0.04}, {"sigma", 0.009}})},
      {"assets", list({obj({{"name", "EURUSD"}, {"spot", 1.1}, {"vol", 0.1}, {"carry", obj({{"flat", 0.02}})}})})},
      {"credits", list({credit("A", 0.008), credit("B", 0.015)})},
      {"correlation", list({list({1, 0.2, 0.1, 0}), list({0.2, 1, 0, 0.3}), list({0.1, 0, 1, 0.5}), list({0, 0.3, 0.5, 1})})},
      {"sim", obj({{"numPaths", 1024}, {"seed", 7}, {"antithetic", true}})},
      {"grid", obj({{"type", "standard"}, {"horizon", 5}})},
      {"own", obj({{"recovery", 0.4}, {"quotes", list({obj({{"maturity", 5}, {"spread", 0.006}})})}})},
      {"imInnerPaths", 16},
  });
}

Value book(Value spec, Value sets, const char* analysis) {
  spec.set("scope", "book");
  spec.set("analysis", analysis);
  spec.set("book", obj({{"name", "test book"}, {"nettingSets", std::move(sets)}}));
  return ccr::app::call("cpuKernels", spec);
}

std::vector<double> numbers(const Value& v) { return v.numbers(); }

}  // namespace

TEST(app_book_netting_sets_share_their_scenarios) {
  // y holds the trades of x and z (with another counterparty): if every netting set draws the
  // same scenarios, the expected value of y is that of x plus that of z on every date, and
  // EE(y) <= EE(x) + EE(z), up to f32 rounding rather than Monte Carlo error.
  const Value r = book(baseSpec(),
                       list({nettingSet("x", "A", list({swap("s", 1e7, 5, "payer")})),
                             nettingSet("y", "B", list({swap("s", 1e7, 5, "payer"), forward("f", 5e6, 3)})),
                             nettingSet("z", "A", list({forward("f", 5e6, 3)}))}),
                       "exposure");
  CHECK(r["error"].isUndefined());
  CHECK(r["scope"].stringData() == "book");
  const auto& sets = r["nettingSets"];
  CHECK(sets.size() == 3);
  const auto evx = numbers(sets[0]["result"]["profile"]["expectedValue"]), evy = numbers(sets[1]["result"]["profile"]["expectedValue"]),
             evz = numbers(sets[2]["result"]["profile"]["expectedValue"]);
  const auto eex = numbers(sets[0]["result"]["profile"]["ee"]), eey = numbers(sets[1]["result"]["profile"]["ee"]),
             eez = numbers(sets[2]["result"]["profile"]["ee"]);
  CHECK(evx.size() == evy.size() && evy.size() == evz.size() && evx.size() > 10);
  for (std::size_t j = 0; j < evx.size(); ++j) {
    CHECK_NEAR(evy[j], evx[j] + evz[j], 2e-6 * 1.5e7);
    CHECK(eey[j] <= eex[j] + eez[j] + 2e-6 * 1.5e7);
  }
  // The book profile is the sum of the netting-set profiles.
  const auto total = numbers(r["total"]["profile"]["ee"]);
  for (std::size_t j = 0; j < total.size(); ++j) CHECK_NEAR(total[j], eex[j] + eey[j] + eez[j], 1e-9 * (1.0 + total[j]));
}

TEST(app_book_of_one_netting_set_is_the_netting_set) {
  Value spec = baseSpec();
  const Value trades = list({swap("s", 1e7, 5, "payer"), forward("f", -5e6, 3)});
  Value single = baseSpec();
  // The single-netting-set analysis with counterparty A only (a book simulates its counterparty only).
  single.set("credits", list({credit("A", 0.008)}));
  single.set("correlation", list({list({1, 0.2, 0.1}), list({0.2, 1, 0}), list({0.1, 0, 1})}));
  single.set("trades", trades);
  single.set("analysis", "cva");
  const Value s = ccr::app::call("cpuKernels", single);
  const Value r = book(spec, list({nettingSet("only", "A", trades)}), "cva");
  CHECK(s["error"].isUndefined() && r["error"].isUndefined());
  CHECK_NEAR(r["total"]["cva"].numberData(), s["cva"].numberData(), 1e-9 * s["cva"].numberData());
  CHECK_NEAR(r["total"]["pathwiseCva"].numberData(), s["pathwiseCva"].numberData(), 1e-9 * s["pathwiseCva"].numberData());
}

TEST(app_book_xva_and_cva_add_up_by_netting_set_and_counterparty) {
  const Value sets = list({nettingSet("x", "A", list({swap("s1", 1e7, 5, "payer"), forward("f", 5e6, 3)})),
                           nettingSet("y", "B", list({swap("s2", 6e6, 4, "receiver")}), true),
                           nettingSet("z", "A", list({forward("g", -3e6, 2)}))});
  const Value x = book(baseSpec(), sets, "xva");
  CHECK(x["error"].isUndefined());
  double total = 0, cva = 0;
  for (std::size_t i = 0; i < 3; ++i) total += x["nettingSets"][i]["summary"]["total"].numberData(), cva += x["nettingSets"][i]["summary"]["cva"].numberData();
  CHECK_NEAR(x["total"]["xva"]["total"].numberData(), total, 1e-9 * std::fabs(total));
  const auto& byCp = x["total"]["byCounterparty"];
  CHECK(byCp.size() == 2 && byCp[0]["name"].stringData() == "A" && byCp[0]["nettingSets"].size() == 2);
  CHECK_NEAR(byCp[0]["cva"].numberData() + byCp[1]["cva"].numberData(), cva, 1e-9 * cva);
  CHECK(x["total"]["trades"].size() == 4);
  for (const char* a : {"collateral", "allocation", "wrongWayRisk", "hedging", "instruments"}) {
    Value spec = baseSpec();
    spec.set("rhos", list({-0.5, 0.5}));
    spec.set("bumps", list({0.01}));
    const Value r = book(spec, sets, a);
    CHECK(r["error"].isUndefined());
    CHECK(r["nettingSets"].size() == 3);
  }
  CHECK(!book(baseSpec(), sets, "models")["error"].isUndefined());  // not a portfolio analysis
  CHECK(!book(baseSpec(), list({nettingSet("x", "C", list({swap("s", 1e6, 5, "payer")}))}), "cva")["error"].isUndefined());
}
