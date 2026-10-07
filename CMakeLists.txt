cmake_minimum_required(VERSION 3.16)

project(zlatko_ccr
  VERSION 0.1.0
  DESCRIPTION "Modelling, pricing and hedging counterparty credit exposure"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CCR_BUILD_TESTS "Build the unit tests" ON)
option(CCR_BUILD_EXAMPLES "Build the example programs" ON)

add_library(ccr
  src/core/matrix.cpp
  src/core/regression.cpp
  src/core/time_grid.cpp
  src/market/yield_curve.cpp
  src/market/credit_curve.cpp
  src/models/hull_white.cpp
  src/models/lognormal_asset.cpp
  src/models/cir_intensity.cpp
  src/models/scenario_generator.cpp
  src/instruments/trade.cpp
  src/instruments/interest_rate_swap.cpp
  src/instruments/asset_forward.cpp
  src/instruments/european_option.cpp
  src/instruments/bermudan_swaption.cpp
  src/instruments/credit_default_swap.cpp
  src/exposure/collateral.cpp
  src/exposure/exposure_profile.cpp
  src/exposure/exposure_engine.cpp
  src/exposure/allocation.cpp
  src/cva/cva.cpp
  src/hedging/sensitivities.cpp
)
add_library(ccr::ccr ALIAS ccr)

target_include_directories(ccr PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ccr PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
elseif(MSVC)
  target_compile_options(ccr PRIVATE /W4)
endif()

if(CCR_BUILD_TESTS)
  enable_testing()
  add_executable(ccr_tests
    tests/test_main.cpp
    tests/test_core.cpp
    tests/test_market.cpp
    tests/test_models.cpp
    tests/test_instruments.cpp
    tests/test_exposure.cpp
    tests/test_cva.cpp
    tests/test_hedging.cpp
  )
  target_link_libraries(ccr_tests PRIVATE ccr)
  add_test(NAME ccr_tests COMMAND ccr_tests)
endif()

if(CCR_BUILD_EXAMPLES)
  add_executable(counterparty_exposure_demo examples/counterparty_exposure_demo.cpp)
  target_link_libraries(counterparty_exposure_demo PRIVATE ccr)
endif()

include(GNUInstallDirs)
install(TARGETS ccr EXPORT ccrTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ccr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ccrTargets NAMESPACE ccr:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccr)
