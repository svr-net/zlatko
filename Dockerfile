# syntax=docker/dockerfile:1
#
# Multi-stage build of the library, its WebAssembly module and the web front end.
#
#   docker build -t zlatko-ccr .                    # web front end (nginx), default target
#   docker run --rm -p 8080:80 zlatko-ccr           # -> http://localhost:8080
#
#   docker build --target native -t zlatko-ccr:native .   # C++ build + unit tests
#   docker run --rm zlatko-ccr:native                     # runs the end-to-end demo
#
#   docker build --target e2e .                     # browser tests of every page (Playwright,
#                                                   # WebGPU on SwiftShader); behind a TLS-intercepting
#                                                   # proxy add --secret id=ca,src=<proxy CA bundle>
#   docker build --target wasm-artifacts --output web/wasm .   # export ccr.js/ccr.wasm to the host
#
# Every build stage runs its tests, so a successful build means they passed.

ARG EMSDK_VERSION=4.0.10
ARG PLAYWRIGHT_VERSION=1.56.1

# ---------------------------------------------------------------- native C++ library
FROM ubuntu:24.04 AS native-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY tests tests
COPY examples examples
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build \
 && ctest --test-dir build --output-on-failure \
 && cmake --install build --prefix /opt/ccr

FROM ubuntu:24.04 AS native
COPY --from=native-build /opt/ccr /opt/ccr
COPY --from=native-build /src/build/counterparty_exposure_demo /usr/local/bin/
COPY --from=native-build /src/build/ccr_tests /usr/local/bin/
CMD ["counterparty_exposure_demo"]

# ---------------------------------------------------------------- WebAssembly module
FROM emscripten/emsdk:${EMSDK_VERSION} AS wasm
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY wasm wasm
COPY web web
# Rebuild from source: never ship the committed binaries from the build context.
RUN rm -f web/wasm/ccr.js web/wasm/ccr.wasm \
 && emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build-wasm -j"$(nproc)" \
 && node web/tests/wasm.test.mjs

FROM scratch AS wasm-artifacts
COPY --from=wasm /src/web/wasm/ccr.js /src/web/wasm/ccr.wasm /

# ---------------------------------------------------------------- browser end-to-end tests
FROM mcr.microsoft.com/playwright:v${PLAYWRIGHT_VERSION}-noble AS e2e
ARG PLAYWRIGHT_VERSION
WORKDIR /web
COPY --from=wasm /src/web /web
# Behind a TLS-intercepting proxy, pass its CA: docker build --secret id=ca,src=/path/ca.pem ...
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/ca; fi \
 && npm init -y >/dev/null && npm install --no-audit --no-fund playwright@${PLAYWRIGHT_VERSION} \
 && node tests/e2e.mjs

# ---------------------------------------------------------------- web front end (default)
FROM nginx:1.27-alpine AS web
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=wasm /src/web /usr/share/nginx/html
RUN rm -rf /usr/share/nginx/html/tests
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
