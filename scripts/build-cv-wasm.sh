#!/usr/bin/env bash
# Build cv-refine.wasm (SIMD, fixed 16 MiB heap, worker-only ES module).
# Requires Emscripten (em++) on PATH: `brew install emscripten` or emsdk.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p public/wasm
em++ cpp/cv_refine.cpp \
  -O3 \
  -std=c++20 \
  -msimd128 \
  -flto \
  -DNDEBUG \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s ENVIRONMENT=worker \
  -s FILESYSTEM=0 \
  -s ALLOW_MEMORY_GROWTH=0 \
  -s INITIAL_MEMORY=16777216 \
  -s STACK_SIZE=1048576 \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_seeded_component_cleanup","_fill_holes","_morphology_open","_morphology_close","_refine_mask"]' \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8"]' \
  -o public/wasm/cv-refine.js

ls -la public/wasm/
