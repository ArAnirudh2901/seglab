#!/usr/bin/env bash
# Builds cpp/cv_refine.cpp → public/wasm/cv-refine.{js,wasm} (committed
# artifacts — end users never need Emscripten). Fixed 16 MB heap, no growth,
# no threads/SharedArrayBuffer, SIMD128, worker-only ES module.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v em++ >/dev/null 2>&1; then
    echo "error: em++ not found on PATH." >&2
    echo "  install: brew install emscripten   (or activate emsdk)" >&2
    exit 1
fi

mkdir -p "$ROOT/public/wasm"

em++ "$ROOT/cpp/cv_refine.cpp" \
    -O3 \
    -std=c++20 \
    -msimd128 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s ENVIRONMENT=worker \
    -s FILESYSTEM=0 \
    -s ALLOW_MEMORY_GROWTH=0 \
    -s INITIAL_MEMORY=16777216 \
    -s STACK_SIZE=1048576 \
    -s EXPORTED_FUNCTIONS='["_malloc","_free","_seeded_component_cleanup","_seeded_cleanup_multi","_fill_holes","_morphology_open","_morphology_close","_refine_mask","_refine_mask_multi"]' \
    -s EXPORTED_RUNTIME_METHODS='["HEAPU8"]' \
    -o "$ROOT/public/wasm/cv-refine.js"

SHA="$(shasum -a 256 "$ROOT/cpp/cv_refine.cpp" | cut -d' ' -f1)"
printf '\n// source cpp/cv_refine.cpp sha256=%s\n' "$SHA" >> "$ROOT/public/wasm/cv-refine.js"
echo "built public/wasm/cv-refine.js + cv-refine.wasm (source sha256 ${SHA:0:12}…)"
