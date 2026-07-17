#!/usr/bin/env bash
# Build raw-develop.wasm — LibRaw sensor develop + libjpeg encode, worker-only
# ES module. The FALLBACK decoder for RAW files with no usable embedded JPEG
# preview (image-raw.js handles the common case). Lazy: never fetched unless
# such a file is opened, and the worker is disposed after each develop.
#
# Memory grows (sensor sizes vary) but is capped at MAXIMUM_MEMORY; the wrapper
# also refuses sensors above a megapixel cap before unpack. libjpeg (emscripten
# port) gives both the decode paths LibRaw needs (lossy DNG / Kodak) and the
# in-worker JPEG encode, so only a compact Blob ever crosses back.
#
# Requires Emscripten (em++) on PATH: `brew install emscripten` or emsdk.
# LibRaw source is fetched once (pinned) into build/libraw (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."

LIBRAW_TAG="0.22.2"
LIBRAW_DIR="build/libraw-${LIBRAW_TAG}"

if [ ! -f "${LIBRAW_DIR}/libraw/libraw.h" ]; then
  echo "==> Fetching LibRaw ${LIBRAW_TAG} (pinned) into ${LIBRAW_DIR}…"
  rm -rf "${LIBRAW_DIR}"
  mkdir -p "$(dirname "${LIBRAW_DIR}")"
  git clone -q --depth 1 --branch "${LIBRAW_TAG}" https://github.com/LibRaw/LibRaw.git "${LIBRAW_DIR}"
else
  echo "==> Reusing ${LIBRAW_DIR} (delete build/ to refetch)."
fi

mkdir -p public/wasm

# LibRaw translation units (the emscripten-proven subset — no OpenMP, no DNG SDK,
# no RawSpeed; those glue files compile to stubs without their -D guards).
LIBRAW_SRC=(
  "${LIBRAW_DIR}"/src/*.cpp
  "${LIBRAW_DIR}"/src/decoders/*.cpp
  "${LIBRAW_DIR}"/src/decompressors/*.cpp
  "${LIBRAW_DIR}"/src/demosaic/*.cpp
  "${LIBRAW_DIR}"/src/integration/*.cpp
  "${LIBRAW_DIR}"/src/metadata/*.cpp
  "${LIBRAW_DIR}"/src/tables/*.cpp
  "${LIBRAW_DIR}"/src/utils/*.cpp
  "${LIBRAW_DIR}"/src/x3f/*.cpp
  "${LIBRAW_DIR}"/src/write/apply_profile.cpp
  "${LIBRAW_DIR}"/src/write/file_write.cpp
  "${LIBRAW_DIR}"/src/write/tiff_writer.cpp
  "${LIBRAW_DIR}"/src/preprocessing/raw2image.cpp
  "${LIBRAW_DIR}"/src/preprocessing/ext_preprocess.cpp
  "${LIBRAW_DIR}"/src/preprocessing/subtract_black.cpp
  "${LIBRAW_DIR}"/src/postprocessing/aspect_ratio.cpp
  "${LIBRAW_DIR}"/src/postprocessing/dcraw_process.cpp
  "${LIBRAW_DIR}"/src/postprocessing/mem_image.cpp
  "${LIBRAW_DIR}"/src/postprocessing/postprocessing_aux.cpp
  "${LIBRAW_DIR}"/src/postprocessing/postprocessing_utils.cpp
  "${LIBRAW_DIR}"/src/postprocessing/postprocessing_utils_dcrdefs.cpp
)

em++ cpp/raw_develop.cpp "${LIBRAW_SRC[@]}" \
  -O3 \
  -std=c++17 \
  -msimd128 \
  -fexceptions \
  -DNDEBUG \
  -DUSE_JPEG \
  -DUSE_JPEG8 \
  -DLIBRAW_NODLL \
  -I"${LIBRAW_DIR}" \
  -s USE_LIBJPEG=1 \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s ENVIRONMENT=worker \
  -s FILESYSTEM=0 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s MAXIMUM_MEMORY=805306368 \
  -s STACK_SIZE=2097152 \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_rd_develop","_rd_release","_rd_error","_rd_width","_rd_height","_rd_jpeg","_rd_jpeg_len"]' \
  -s EXPORTED_RUNTIME_METHODS='["HEAPU8","UTF8ToString"]' \
  -o public/wasm/raw-develop.js

ls -la public/wasm/raw-develop.*
