// cv_refine — see cv_refine.h. Design rules: fixed reusable workspace,
// overflow-checked sizes, scanline flood fill with a preallocated queue
// (no recursion, no per-pixel allocation), simple loops that -msimd128
// can auto-vectorize where profitable.
#include "cv_refine.h"
#include <cstddef>
#include <cstring>
#include <new>

namespace {

constexpr size_t MAX_PIXELS = static_cast<size_t>(CV_MAX_SIDE) * CV_MAX_SIDE;

struct Workspace {
    int32_t* queue = nullptr;   // flood-fill queue, one slot per pixel
    int32_t* labels = nullptr;  // component labels / flags
    size_t capacity = 0;        // pixels
};

Workspace ws;

bool valid_dims(int width, int height) {
    if (width <= 0 || height <= 0) return false;
    if (width > CV_MAX_SIDE || height > CV_MAX_SIDE) return false;
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    return pixels <= MAX_PIXELS;
}

bool ensure_workspace(size_t pixels) {
    if (pixels > MAX_PIXELS) return false;
    if (pixels <= ws.capacity) return true;
    delete[] ws.queue;
    delete[] ws.labels;
    ws.queue = new (std::nothrow) int32_t[pixels];
    ws.labels = new (std::nothrow) int32_t[pixels];
    if (!ws.queue || !ws.labels) {
        delete[] ws.queue;
        delete[] ws.labels;
        ws.queue = nullptr;
        ws.labels = nullptr;
        ws.capacity = 0;
        return false;
    }
    ws.capacity = pixels;
    return true;
}

// 4-connected flood from `start` over pixels where on(mask[i]) matches
// `value`, labelling them `label`. Returns the component area.
inline bool on(uint8_t v) { return v >= 128; }

size_t flood(const uint8_t* mask, int w, int h, int32_t start, bool value, int32_t label) {
    size_t head = 0;
    size_t tail = 0;
    ws.queue[tail++] = start;
    ws.labels[start] = label;
    size_t area = 0;
    const int32_t size = w * h;
    while (head < tail) {
        const int32_t i = ws.queue[head++];
        area += 1;
        const int32_t x = i % w;
        const int32_t nbr[4] = { i - 1, i + 1, i - w, i + w };
        const bool ok[4] = { x > 0, x < w - 1, i >= w, i < size - w };
        for (int k = 0; k < 4; k += 1) {
            const int32_t j = nbr[k];
            if (ok[k] && ws.labels[j] == 0 && on(mask[j]) == value) {
                ws.labels[j] = label;
                ws.queue[tail++] = j;
            }
        }
    }
    return area;
}

// Separable binary dilate (isMax) / erode over a square radius. src/dst are
// distinct buffers; values are strictly 0/255 afterwards. Border-clamped.
void morph_pass(const uint8_t* src, uint8_t* dst, int w, int h, int radius, bool isMax) {
    // Horizontal into dst.
    for (int y = 0; y < h; y += 1) {
        const uint8_t* row = src + static_cast<size_t>(y) * w;
        uint8_t* orow = dst + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; x += 1) {
            const int x0 = x - radius < 0 ? 0 : x - radius;
            const int x1 = x + radius >= w ? w - 1 : x + radius;
            uint8_t v = isMax ? 0 : 255;
            for (int k = x0; k <= x1; k += 1) {
                const uint8_t b = on(row[k]) ? 255 : 0;
                v = isMax ? (b > v ? b : v) : (b < v ? b : v);
                if (v == (isMax ? 255 : 0)) break;
            }
            orow[x] = v;
        }
    }
}

void morph_vertical(uint8_t* buf, uint8_t* col, int w, int h, int radius, bool isMax) {
    // Vertical in place, using `col` as one column of scratch.
    for (int x = 0; x < w; x += 1) {
        for (int y = 0; y < h; y += 1) col[y] = buf[static_cast<size_t>(y) * w + x];
        for (int y = 0; y < h; y += 1) {
            const int y0 = y - radius < 0 ? 0 : y - radius;
            const int y1 = y + radius >= h ? h - 1 : y + radius;
            uint8_t v = isMax ? 0 : 255;
            for (int k = y0; k <= y1; k += 1) {
                const uint8_t b = col[k];
                v = isMax ? (b > v ? b : v) : (b < v ? b : v);
                if (v == (isMax ? 255 : 0)) break;
            }
            buf[static_cast<size_t>(y) * w + x] = v;
        }
    }
}

// dilate/erode mask -> mask using scratch (w*h) + workspace queue as column scratch.
void morph_binary(uint8_t* mask, uint8_t* scratch, int w, int h, int radius, bool isMax) {
    morph_pass(mask, scratch, w, h, radius, isMax);
    // Reuse the int32 queue's storage as the column scratch (h bytes needed).
    uint8_t* col = reinterpret_cast<uint8_t*>(ws.queue);
    morph_vertical(scratch, col, w, h, radius, isMax);
    std::memcpy(mask, scratch, static_cast<size_t>(w) * h);
}

} // namespace

extern "C" {

void seeded_component_cleanup(uint8_t* mask, int width, int height, int seed_x, int seed_y, int min_area) {
    if (!mask || !valid_dims(width, height)) return;
    const size_t pixels = static_cast<size_t>(width) * height;
    if (!ensure_workspace(pixels)) return;
    std::memset(ws.labels, 0, pixels * sizeof(int32_t));

    int32_t seedLabel = 0;
    int32_t next = 0;
    // Label every foreground component, remembering the seeded one.
    for (size_t i = 0; i < pixels; i += 1) {
        if (!on(mask[i]) || ws.labels[i]) continue;
        next += 1;
        const size_t area = flood(mask, width, height, static_cast<int32_t>(i), true, next);
        // flood() stored the area implicitly; recompute cheaply via queue? —
        // store area per label in the queue tail instead: small label count,
        // reuse a fixed array below.
        (void)area;
    }
    // Second pass: per-label areas (labels are 1..next, bounded by pixels).
    // Reuse queue storage as the area table.
    int32_t* areas = ws.queue;
    for (int32_t k = 0; k <= next; k += 1) areas[k] = 0;
    for (size_t i = 0; i < pixels; i += 1) if (ws.labels[i]) areas[ws.labels[i]] += 1;

    if (seed_x >= 0 && seed_y >= 0 && seed_x < width && seed_y < height) {
        // Accept a seed a few pixels off the mask (clicks can land just outside).
        for (int r = 0; r <= 4 && !seedLabel; r += 1) {
            for (int dy = -r; dy <= r && !seedLabel; dy += 1) {
                for (int dx = -r; dx <= r; dx += 1) {
                    const int px = seed_x + dx;
                    const int py = seed_y + dy;
                    if (px < 0 || py < 0 || px >= width || py >= height) continue;
                    const int32_t l = ws.labels[static_cast<size_t>(py) * width + px];
                    if (l) { seedLabel = l; break; }
                }
            }
        }
    }

    for (size_t i = 0; i < pixels; i += 1) {
        const int32_t l = ws.labels[i];
        if (!l) continue;
        const bool keep = (l == seedLabel) || (min_area <= 0 ? seedLabel == 0 : areas[l] >= min_area);
        if (!keep) mask[i] = 0;
    }
}

namespace {
// Shared hole fill. max_hole <= 0 fills every hole; otherwise only holes at
// or below max_hole pixels are filled (large legitimate gaps stay open).
void fill_holes_impl(uint8_t* mask, int width, int height, int32_t max_hole) {
    if (!mask) return;
    const size_t pixels = static_cast<size_t>(width) * height;
    if (!ensure_workspace(pixels)) return;
    std::memset(ws.labels, 0, pixels * sizeof(int32_t));
    // Flood the border-connected background with label 1.
    for (int x = 0; x < width; x += 1) {
        const int32_t top = x;
        const int32_t bot = (height - 1) * width + x;
        if (!on(mask[top]) && !ws.labels[top]) flood(mask, width, height, top, false, 1);
        if (!on(mask[bot]) && !ws.labels[bot]) flood(mask, width, height, bot, false, 1);
    }
    for (int y = 0; y < height; y += 1) {
        const int32_t left = y * width;
        const int32_t right = y * width + width - 1;
        if (!on(mask[left]) && !ws.labels[left]) flood(mask, width, height, left, false, 1);
        if (!on(mask[right]) && !ws.labels[right]) flood(mask, width, height, right, false, 1);
    }
    if (max_hole <= 0) {
        for (size_t i = 0; i < pixels; i += 1) {
            if (!on(mask[i]) && ws.labels[i] == 0) mask[i] = 255;
        }
        return;
    }
    // Label the holes (2, 3, ...) and measure them; fill only the small ones.
    int32_t next = 1;
    for (size_t i = 0; i < pixels; i += 1) {
        if (on(mask[i]) || ws.labels[i]) continue;
        next += 1;
        flood(mask, width, height, static_cast<int32_t>(i), false, next);
    }
    int32_t* areas = ws.queue;
    for (int32_t k = 0; k <= next; k += 1) areas[k] = 0;
    for (size_t i = 0; i < pixels; i += 1) if (!on(mask[i]) && ws.labels[i] > 1) areas[ws.labels[i]] += 1;
    for (size_t i = 0; i < pixels; i += 1) {
        const int32_t l = ws.labels[i];
        if (!on(mask[i]) && l > 1 && areas[l] <= max_hole) mask[i] = 255;
    }
}
} // namespace

void fill_holes(uint8_t* mask, int width, int height) {
    if (!mask || !valid_dims(width, height)) return;
    fill_holes_impl(mask, width, height, 0);
}

void morphology_open(uint8_t* mask, uint8_t* scratch, int width, int height, int kernel_radius) {
    if (!mask || !scratch || !valid_dims(width, height) || kernel_radius <= 0) return;
    if (!ensure_workspace(static_cast<size_t>(width) * height)) return;
    morph_binary(mask, scratch, width, height, kernel_radius, false); // erode
    morph_binary(mask, scratch, width, height, kernel_radius, true);  // dilate
}

void morphology_close(uint8_t* mask, uint8_t* scratch, int width, int height, int kernel_radius) {
    if (!mask || !scratch || !valid_dims(width, height) || kernel_radius <= 0) return;
    if (!ensure_workspace(static_cast<size_t>(width) * height)) return;
    morph_binary(mask, scratch, width, height, kernel_radius, true);  // dilate
    morph_binary(mask, scratch, width, height, kernel_radius, false); // erode
}

void refine_mask(
    const uint8_t* rgb, uint8_t* mask, uint8_t* scratch,
    int width, int height, int seed_x, int seed_y,
    int min_area, int open_radius, int close_radius) {
    (void)rgb; // reserved for edge-guided refinement (Phase B)
    if (!mask || !valid_dims(width, height)) return;
    seeded_component_cleanup(mask, width, height, seed_x, seed_y, min_area);
    if (open_radius > 0 && scratch) morphology_open(mask, scratch, width, height, open_radius);
    if (close_radius > 0 && scratch) morphology_close(mask, scratch, width, height, close_radius);
    // Pipeline fill is bounded like the JS hygiene pass: pinholes close, big
    // legitimate gaps (background through a frame) stay open.
    size_t fg = 0;
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; i += 1) if (on(mask[i])) fg += 1;
    const int32_t max_hole = static_cast<int32_t>(fg / 100 > 64 ? fg / 100 : 64);
    fill_holes_impl(mask, width, height, max_hole);
}

}
