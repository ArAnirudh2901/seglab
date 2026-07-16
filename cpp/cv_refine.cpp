// cv_refine — see cv_refine.h for the contract. wasm32 target, no libc++
// containers, no allocation: all workspaces are fixed static arenas sized
// for the 768×768 ceiling (~7.4 MB total, inside the fixed 16 MB heap).

#include "cv_refine.h"

#include <string.h>

namespace {

constexpr int kMaxSide = 768;
constexpr int kMaxPixels = kMaxSide * kMaxSide;      // 589,824
constexpr int kMaxLabels = kMaxPixels / 2 + 2;       // checkerboard worst case

int32_t g_labels[kMaxPixels];
int32_t g_stack[kMaxPixels];
int32_t g_areas[kMaxLabels];
uint8_t g_keep[kMaxLabels];
uint8_t g_bin[kMaxPixels];

bool valid_dims(int w, int h) {
    if (w <= 0 || h <= 0 || w > kMaxSide || h > kMaxSide) return false;
    return static_cast<int64_t>(w) * static_cast<int64_t>(h) <= kMaxPixels;
}

// 4-connected labeling of g_bin into g_labels/g_areas (labels 1..n).
// Iterative stack flood fill — mirrors js/sam-core.js labelComponents.
int label_components(int w, int h) {
    const int size = w * h;
    memset(g_labels, 0, sizeof(int32_t) * size);
    int next = 0;
    for (int start = 0; start < size; ++start) {
        if (!g_bin[start] || g_labels[start]) continue;
        if (next >= kMaxLabels - 1) break;  // unreachable with 4-connectivity
        ++next;
        int area = 0;
        int top = 0;
        g_stack[top++] = start;
        g_labels[start] = next;
        while (top > 0) {
            const int i = g_stack[--top];
            ++area;
            const int x = i % w;
            if (x > 0 && g_bin[i - 1] && !g_labels[i - 1]) { g_labels[i - 1] = next; g_stack[top++] = i - 1; }
            if (x < w - 1 && g_bin[i + 1] && !g_labels[i + 1]) { g_labels[i + 1] = next; g_stack[top++] = i + 1; }
            if (i >= w && g_bin[i - w] && !g_labels[i - w]) { g_labels[i - w] = next; g_stack[top++] = i - w; }
            if (i < w * (h - 1) && g_bin[i + w] && !g_labels[i + w]) { g_labels[i + w] = next; g_stack[top++] = i + w; }
        }
        g_areas[next - 1] = area;
    }
    return next;
}

// Component label at/near a seed — expanding Chebyshev rings, radius 8
// (a positive click can land a few pixels outside the decoded mask).
int label_near_seed(int w, int h, int sx, int sy) {
    constexpr int kRadius = 8;
    for (int r = 0; r <= kRadius; ++r) {
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                if ((dx < 0 ? -dx : dx) != r && (dy < 0 ? -dy : dy) != r) continue;  // ring only
                const int px = sx + dx;
                const int py = sy + dy;
                if (px < 0 || py < 0 || px >= w || py >= h) continue;
                const int l = g_labels[py * w + px];
                if (l) return l;
            }
        }
    }
    return 0;
}

void binarize(const uint8_t* mask, int size) {
    for (int i = 0; i < size; ++i) g_bin[i] = mask[i] >= 128 ? 1 : 0;
}

// Separable binary dilate/erode (Chebyshev square kernel), src -> dst via
// g_bin as the intermediate row pass. Values 0/255.
void morph_pass(const uint8_t* src, uint8_t* dst, int w, int h, int radius, bool is_max) {
    const uint8_t init = is_max ? 0 : 255;
    // Horizontal into g_bin (reused as row buffer, 0/255 domain).
    for (int y = 0; y < h; ++y) {
        const int row = y * w;
        for (int x = 0; x < w; ++x) {
            uint8_t v = init;
            const int lo = x - radius < 0 ? 0 : x - radius;
            const int hi = x + radius >= w ? w - 1 : x + radius;
            for (int i = lo; i <= hi; ++i) {
                const uint8_t s = src[row + i];
                v = is_max ? (s > v ? s : v) : (s < v ? s : v);
            }
            g_bin[row + x] = v;
        }
    }
    // Vertical into dst.
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            uint8_t v = init;
            const int lo = y - radius < 0 ? 0 : y - radius;
            const int hi = y + radius >= h ? h - 1 : y + radius;
            for (int i = lo; i <= hi; ++i) {
                const uint8_t s = g_bin[i * w + x];
                v = is_max ? (s > v ? s : v) : (s < v ? s : v);
            }
            dst[y * w + x] = v;
        }
    }
}

void snap_binary(uint8_t* mask, int size) {
    for (int i = 0; i < size; ++i) mask[i] = mask[i] >= 128 ? 255 : 0;
}

}  // namespace

extern "C" {

int seeded_cleanup_multi(uint8_t* mask, int width, int height,
                         const int32_t* seeds, int n_seeds, int min_area) {
    if (!valid_dims(width, height)) return CV_ERR_DIMS;
    if (!mask || n_seeds < 0 || (n_seeds > 0 && !seeds)) return CV_ERR_ARGS;
    const int size = width * height;

    binarize(mask, size);
    const int n = label_components(width, height);
    if (n == 0) { memset(mask, 0, size); return 0; }

    memset(g_keep, 0, n);
    int kept = 0;
    int largest_kept = 0;
    for (int s = 0; s < n_seeds; ++s) {
        const int l = label_near_seed(width, height, seeds[s * 2], seeds[s * 2 + 1]);
        if (l && !g_keep[l - 1]) {
            g_keep[l - 1] = 1;
            ++kept;
            if (g_areas[l - 1] > largest_kept) largest_kept = g_areas[l - 1];
        }
    }
    if (kept == 0) {
        // No seed hit anything (box/lasso edge cases) — keep the largest.
        int best = 0;
        for (int k = 1; k < n; ++k) if (g_areas[k] > g_areas[best]) best = k;
        g_keep[best] = 1;
        kept = 1;
        largest_kept = g_areas[best];
    }
    // Occluder-split survival: unseeded components above the area floor stay.
    const int floor_area = min_area > largest_kept / 100 ? min_area : largest_kept / 100;
    for (int k = 0; k < n; ++k) {
        if (!g_keep[k] && g_areas[k] >= floor_area) { g_keep[k] = 1; ++kept; }
    }
    for (int i = 0; i < size; ++i) {
        const int l = g_labels[i];
        mask[i] = (g_bin[i] && l && g_keep[l - 1]) ? 255 : 0;
    }
    return kept;
}

int seeded_component_cleanup(uint8_t* mask, int width, int height,
                             int seed_x, int seed_y, int min_area) {
    const int32_t seed[2] = { seed_x, seed_y };
    return seeded_cleanup_multi(mask, width, height, seed, 1, min_area);
}

int fill_holes(uint8_t* mask, int width, int height) {
    if (!valid_dims(width, height)) return CV_ERR_DIMS;
    if (!mask) return CV_ERR_ARGS;
    const int size = width * height;

    int fg = 0;
    for (int i = 0; i < size; ++i) {
        const uint8_t on = mask[i] >= 128;
        g_bin[i] = !on;  // label the BACKGROUND
        fg += on;
    }
    if (!fg) return 0;
    const int n = label_components(width, height);

    // Background components touching the border are outside; the rest are
    // holes — fill those below max(64, 1% of foreground).
    memset(g_keep, 0, n);  // g_keep = touches-border flag here
    for (int x = 0; x < width; ++x) {
        const int t = g_labels[x];
        const int b = g_labels[(height - 1) * width + x];
        if (t) g_keep[t - 1] = 1;
        if (b) g_keep[b - 1] = 1;
    }
    for (int y = 0; y < height; ++y) {
        const int l = g_labels[y * width];
        const int r = g_labels[y * width + width - 1];
        if (l) g_keep[l - 1] = 1;
        if (r) g_keep[r - 1] = 1;
    }
    const int max_hole = fg / 100 > 64 ? fg / 100 : 64;
    int filled = 0;
    for (int i = 0; i < size; ++i) {
        const int l = g_labels[i];
        if (l && !g_keep[l - 1] && g_areas[l - 1] <= max_hole) {
            mask[i] = 255;
            ++filled;
        }
    }
    return filled;
}

int morphology_open(uint8_t* mask, uint8_t* scratch, int width, int height,
                    int kernel_radius) {
    if (!valid_dims(width, height)) return CV_ERR_DIMS;
    if (!mask || !scratch || kernel_radius < 0 || kernel_radius > kMaxSide) return CV_ERR_ARGS;
    if (kernel_radius == 0) return 0;
    snap_binary(mask, width * height);
    morph_pass(mask, scratch, width, height, kernel_radius, /*is_max=*/false);  // erode
    morph_pass(scratch, mask, width, height, kernel_radius, /*is_max=*/true);   // dilate
    return 0;
}

int morphology_close(uint8_t* mask, uint8_t* scratch, int width, int height,
                     int kernel_radius) {
    if (!valid_dims(width, height)) return CV_ERR_DIMS;
    if (!mask || !scratch || kernel_radius < 0 || kernel_radius > kMaxSide) return CV_ERR_ARGS;
    if (kernel_radius == 0) return 0;
    snap_binary(mask, width * height);
    morph_pass(mask, scratch, width, height, kernel_radius, /*is_max=*/true);   // dilate
    morph_pass(scratch, mask, width, height, kernel_radius, /*is_max=*/false);  // erode
    return 0;
}

int refine_mask_multi(uint8_t* mask, uint8_t* scratch, int width, int height,
                      const int32_t* seeds, int n_seeds,
                      int min_area, int open_radius, int close_radius) {
    const int kept = seeded_cleanup_multi(mask, width, height, seeds, n_seeds, min_area);
    if (kept < 0) return kept;
    int rc = fill_holes(mask, width, height);
    if (rc < 0) return rc;
    if (close_radius > 0) {
        rc = morphology_close(mask, scratch, width, height, close_radius);
        if (rc < 0) return rc;
    }
    if (open_radius > 0) {
        rc = morphology_open(mask, scratch, width, height, open_radius);
        if (rc < 0) return rc;
    }
    return kept;
}

int refine_mask(const uint8_t* rgb, uint8_t* mask, uint8_t* scratch,
                int width, int height, int seed_x, int seed_y,
                int min_area, int open_radius, int close_radius) {
    (void)rgb;  // reserved for Phase B edge refinement
    const int32_t seed[2] = { seed_x, seed_y };
    return refine_mask_multi(mask, scratch, width, height, seed, 1,
                             min_area, open_radius, close_radius);
}

}  // extern "C"
