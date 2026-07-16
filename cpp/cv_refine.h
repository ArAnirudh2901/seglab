/**
 * cv_refine — compact mask-hygiene kernels for SEGLAB (wasm32, SIMD build).
 *
 * Contract:
 *  - masks are one-channel uint8, width*height bytes, fg = value >= 128;
 *    outputs are strictly 0/255;
 *  - max(width, height) <= 768 — larger inputs are rejected, never clipped;
 *  - all workspaces are static (no per-call allocation; ALLOW_MEMORY_GROWTH=0);
 *  - every function returns >= 0 on success or a CV_ERR_* code; nothing
 *    throws, nothing allocates.
 *
 * Algorithms are direct ports of the proven JS pipeline (js/sam-core.js
 * cleanupMaskAlpha, js/edge-refine.js morph) so wasm and JS fallback agree.
 */
#ifndef SEGLAB_CV_REFINE_H
#define SEGLAB_CV_REFINE_H

#include <stdint.h>

#define CV_ERR_DIMS (-1)
#define CV_ERR_ARGS (-2)

#ifdef __cplusplus
extern "C" {
#endif

/* Keep the component at/near (seed_x, seed_y) plus any component whose area
 * is >= max(min_area, 1% of the largest kept) — occluder-split survival.
 * Returns the number of kept components. */
int seeded_component_cleanup(uint8_t* mask, int width, int height,
                             int seed_x, int seed_y, int min_area);

/* Multi-seed variant (real click sets); seeds = [x0,y0, x1,y1, ...]. */
int seeded_cleanup_multi(uint8_t* mask, int width, int height,
                         const int32_t* seeds, int n_seeds, int min_area);

/* Fill interior holes <= max(64, 1% of foreground). Returns filled pixels. */
int fill_holes(uint8_t* mask, int width, int height);

/* Binary morphology, separable Chebyshev (square) kernel. `scratch` must
 * hold width*height bytes. Return 0 on success. */
int morphology_open(uint8_t* mask, uint8_t* scratch, int width, int height,
                    int kernel_radius);
int morphology_close(uint8_t* mask, uint8_t* scratch, int width, int height,
                     int kernel_radius);

/* Full hygiene pass: cleanup -> holes -> close -> open. `rgb` (w*h*3) is
 * accepted for future edge refinement and ignored today (may be null).
 * Returns kept-component count. */
int refine_mask(const uint8_t* rgb, uint8_t* mask, uint8_t* scratch,
                int width, int height, int seed_x, int seed_y,
                int min_area, int open_radius, int close_radius);

/* refine_mask with a seed list — the entry point the worker uses. */
int refine_mask_multi(uint8_t* mask, uint8_t* scratch, int width, int height,
                      const int32_t* seeds, int n_seeds,
                      int min_area, int open_radius, int close_radius);

#ifdef __cplusplus
}
#endif

#endif /* SEGLAB_CV_REFINE_H */
