// cv_refine — bounded binary-mask cleanup for the ≤1024px interaction proxy.
// One-channel uint8 masks (0..255, ≥128 = on). Fixed reusable workspace, no
// growth; every entry point rejects sides above CV_MAX_SIDE.
#pragma once
#include <cstdint>

#define CV_MAX_SIDE 1024

extern "C" {

// Keep the component containing (seed_x, seed_y); other components survive
// only when their area is >= min_area. With a seed and min_area <= 0, ONLY
// the seeded component survives. A negative seed skips seeding: min_area
// alone filters small objects (min_area <= 0 is then a no-op).
// Dropped pixels are zeroed; kept pixels retain their original (soft) values.
void seeded_component_cleanup(
    uint8_t* mask, int width, int height, int seed_x, int seed_y, int min_area);

// Fill background regions not connected to the image border (holes) with 255.
void fill_holes(uint8_t* mask, int width, int height);

// Binary morphology, square structuring element of the given radius.
// Output is binary (0/255). `scratch` must be width*height bytes.
void morphology_open(uint8_t* mask, uint8_t* scratch, int width, int height, int kernel_radius);
void morphology_close(uint8_t* mask, uint8_t* scratch, int width, int height, int kernel_radius);

// Full pipeline: seeded cleanup -> open -> close -> hole fill. `rgb` is
// reserved for future edge-guided refinement and may be null.
void refine_mask(
    const uint8_t* rgb, uint8_t* mask, uint8_t* scratch,
    int width, int height, int seed_x, int seed_y,
    int min_area, int open_radius, int close_radius);

}
