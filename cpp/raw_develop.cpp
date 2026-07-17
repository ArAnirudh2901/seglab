// raw_develop — bounded RAW→JPEG develop for the rare camera RAW that carries
// NO usable embedded JPEG preview (image-raw.js handles the common case by
// lifting the camera's own preview; this is the fallback when there is none).
//
// LibRaw demosaics the sensor — half-resolution by default, which a ≤1024px
// masking proxy never exceeds — and libjpeg re-encodes the result IN THE WORKER,
// so only a compact JPEG Blob crosses back to the main thread. A full-res RGBA
// frame never leaves this heap (blob-only residency), and the caller disposes
// the worker after each develop so LibRaw's big sensor buffers are reclaimed.
//
// One job at a time. Every entry point returns an error code, never throws
// across the ABI; the megapixel cap is checked BEFORE unpack (where LibRaw
// allocates the full sensor buffer) so the peak stays bounded and predictable.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>

#include "libraw/libraw.h"
#include <jpeglib.h>

namespace {

LibRaw* g_raw = nullptr;
unsigned char* g_jpeg = nullptr;   // libjpeg-owned (malloc); freed in rd_release
unsigned long g_jpeg_len = 0;
int g_width = 0;
int g_height = 0;
char g_err[192] = {0};

void set_err(const char* msg) {
    std::strncpy(g_err, msg && *msg ? msg : "unknown", sizeof(g_err) - 1);
    g_err[sizeof(g_err) - 1] = 0;
}

void drop_raw() {
    if (g_raw) { g_raw->recycle(); delete g_raw; g_raw = nullptr; }
}

} // namespace

extern "C" {

const char* rd_error() { return g_err; }
int rd_width() { return g_width; }
int rd_height() { return g_height; }
const unsigned char* rd_jpeg() { return g_jpeg; }
int rd_jpeg_len() { return static_cast<int>(g_jpeg_len); }

// Free the previous develop's JPEG + LibRaw state. Safe to call repeatedly.
void rd_release() {
    if (g_jpeg) { free(g_jpeg); g_jpeg = nullptr; }
    g_jpeg_len = 0;
    g_width = 0;
    g_height = 0;
    drop_raw();
}

// Develop RAW bytes [data, data+len) into an in-memory JPEG.
//   half    : 1 = half-resolution demosaic (the masking proxy target); 0 = full
//   max_mp  : refuse sensors above this many megapixels (bounds the peak); ≤0 = no cap
//   quality : JPEG quality 1..100 (≤0 → 90)
// Returns 0 on success — then read rd_jpeg()/rd_jpeg_len()/rd_width()/rd_height().
// Non-zero on failure — read rd_error(). Always call rd_release() afterwards.
int rd_develop(const unsigned char* data, int len, int half, int max_mp, int quality) {
    rd_release();
    g_err[0] = 0;
    if (!data || len <= 0) { set_err("empty input"); return 1; }

    g_raw = new (std::nothrow) LibRaw();
    if (!g_raw) { set_err("libraw alloc failed"); return 2; }

    try {
        int rc = g_raw->open_buffer(const_cast<unsigned char*>(data), static_cast<size_t>(len));
        if (rc != LIBRAW_SUCCESS) { set_err(libraw_strerror(rc)); drop_raw(); return 3; }

        // Bound the peak BEFORE unpack, which allocates the full sensor buffer.
        const double mp = static_cast<double>(g_raw->imgdata.sizes.raw_width)
                        * g_raw->imgdata.sizes.raw_height / 1.0e6;
        if (max_mp > 0 && mp > max_mp) { set_err("sensor exceeds megapixel cap"); drop_raw(); return 4; }

        rc = g_raw->unpack();
        if (rc != LIBRAW_SUCCESS) { set_err(libraw_strerror(rc)); drop_raw(); return 5; }

        // Neutral, low-memory develop suited to a select/cutout proxy: sRGB,
        // camera white balance, 8-bit out, cheap bilinear demosaic at half size.
        libraw_output_params_t& p = g_raw->imgdata.params;
        p.output_bps = 8;
        p.output_color = 1;              // sRGB
        p.use_camera_wb = 1;             // the camera's own white balance
        p.no_auto_bright = 0;            // a viewable exposure, like the preview
        p.half_size = half ? 1 : 0;
        p.user_qual = half ? 0 : 3;      // linear when half; AHD at full res

        rc = g_raw->dcraw_process();
        if (rc != LIBRAW_SUCCESS) { set_err(libraw_strerror(rc)); drop_raw(); return 6; }

        int errc = 0;
        libraw_processed_image_t* img = g_raw->dcraw_make_mem_image(&errc);
        if (!img || errc != 0 || img->type != LIBRAW_IMAGE_BITMAP
                || img->colors != 3 || img->bits != 8) {
            if (img) LibRaw::dcraw_clear_mem(img);
            set_err("develop produced no 8-bit RGB");
            drop_raw();
            return 7;
        }
        g_width = static_cast<int>(img->width);
        g_height = static_cast<int>(img->height);

        // Encode the interleaved RGB8 to JPEG in memory (libjpeg-turbo).
        jpeg_compress_struct cinfo;
        jpeg_error_mgr jerr;
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_mem_dest(&cinfo, &g_jpeg, &g_jpeg_len);
        cinfo.image_width = img->width;
        cinfo.image_height = img->height;
        cinfo.input_components = 3;
        cinfo.in_color_space = JCS_RGB;
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, quality > 0 ? quality : 90, TRUE);
        jpeg_start_compress(&cinfo, TRUE);
        const int stride = static_cast<int>(img->width) * 3;
        while (cinfo.next_scanline < cinfo.image_height) {
            JSAMPROW row = reinterpret_cast<JSAMPROW>(img->data + cinfo.next_scanline * stride);
            jpeg_write_scanlines(&cinfo, &row, 1);
        }
        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);
        LibRaw::dcraw_clear_mem(img);

        // Recycle LibRaw's big buffers immediately; keep only the JPEG.
        drop_raw();
        if (!g_jpeg || g_jpeg_len == 0) { set_err("jpeg encode produced nothing"); return 8; }
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        drop_raw();
        return 9;
    } catch (...) {
        set_err("unhandled develop exception");
        drop_raw();
        return 9;
    }
}

} // extern "C"
