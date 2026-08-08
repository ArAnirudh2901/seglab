/* Stand-in for @huggingface/transformers implementing exactly the surface
 * sam-engine.js uses. The decoder is a real flood-fill segmenter over the
 * actual canvas pixels, so mask assertions stay meaningful. Everything else
 * in the app under test is the production code. */
export class RawImage {
  constructor(data, width, height, channels) {
    this.data = data; this.width = width; this.height = height; this.channels = channels
  }
}
export class Tensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims }
}

const floodFrom = (px, w, h, sx, sy, tol = 30) => {
  const out = new Uint8Array(w * h)
  const x0 = Math.round(sx), y0 = Math.round(sy)
  if (x0 < 0 || y0 < 0 || x0 >= w || y0 >= h) return out
  const seed = (y0 * w + x0) * 4
  const sr = px[seed], sg = px[seed + 1], sb = px[seed + 2]
  const stack = new Int32Array(w * h); let top = 0
  stack[top++] = y0 * w + x0; out[y0 * w + x0] = 1
  const near = (i) => {
    const j = i * 4
    return Math.abs(px[j] - sr) <= tol && Math.abs(px[j+1] - sg) <= tol && Math.abs(px[j+2] - sb) <= tol
  }
  while (top > 0) {
    const i = stack[--top]; const x = i % w
    const push = (n) => { if (!out[n] && near(n)) { out[n] = 1; stack[top++] = n } }
    if (x > 0) push(i - 1)
    if (x < w - 1) push(i + 1)
    if (i >= w) push(i - w)
    if (i < w * (h - 1)) push(i + w)
  }
  return out
}

const makeProcessor = () => {
  const proc = async (image) => {
    const scale = 1024 / Math.max(image.width, image.height)
    return {
      // carry the real pixels through so the "encoder" can cache them
      pixel_values: new Tensor('uint8', image.data, [1, 4, image.height, image.width]),
      original_sizes: [[image.height, image.width]],
      reshaped_input_sizes: [[Math.round(image.height * scale), Math.round(image.width * scale)]],
    }
  }
  // masks already come back at original size
  proc.post_process_masks = async (pred) => [pred]
  return proc
}

const makeModel = () => {
  const model = async (inputs) => {
    const emb = inputs.image_embeddings
    const { data: px, w, h } = emb
    const pts = inputs.input_points.data
    const labels = inputs.input_labels.data
    const [, , rh, rw] = [0, 0, emb.reshaped[0], emb.reshaped[1]]
    const n = pts.length / 2
    const size = w * h
    const flood = new Uint8Array(size)
    // positives union
    for (let i = 0; i < n; i++) {
      if (Number(labels[i]) !== 1) continue
      const sx = (pts[i * 2] * w) / rw, sy = (pts[i * 2 + 1] * h) / rh
      const f = floodFrom(px, w, h, sx, sy)
      for (let k = 0; k < size; k++) if (f[k]) flood[k] = 1
    }
    // negatives subtract
    for (let i = 0; i < n; i++) {
      if (Number(labels[i]) !== 0) continue
      const sx = (pts[i * 2] * w) / rw, sy = (pts[i * 2 + 1] * h) / rh
      const f = floodFrom(px, w, h, sx, sy)
      for (let k = 0; k < size; k++) if (f[k]) flood[k] = 0
    }
    // three candidates; channel 1 is the good one and scores highest, so a
    // broken pickBestMask shows up immediately
    const data = new Uint8Array(size * 3)
    for (let k = 0; k < size; k++) {
      data[k] = (k % 97 === 0) ? 1 : 0          // ch0: sparse noise
      data[size + k] = flood[k]                  // ch1: the real answer
      data[size * 2 + k] = 1                     // ch2: solid frame
    }
    return {
      pred_masks: new Tensor('uint8', data, [1, 3, h, w]),
      iou_scores: new Tensor('float32', new Float32Array([0.21, 0.93, 0.44]), [1, 1, 3]),
    }
  }
  model.get_image_embeddings = async ({ pixel_values }) => ({
    image_embeddings: {
      data: pixel_values.data,
      w: pixel_values.dims[3],
      h: pixel_values.dims[2],
      reshaped: [Math.round(pixel_values.dims[2] * (1024 / Math.max(pixel_values.dims[3], pixel_values.dims[2]))),
                 Math.round(pixel_values.dims[3] * (1024 / Math.max(pixel_values.dims[3], pixel_values.dims[2])))],
    },
  })
  return model
}

const fakeDownload = (progress_callback, file, total) => {
  if (!progress_callback) return
  progress_callback({ status: 'initiate', file })
  for (let p = 0; p <= 100; p += 25) {
    progress_callback({ status: 'progress', file, loaded: (total * p) / 100, total, progress: p })
  }
  progress_callback({ status: 'done', file })
}

export const SamModel = {
  from_pretrained: async (name, opts = {}) => {
    fakeDownload(opts.progress_callback, 'onnx/model_quantized.onnx', 14_000_000)
    return makeModel()
  },
}
export const Sam3TrackerModel = {
  from_pretrained: async () => { throw new Error('stub: flagship not available') },
}
export const AutoProcessor = {
  from_pretrained: async (name, opts = {}) => {
    fakeDownload(opts.progress_callback, 'preprocessor_config.json', 500)
    return makeProcessor()
  },
}
