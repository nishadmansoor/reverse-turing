/**
 * Feature extraction — JavaScript port of src/features.py
 * =======================================================
 * The browser build has to reproduce the Python pipeline exactly. A model is
 * only as good as the inputs it is handed, and a port that is subtly different
 * degrades predictions without ever throwing an error.
 *
 * Every function here is checked against Python-generated expected values by
 * web/parity.html. Change anything in this file and run that page.
 *
 * Python behaviours worth calling out, because the obvious JS equivalent is
 * wrong for each of them:
 *
 *   str.split()    — splits on runs of whitespace AND drops empty strings.
 *                    "  a  b ".split() is ["a","b"]; the JS equivalent of
 *                    .split(/\s+/) would yield ["","a","b",""].
 *   np.var         — population variance (divide by N), not the sample
 *                    variance (N-1) that most libraries default to.
 *   string.punctuation — a fixed 32-character ASCII set, not a unicode
 *                    category. Loaded from params.json rather than retyped.
 */

let PARAMS = null;

export function setParams(params) {
  PARAMS = params;
}

/** Python's str.split() with no argument. */
export function pySplit(text) {
  const out = [];
  for (const part of text.split(/\s+/)) {
    if (part.length > 0) out.push(part);
  }
  return out;
}

/** Population variance, matching np.var's default ddof=0. */
function populationVariance(values) {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length;
}

function mean(values) {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function countPunctuation(text) {
  const punct = PARAMS.punctuation;
  let n = 0;
  for (const ch of text) if (punct.includes(ch)) n += 1;
  return n;
}

/**
 * The five stylometric features, in the fixed order the models expect.
 * Returns zeros for unusable input, exactly as the Python version does.
 */
export function extractFeatures(text) {
  const sentences = text.split(".")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const words = pySplit(text.toLowerCase());

  if (sentences.length === 0 || words.length === 0) {
    return [0, 0, 0, 0, 0];
  }

  const sentenceLengths = sentences.map((s) => pySplit(s).length);
  const unique = new Set(words);

  return [
    mean(sentenceLengths),
    populationVariance(sentenceLengths),
    mean(words.map((w) => w.length)),
    unique.size / words.length,
    countPunctuation(text) / text.length,
  ];
}

/** One grid row per sentence — mirrors _sentence_row in features.py. */
function sentenceRow(sentence) {
  const words = pySplit(sentence);
  const nWords = words.length;
  const nChars = Math.max(sentence.length, 1);

  let digits = 0;
  let uppers = 0;
  for (const ch of sentence) {
    if (ch >= "0" && ch <= "9") digits += 1;
    // Python's str.isupper() is true only for cased characters that are
    // uppercase, so digits and punctuation must not count here.
    if (ch !== ch.toLowerCase() && ch === ch.toUpperCase()) uppers += 1;
  }

  const lowerUnique = new Set(words.map((w) => w.toLowerCase()));

  return [
    sentence.length,
    nWords,
    nWords ? mean(words.map((w) => w.length)) : 0,
    countPunctuation(sentence),
    (sentence.match(/,/g) || []).length,
    digits,
    uppers / nChars,
    nWords ? lowerUnique.size / nWords : 0,
  ];
}

/**
 * Render text as the 224x224x3 BGR heatmap the CNN was trained on.
 *
 * Returns a Float32Array in NCHW order, already divided by 255 — the exact
 * tensor the ONNX graph expects.
 */
export function textToHeatmap(text) {
  const { rows: gridRows, size, scales, jet_bgr: jet } = PARAMS.heatmap;
  const nCols = scales.length;

  let rows = text.split(".")
    .filter((s) => s.trim().length > 0)
    .map(sentenceRow);
  if (rows.length === 0) rows = [new Array(nCols).fill(0)];
  rows = rows.slice(0, gridRows);

  // Normalise against fixed ranges, then zero-pad to a full grid.
  //
  // Each step is rounded to float32 with Math.fround because numpy does this
  // arithmetic on a float32 array. Computing in float64 and truncating at the
  // end lands on a different integer whenever a value sits just under a whole
  // number — which showed up as a handful of off-by-one pixels in the parity
  // fixture.
  const grid = new Uint8Array(gridRows * nCols);
  for (let r = 0; r < rows.length; r++) {
    for (let c = 0; c < nCols; c++) {
      const value = Math.fround(rows[r][c]);
      let scaled = Math.fround(value / Math.fround(scales[c]));
      scaled = Math.min(Math.max(scaled, 0), 1);
      scaled = Math.fround(scaled * 255);
      // numpy's astype(uint8) truncates toward zero.
      grid[r * nCols + c] = Math.trunc(scaled);
    }
  }

  // cv2.resize(..., INTER_NEAREST): source index is floor(dst * src/dst).
  const scaleY = gridRows / size;
  const scaleX = nCols / size;
  const out = new Float32Array(3 * size * size);
  const plane = size * size;

  for (let y = 0; y < size; y++) {
    const sy = Math.min(Math.floor(y * scaleY), gridRows - 1);
    for (let x = 0; x < size; x++) {
      const sx = Math.min(Math.floor(x * scaleX), nCols - 1);
      const value = grid[sy * nCols + sx];
      const bgr = jet[value];
      const px = y * size + x;
      // NCHW, channels in BGR order as cv2.applyColorMap produces them.
      out[px] = bgr[0] / 255;
      out[plane + px] = bgr[1] / 255;
      out[2 * plane + px] = bgr[2] / 255;
    }
  }
  return out;
}

/** Checksum helpers used only by the parity page. */
export function heatmapDebug(text) {
  const { size, jet_bgr: jet } = PARAMS.heatmap;
  const chw = textToHeatmap(text);
  const plane = size * size;
  let sum = 0;
  for (let i = 0; i < chw.length; i++) sum += Math.round(chw[i] * 255);
  const at = (r, c, ch) => Math.round(chw[ch * plane + r * size + c] * 255);
  return {
    sum,
    samples: [at(0, 0, 0), at(0, 0, 2), at(50, 50, 1), at(112, 112, 0), at(223, 223, 2)],
  };
}

/** Logistic regression: sigmoid(x · coef + intercept). */
export function logisticAiProb(features, model) {
  let z = model.intercept;
  for (let i = 0; i < features.length; i++) z += features[i] * model.coef[i];
  return 1 / (1 + Math.exp(-z));
}

/** The five meta-classifier inputs, in the order the model was fitted on. */
export function metaFeatures(bertAiProb, styloAiProb, cnnAiProb, text) {
  const wordCount = pySplit(text).length;
  const sentenceCount = text.split(".").filter((s) => s.trim().length > 0).length;
  return [bertAiProb, styloAiProb, cnnAiProb, wordCount, sentenceCount];
}

/** StandardScaler followed by PCA, for the Dataset Explorer overlay. */
export function pcaTransform(features) {
  const p = PARAMS.pca;
  const scaled = features.map((v, i) => (v - p.scaler_mean[i]) / p.scaler_scale[i]);
  const centered = scaled.map((v, i) => v - p.pca_mean[i]);
  return p.components.map((component) =>
    component.reduce((acc, w, i) => acc + w * centered[i], 0)
  );
}

export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}
