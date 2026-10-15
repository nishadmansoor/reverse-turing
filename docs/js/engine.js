/**
 * Inference engine — the browser's replacement for app/app.py
 * ===========================================================
 * Runs the same int8 ONNX graphs the server ran, through ONNX Runtime Web
 * (WASM). Nothing leaves the browser: the text being classified is never sent
 * anywhere.
 *
 * The transformer is ~64 MB, so it is fetched lazily on first use with
 * progress reported to the caller, rather than blocking the page load.
 */

import {
  extractFeatures, textToHeatmap, logisticAiProb,
  metaFeatures, softmax, setParams,
} from "./features.js";

const MODEL_BASE = "./models";
const MAX_TOKENS = 256;

let params = null;
let tokenizer = null;
let bertSession = null;
let cnnSession = null;
let loadPromise = null;

export async function loadParams() {
  if (params) return params;
  params = await fetch("./data/params.json").then((r) => r.json());
  setParams(params);
  return params;
}

/** Fetch with byte-level progress, so a 64 MB download isn't a blank wait. */
async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);

  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received, total);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out;
}

/**
 * Load the models. Safe to call repeatedly — concurrent callers share one
 * in-flight load rather than each starting their own 64 MB download.
 */
export function ensureModels(onStatus) {
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    await loadParams();

    onStatus?.({ phase: "runtime", message: "Starting inference runtime" });
    const ort = window.ort;
    if (!ort) throw new Error("ONNX Runtime failed to load");
    ort.env.wasm.numThreads = 1;      // keep it predictable across browsers
    ort.env.wasm.simd = true;

    onStatus?.({ phase: "tokenizer", message: "Loading tokenizer" });
    const { AutoTokenizer, env } = await import(
      "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2/dist/transformers.min.js"
    );
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = MODEL_BASE;
    tokenizer = await AutoTokenizer.from_pretrained("tokenizer");

    onStatus?.({ phase: "cnn", message: "Loading visual model" });
    const cnnBytes = await fetchWithProgress(`${MODEL_BASE}/cnn_int8.onnx`);
    cnnSession = await ort.InferenceSession.create(cnnBytes, {
      executionProviders: ["wasm"],
    });

    onStatus?.({ phase: "transformer", message: "Loading language model" });
    const bertBytes = await fetchWithProgress(
      `${MODEL_BASE}/transformer_int8.onnx`,
      (received, total) => onStatus?.({
        phase: "transformer",
        message: "Loading language model",
        received, total,
      }),
    );
    bertSession = await ort.InferenceSession.create(bertBytes, {
      executionProviders: ["wasm"],
    });

    onStatus?.({ phase: "ready", message: "Ready" });
  })().catch((err) => {
    loadPromise = null;          // let the caller retry after a failure
    throw err;
  });

  return loadPromise;
}

async function predictTransformer(text) {
  const ort = window.ort;
  const encoded = await tokenizer(text, {
    padding: "max_length",
    truncation: true,
    max_length: MAX_TOKENS,
  });

  // transformers.js hands back tensors whose .data is already a BigInt64Array
  // shaped [1, MAX_TOKENS] — exactly what the int64 graph inputs want, so
  // they pass straight through without conversion.
  const dims = [1, MAX_TOKENS];
  const output = await bertSession.run({
    input_ids: new ort.Tensor("int64", encoded.input_ids.data, dims),
    attention_mask: new ort.Tensor("int64", encoded.attention_mask.data, dims),
  });
  const logits = Array.from(output[bertSession.outputNames[0]].data);
  return softmax(logits)[1];
}

async function predictCnn(text) {
  const ort = window.ort;
  const size = params.heatmap.size;
  const input = textToHeatmap(text);
  const output = await cnnSession.run({
    images: new ort.Tensor("float32", input, [1, 3, size, size]),
  });
  const logits = Array.from(output[cnnSession.outputNames[0]].data);
  return softmax(logits)[1];
}

/** Full pipeline over one text — mirrors classify() in app/app.py. */
export async function classify(text) {
  await ensureModels();

  const features = extractFeatures(text);
  const styloAi = logisticAiProb(features, params.stylometric);
  const [bertAi, cnnAi] = await Promise.all([
    predictTransformer(text),
    predictCnn(text),
  ]);

  const ensembleAi = logisticAiProb(
    metaFeatures(bertAi, styloAi, cnnAi, text), params.meta
  );

  const verdict = ensembleAi >= 0.5 ? "AI" : "Human";
  const label = (p) => (p >= 0.5 ? "AI" : "Human");
  const pct = (p) => Math.round(p * 1000) / 10;
  const conf = (p) => pct(Math.max(p, 1 - p));

  const names = ["BERT", "Stylometric", "CNN"];
  const preds = [bertAi, styloAi, cnnAi].map((p) => label(p));

  return {
    text,
    features,
    word_count: text.trim().split(/\s+/).filter(Boolean).length,
    bert: label(bertAi), bert_ai_prob: pct(bertAi), bert_confidence: conf(bertAi),
    stylometric: label(styloAi), stylo_ai_prob: pct(styloAi), stylo_confidence: conf(styloAi),
    opencv: label(cnnAi), opencv_ai_prob: pct(cnnAi), opencv_confidence: conf(cnnAi),
    ensemble: verdict,
    ensemble_ai_prob: pct(ensembleAi),
    ensemble_confidence: conf(ensembleAi),
    models_agree: new Set(preds).size === 1,
    dissenters: names.filter((_, i) => preds[i] !== verdict),
  };
}
