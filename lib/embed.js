// Session 9: embeddings for a one-file browser app.
//
//   import { embed, similarity, createIndex, groups } from
//     'https://cdn.jsdelivr.net/gh/ccmpm/session-9@main/lib/embed.js';
//
// Vectors come from embeddings.ccm.pm, which forwards to OpenAI's
// text-embedding-3-small with ccm.pm's key, so the app needs no key of its
// own. Every vector is kept in the browser (IndexedDB), so a reload does not
// embed the same text twice.

export const ENDPOINT = 'https://embeddings.ccm.pm/v1/embeddings';
export const MODEL = 'text-embedding-3-small';
export const DIMENSIONS = 512;

const BATCH_ITEMS = 256;          // texts per request
const BATCH_CHARS = 100_000;      // characters per request
const MAX_TEXT_CHARS = 24_000;    // the model reads about 8,000 tokens of one text; longer text is cut

// ---- cache: IndexedDB when the browser allows it, memory otherwise --------

const memory = new Map();
let dbPromise = null;
function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    try {
      const req = indexedDB.open('ccm-embed', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('vectors');
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch { resolve(null); }
  });
  return dbPromise;
}
async function cacheGet(keys) {
  const out = keys.map((k) => memory.get(k));
  const db = await openDb();
  if (!db) return out;
  await new Promise((resolve) => {
    try {
      const store = db.transaction('vectors').objectStore('vectors');
      let left = 0;
      keys.forEach((k, i) => {
        if (out[i]) return;
        left++;
        const r = store.get(k);
        r.onsuccess = () => { if (r.result) { out[i] = new Float32Array(r.result); memory.set(k, out[i]); } if (--left === 0) resolve(); };
        r.onerror = () => { if (--left === 0) resolve(); };
      });
      if (left === 0) resolve();
    } catch { resolve(); }
  });
  return out;
}
async function cachePut(entries) {
  for (const [k, v] of entries) memory.set(k, v);
  const db = await openDb();
  if (!db) return;
  try {
    const store = db.transaction('vectors', 'readwrite').objectStore('vectors');
    for (const [k, v] of entries) store.put(v.buffer, k);
  } catch { /* memory cache still holds them */ }
}

// ---- embed ----------------------------------------------------------------

async function request(texts, opts) {
  for (let attempt = 0; ; attempt++) {
    let res;
    try {
      res = await fetch(opts.endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: opts.model, input: texts, dimensions: opts.dimensions }),
      });
    } catch (e) {
      if (attempt < 3) { await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt)); continue; }
      throw new Error(`embed: cannot reach ${opts.endpoint} (${e.message})`);
    }
    if ((res.status === 429 || res.status >= 500) && attempt < 3) {
      await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
      continue;
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(`embed: ${res.status} ${body.error?.message ?? res.statusText}`);
    return body.data.sort((a, b) => a.index - b.index).map((d) => normalize(Float32Array.from(d.embedding)));
  }
}

/**
 * Turn text into vectors. A string gives one vector, an array gives an array
 * of vectors in the same order. Each vector is a Float32Array of length 512,
 * scaled to length 1.
 * options: { onProgress(done, total), dimensions, model, endpoint }
 */
export async function embed(input, options = {}) {
  const single = typeof input === 'string';
  const texts = (single ? [input] : input).map((t) => {
    const s = String(t ?? '').trim();
    return (s || ' ').slice(0, MAX_TEXT_CHARS);
  });
  const opts = { endpoint: ENDPOINT, model: MODEL, dimensions: DIMENSIONS, ...options };
  const key = (t) => `${opts.model}|${opts.dimensions}|${t}`;
  const vectors = await cacheGet(texts.map(key));

  const missing = [];
  const seen = new Map();
  texts.forEach((t, i) => {
    if (vectors[i]) return;
    if (seen.has(t)) seen.get(t).push(i);
    else { seen.set(t, [i]); missing.push(t); }
  });

  let done = texts.length - [...seen.values()].reduce((n, ix) => n + ix.length, 0);
  opts.onProgress?.(done, texts.length);
  let batch = [], chars = 0;
  const flush = async () => {
    if (!batch.length) return;
    const got = await request(batch, opts);
    const entries = [];
    batch.forEach((t, j) => {
      entries.push([key(t), got[j]]);
      for (const i of seen.get(t)) { vectors[i] = got[j]; done++; }
    });
    await cachePut(entries);
    opts.onProgress?.(done, texts.length);
    batch = []; chars = 0;
  };
  for (const t of missing) {
    if (batch.length >= BATCH_ITEMS || chars + t.length > BATCH_CHARS) await flush();
    batch.push(t); chars += t.length;
  }
  await flush();
  return single ? vectors[0] : vectors;
}

// ---- math -----------------------------------------------------------------

function normalize(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

/** How alike two vectors are: 1 is the same meaning, around 0 is unrelated. */
export function similarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na * nb) || 1);
}

/**
 * Embed a list of items once, then search it by meaning.
 *   const index = await createIndex(items, { text: item => item.title + '. ' + item.body });
 *   index.search('kids at the beach', 5)   // [{ item, score }], best first
 *   index.similar(items[3], 5)             // items closest to one item, itself left out
 *   index.vectors[i]                       // the vector of items[i]
 * options: { text(item), onProgress(done, total) }
 */
export async function createIndex(items, options = {}) {
  const text = options.text ?? ((x) => (typeof x === 'string' ? x : JSON.stringify(x)));
  const vectors = await embed(items.map(text), { onProgress: options.onProgress });
  const rank = (q, k, skip = -1) => items
    .map((item, i) => ({ item, score: i === skip ? -Infinity : similarity(q, vectors[i]) }))
    .filter((r) => r.score > -Infinity)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
  return {
    items,
    vectors,
    async search(query, k = 5) {
      const q = typeof query === 'string' ? await embed(query) : query;
      return rank(q, k);
    },
    similar(item, k = 5) {
      const i = items.indexOf(item);
      if (i < 0) throw new Error('similar: item is not in this index');
      return rank(vectors[i], k, i);
    },
  };
}

/**
 * Sort vectors into k groups of similar meaning (k-means, cosine).
 * Returns an array of k arrays of indexes into `vectors`, biggest group first.
 * Same input gives the same groups.
 */
export function groups(vectors, k, rounds = 25) {
  const n = vectors.length;
  if (!n) return [];
  k = Math.max(1, Math.min(k, n));
  let seed = 1;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  // k-means++ start: each new centre far from the ones already picked
  const centres = [vectors[Math.floor(random() * n)]];
  const dist = new Float64Array(n).fill(Infinity);
  while (centres.length < k) {
    const last = centres[centres.length - 1];
    let total = 0;
    for (let i = 0; i < n; i++) { dist[i] = Math.min(dist[i], 1 - similarity(vectors[i], last)); total += dist[i]; }
    let r = random() * total, pick = n - 1;
    for (let i = 0; i < n; i++) { r -= dist[i]; if (r <= 0) { pick = i; break; } }
    centres.push(vectors[pick]);
  }
  const owner = new Int32Array(n).fill(-1);
  for (let round = 0; round < rounds; round++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bestScore = -Infinity;
      for (let c = 0; c < k; c++) { const s = similarity(vectors[i], centres[c]); if (s > bestScore) { bestScore = s; best = c; } }
      if (owner[i] !== best) { owner[i] = best; moved = true; }
    }
    if (!moved) break;
    for (let c = 0; c < k; c++) {
      const sum = new Float32Array(vectors[0].length);
      let count = 0;
      for (let i = 0; i < n; i++) if (owner[i] === c) { count++; for (let d = 0; d < sum.length; d++) sum[d] += vectors[i][d]; }
      if (count) centres[c] = normalize(sum);
    }
  }
  const out = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) out[owner[i]].push(i);
  return out.filter((g) => g.length).sort((a, b) => b.length - a.length);
}
