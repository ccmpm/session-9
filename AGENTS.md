# Session 9 embeddings app

Rules for the coding agent that builds a session 9 app: one file, `index.html`, that works when opened by double-clicking it.

## Embeddings are already built

Use `lib/embed.js` for every embedding. Import it from `https://cdn.jsdelivr.net/gh/ccmpm/session-9@main/lib/embed.js`; do not copy it into the project. It gets vectors from `https://embeddings.ccm.pm`, which runs OpenAI's `text-embedding-3-small` on ccm.pm's account. The app needs no API key.

- You must not ask for, store or send an API key. There is none to add.
- You must not call api.openai.com or any other embeddings service yourself.
- You must not add a server, a database, a build step or npm.
- You must not change `lib/embed.js` or write your own copy of it.

```js
import { embed, similarity, createIndex, groups } from 'https://cdn.jsdelivr.net/gh/ccmpm/session-9@main/lib/embed.js';

// Search a list by meaning: the usual starting point
const index = await createIndex(items, {
  text: item => item.title + '. ' + item.description,   // the words that carry the item's meaning
  onProgress: (done, total) => {},                       // show progress: a few hundred items take a few seconds
});
await index.search('kids at the beach', 5)   // [{ item, score }], best first. score: 1 = same meaning
index.similar(items[3], 5)                   // the items closest to one item, without the item itself
index.vectors[i]                             // the vector of items[i]

// Lower level
const v = await embed('some text')           // one Float32Array of 512 numbers
const vs = await embed(['a', 'b', 'c'])      // an array of them, same order
similarity(v, vs[0])                         // 1 = same meaning, around 0 = unrelated

// Sort items into topics on their own
groups(index.vectors, 5)                     // [[i, j, ...], ...] indexes into items, biggest group first
```

`https://raw.githubusercontent.com/ccmpm/session-9/main/example.html` is a small working one-file example: search a list of emails by meaning, and sort them into topics.

## Facts about the vectors

- Every vector is kept in the browser. A reload does not embed the same text again, so the second load is fast.
- Scores are relative. Between unrelated texts they are around 0.1 to 0.3, and close matches are often 0.5 to 0.7, not 1. Rank by score; if you need a cut-off, pick it by trying real queries, and show the score in the page while building.
- Embed the words a user would search for. For a product, the name and description, not the price or the id. Numbers and ids carry no meaning to the model.
- Text only. To search photos or songs, embed a description of each one: a caption, the title, the artist, the genre, the mood.
- One text can be up to about 20,000 characters. For long documents, split them into paragraphs and embed each paragraph as its own item, keeping a link to its document.
- Embed the dataset once at load with `createIndex`, then embed only the query on each search. Wait 300 ms after the last key press before searching.

## The data

- The app opens from disk (`file://`), so it cannot `fetch` a file next to it. Put the data in the page as a JavaScript array, or let the user pick a file with `<input type="file">` (CSV or JSON) and read it with `FileReader`, or fetch a public https URL that allows it.
- When the team has no data yet, write a realistic sample of 50 to 200 items into the page. Use English names.

## Running and testing

- One file, `index.html`, plain HTML with one `<script type="module">`. Everything it needs comes from `https://cdn.jsdelivr.net`.
- It must work opened from disk (`file://`): import only from https URLs, never from a local path.
- Put `window.index = index` so the console can inspect it. Example: `await index.search('boss')`.
- Show an error on the page when an embedding call fails, with its message. A 403 means embeddings.ccm.pm is closed: it is open on session day only.
