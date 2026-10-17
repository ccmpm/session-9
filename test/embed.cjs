// Checks lib/embed.js and example.html in a real browser, opened from disk
// (file://) the way teams open their app, against the live embeddings.ccm.pm.
// The jsDelivr import is served from this folder, so a change is tested before it is pushed.
// Run: npm test   (first time: npm install && npx playwright install chromium)
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const root = path.join(__dirname, '..');
const CDN = 'https://cdn.jsdelivr.net/gh/ccmpm/session-9@main/lib/embed.js';

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext();
  let calls = 0;
  await context.route(CDN, (r) => r.fulfill({ contentType: 'text/javascript', body: fs.readFileSync(path.join(root, 'lib/embed.js')) }));
  context.on('request', (r) => { if (r.url().startsWith('https://embeddings.ccm.pm')) calls++; });
  const errors = [];
  const results = [];
  const check = (name, ok) => { results.push([name, !!ok]); console.log((ok ? 'pass  ' : 'FAIL  ') + name); };

  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto('file://' + path.join(root, 'example.html'));
  await page.waitForFunction(() => window.index, null, { timeout: 30000 });

  const r = await page.evaluate(async (CDN) => {
    const { embed, similarity, groups } = await import(CDN);
    const one = await embed('boss');
    const many = await embed(['manager', 'pizza', 'boss', 'manager']);
    const norm = Math.sqrt(one.reduce((n, x) => n + x * x, 0));
    const top = (await index.search('boss', 3)).map((x) => x.item.subject);
    const money = (await index.search('money problems', 3)).map((x) => x.item.subject);
    const sim = index.similar(index.items[3], 2).map((x) => x.item.subject);
    const g = groups(index.vectors, 3);
    let bad = '';
    try { await embed('x', { model: 'gpt-4o' }); } catch (e) { bad = e.message; }
    return {
      oneIsVector: one instanceof Float32Array && one.length === 512,
      manyShape: Array.isArray(many) && many.length === 4,
      norm,
      bossManager: similarity(many[2], many[0]), bossPizza: similarity(many[2], many[1]),
      dup: similarity(many[0], many[3]),
      top, money, sim,
      groupSizes: g.map((x) => x.length), groupTotal: g.flat().length,
      bad,
      rows: document.querySelectorAll('#topics .group').length,
    };
  }, CDN);

  check('embed(string) gives one Float32Array of 512', r.oneIsVector);
  check('embed(array) gives one vector per text, same order', r.manyShape);
  check('vectors have length 1', Math.abs(r.norm - 1) < 1e-4);
  check('the same text gives the same vector', r.dup > 0.9999);
  check(`"boss" is closer to "manager" than to "pizza" (${r.bossManager.toFixed(2)} vs ${r.bossPizza.toFixed(2)})`, r.bossManager > r.bossPizza);
  const boss = (s) => /manager|director|supervisor|CEO/.test(s);
  check(`search "boss": top hit and 2 of 3 are manager, director, supervisor or CEO emails: ${r.top.join(' | ')}`,
    boss(r.top[0]) && r.top.filter(boss).length >= 2);
  check(`search "money problems" finds invoice, refund or charge emails: ${r.money.join(' | ')}`,
    r.money.filter((s) => /Invoice|Refund|charged|Payment/.test(s)).length >= 2);
  check(`similar() leaves the item out: ${r.sim.join(' | ')}`, !r.sim.includes('Invoice 4471 is overdue') && r.sim.length === 2);
  check(`groups() puts every item in one of 3 groups (${r.groupSizes.join(', ')})`, r.groupSizes.length === 3 && r.groupTotal === 12);
  check(`a refused request raises its message: "${r.bad}"`, /model must be one of/.test(r.bad));
  check('example.html shows the topics', r.rows === 3);

  const typed = await (async () => {
    await page.fill('#q', 'party food');
    await page.waitForSelector('#results .row', { timeout: 15000 });
    return page.$$eval('#results .row .subject', (els) => els.map((e) => e.textContent));
  })();
  check(`example.html search box works: "party food" → ${typed[0]}`, /Pizza|Cake|lunch|Drinks/.test(typed[0]));

  // reload: everything comes from the browser's cache
  const before = calls;
  await page.reload();
  await page.waitForFunction(() => window.index, null, { timeout: 30000 });
  check(`a reload embeds nothing again (${calls - before} calls)`, calls === before);

  check('no page errors', errors.length === 0);
  if (errors.length) console.log(errors.join('\n'));
  await browser.close();
  const failed = results.filter(([, ok]) => !ok).length;
  console.log(`\n${results.length - failed} of ${results.length} checks pass`);
  process.exit(failed ? 1 : 0);
})();
