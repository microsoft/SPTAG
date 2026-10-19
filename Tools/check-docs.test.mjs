import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { checkMarkdown, headingAnchors, visibleLines } from './check-docs.mjs';

function fixture(t, text) {
  const root = mkdtempSync(join(tmpdir(), 'sptag-doc-check-'));
  t.after(() => rmSync(root, { recursive: true }));
  const file = join(root, 'README.md');
  writeFileSync(file, text);
  return { root, file };
}

test('heading anchors handle formatting, duplicate titles, and Unicode', () => {
  assert.deepEqual([...headingAnchors('# **Build**\n## Build\n## `API` usage\n## 中文')], [
    'build', 'build-1', 'api-usage', '中文',
  ]);
});

test('code fences ignore example links and require a matching close', () => {
  assert.deepEqual(visibleLines('````text\n[x](missing)\n```\n````').unclosedFence, false);
  assert.equal(visibleLines('~~~text\n```\n').unclosedFence, true);
});

test('valid local file, heading, image, reference, and external links pass', (t) => {
  const { root, file } = fixture(t, '# Home\n[x](guide.md#api)\n![image](img/p.png)\n[ref][g]\n[g]: guide.md#api\n[web](https://example.com)\n[home](#home)');
  mkdirSync(join(root, 'img'));
  writeFileSync(join(root, 'img', 'p.png'), '');
  writeFileSync(join(root, 'guide.md'), '# API');
  assert.deepEqual(checkMarkdown(file, root), []);
});

test('missing files, headings, HTML images, and references fail explicitly', (t) => {
  const { root, file } = fixture(t, '# Home\n[x](missing.md)\n[x](#absent)\n<img src="missing.png"/>\n[x][absent]');
  const errors = checkMarkdown(file, root);
  assert.equal(errors.length, 4);
  assert.match(errors[0], /missing target/);
  assert.match(errors[1], /missing heading/);
  assert.match(errors[3], /undefined reference/);
});

test('paths with spaces and escaped characters are checked', (t) => {
  const { root, file } = fixture(t, '[x](<a b.md#hello>)\n[x](a%20b.md#hello)');
  writeFileSync(join(root, 'a b.md'), '# Hello');
  assert.deepEqual(checkMarkdown(file, root), []);
});

test('fenced and inline code are not interpreted as links', (t) => {
  const { root, file } = fixture(t, '```markdown\n[x](missing.md)\n```\n`[x](missing.md)`');
  assert.deepEqual(checkMarkdown(file, root), []);
});

test('unclosed fences and out-of-repository paths fail', (t) => {
  const { root, file } = fixture(t, '[x](../outside.md)\n```bash\nexample');
  const errors = checkMarkdown(file, root);
  assert.equal(errors.length, 2);
  assert.match(errors[0], /unclosed code fence/);
  assert.match(errors[1], /escapes repository/);
});
