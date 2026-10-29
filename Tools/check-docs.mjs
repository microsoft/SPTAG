import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, extname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

export function visibleLines(text) {
  let fence;
  const lines = text.split(/\r?\n/).map((line) => {
    const marker = /^\s{0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (marker) {
      if (!fence) fence = { character: marker[1][0], length: marker[1].length };
      else if (marker[1][0] === fence.character && marker[1].length >= fence.length && !marker[2].trim()) fence = undefined;
      return '';
    }
    return fence ? '' : line;
  });
  return { lines, unclosedFence: Boolean(fence) };
}

export function headingAnchors(text) {
  const anchors = new Set();
  const duplicates = new Map();
  const { lines } = visibleLines(text);
  for (let i = 0; i < lines.length; i++) {
    const heading = /^\s{0,3}#{1,6}\s+(.+?)(?:\s+#+)?$/.exec(lines[i]);
    const setext = i + 1 < lines.length && /^\s{0,3}(?:=+|-+)\s*$/.test(lines[i + 1]) && lines[i].trim();
    if (!heading && !setext) continue;
    const title = (heading ? heading[1] : lines[i]).replace(/<[^>]+>/g, '');
    const slug = title.toLowerCase().replace(/[^\p{L}\p{N}\p{M}_\-\s]/gu, '').replace(/\s/g, '-');
    const count = duplicates.get(slug) ?? 0;
    duplicates.set(slug, count + 1);
    anchors.add(count ? `${slug}-${count}` : slug);
  }
  for (const match of text.matchAll(/\b(?:id|name)=["']([^"']+)["']/g)) anchors.add(match[1]);
  return anchors;
}

export function checkMarkdown(file, root) {
  const text = readFileSync(file, 'utf8');
  const { lines, unclosedFence } = visibleLines(text);
  const errors = [];
  if (unclosedFence) errors.push(`${relative(root, file)}: unclosed code fence`);
  const references = new Map();
  for (const line of lines) {
    const definition = /^\s{0,3}\[([^\]]+)\]:\s*<?([^\s>]+)>?/.exec(line);
    if (definition) references.set(definition[1].toLowerCase(), definition[2]);
  }
  function check(target, line) {
    if (/^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(target)) return;
    const hash = target.indexOf('#');
    const path = decodeURIComponent(hash < 0 ? target : target.slice(0, hash));
    const fragment = hash < 0 ? '' : decodeURIComponent(target.slice(hash + 1));
    const destination = path ? resolve(dirname(file), path) : file;
    const inside = relative(root, destination);
    if (inside === '..' || inside.startsWith(`..${sep}`) || isAbsolute(inside)) {
      errors.push(`${relative(root, file)}:${line}: link escapes repository: ${target}`);
    } else if (!existsSync(destination)) {
      errors.push(`${relative(root, file)}:${line}: missing target: ${target}`);
    } else if (fragment && extname(destination).toLowerCase() === '.md' &&
               !headingAnchors(readFileSync(destination, 'utf8')).has(fragment)) {
      errors.push(`${relative(root, file)}:${line}: missing heading: ${target}`);
    }
  }
  for (const [index, source] of lines.entries()) {
    const line = source.replace(/(`+).*?\1/g, '');
    for (const match of line.matchAll(/!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+["'][^"']*["'])?\s*\)/g)) {
      check(match[1] ?? match[2], index + 1);
    }
    for (const match of line.matchAll(/\b(?:src|href)=["']([^"']+)["']/g)) check(match[1], index + 1);
    const definition = /^\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?/.exec(line);
    if (definition) check(definition[1], index + 1);
    for (const match of line.matchAll(/!?\[([^\]]+)\]\[([^\]]*)\]/g)) {
      const label = (match[2] || match[1]).toLowerCase();
      if (!references.has(label)) errors.push(`${relative(root, file)}:${index + 1}: undefined reference: ${label}`);
    }
  }
  return errors;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const files = [...new Set(execFileSync('git', [
    'ls-files', '--cached', '--others', '--exclude-standard', '-z',
  ], { cwd: root, encoding: 'utf8' }).split('\0'))]
    .filter((file) => file.endsWith('.md') && !file.startsWith('ThirdParty/') && existsSync(resolve(root, file)));
  const errors = files.flatMap((file) => checkMarkdown(resolve(root, file), root));
  if (errors.length) {
    console.error(errors.join('\n'));
    process.exitCode = 1;
  } else {
    console.log(`Documentation checks passed: ${files.length} Markdown files (local links, headings, images, and code fences).`);
  }
}
