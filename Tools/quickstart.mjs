import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { accessSync, constants, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

if (process.argv.length !== 3) {
  console.error('Usage: node Tools/quickstart.mjs <directory-containing-indexbuilder-and-indexsearcher>');
  process.exit(1);
}

const binaryDirectory = resolve(process.argv[2]);
const executable = (name) => join(binaryDirectory, `${name}${process.platform === 'win32' ? '.exe' : ''}`);
for (const name of ['indexbuilder', 'indexsearcher']) {
  accessSync(executable(name), constants.X_OK);
}

const count = 1024;
const dimension = 8;
const expectedIds = [0, 127, 511];
const directory = mkdtempSync(join(tmpdir(), 'sptag-quickstart-'));

function writeVectors(path, ids) {
  const buffer = Buffer.alloc(8 + ids.length * dimension * 4);
  buffer.writeInt32LE(ids.length, 0);
  buffer.writeInt32LE(dimension, 4);
  for (const [row, id] of ids.entries()) {
    for (let column = 0; column < dimension; column++) {
      buffer.writeFloatLE(id + column / dimension, 8 + (row * dimension + column) * 4);
    }
  }
  writeFileSync(path, buffer);
}

function run(name, args) {
  const result = spawnSync(executable(name), args, {
    cwd: directory,
    stdio: 'inherit',
    timeout: 120000,
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `${name} failed (signal: ${result.signal ?? 'none'})`);
}

try {
  const vectors = join(directory, 'vectors.bin');
  const queries = join(directory, 'queries.bin');
  const index = join(directory, 'index');
  writeVectors(vectors, Array.from({ length: count }, (_, id) => id));
  writeVectors(queries, expectedIds);
  run('indexbuilder', [
    '-d', String(dimension), '-v', 'Float', '-f', 'DEFAULT', '-i', vectors,
    '-o', index, '-a', 'BKT', '-t', '1',
    'Index.DistCalcMethod=L2', 'Index.NumberOfThreads=1',
  ]);
  accessSync(join(index, 'indexloader.ini'));
  for (let attempt = 0; attempt < 2; attempt++) {
    const output = join(directory, `results-${attempt}.txt`);
    run('indexsearcher', [
      '-d', String(dimension), '-v', 'Float', '-f', 'DEFAULT', '-i', queries,
      '-x', index, '-o', output, '-k', '1', '-t', '1',
    ]);
    const rows = readFileSync(output, 'utf8').trim().split(/\r?\n/);
    assert.equal(rows.length, expectedIds.length, 'Unexpected query result count');
    for (const [query, row] of rows.entries()) {
      const match = /^(\d+):([0-9.eE+-]+)@(\d+)\|$/.exec(row);
      assert.ok(match, `Invalid result shape: ${row}`);
      assert.equal(Number(match[1]), query, 'Unexpected query ID');
      assert.equal(Number(match[3]), expectedIds[query], 'Incorrect nearest neighbor');
      assert.equal(Number(match[2]), 0, 'Self-match distance must be zero');
    }
  }
  console.log('Quick start passed: 3/3 nearest neighbors matched after save/load and repeat search.');
} finally {
  rmSync(directory, { recursive: true });
}
