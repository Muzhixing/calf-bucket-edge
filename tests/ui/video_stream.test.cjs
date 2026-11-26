const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../../src/bucket_robot/runtime/video_stream.js'), 'utf8');
const { MultipartFrames, LatestDecoder } = vm.runInNewContext(source + '; videoStream', { TextDecoder, Uint8Array });
function packet(id, body) {
  return Buffer.concat([Buffer.from(`--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${body.length}\r\nX-Frame-Id: ${id}\r\nX-Frame-Age-Ms: 35.2\r\n\r\n`), body, Buffer.from('\r\n')]);
}
test('multipart handles every split point, including header and JPEG delimiters across chunks', () => {
  const body = Buffer.from('JPEG\r\n\r\n--frame-test'), data = packet(7, body);
  for (let split = 1; split < data.length; split++) {
    const parser = new MultipartFrames();
    const frames = [...parser.push(data.subarray(0, split)), ...parser.push(data.subarray(split))];
    const frame = frames[0];assert.equal(frames.length,1);
    assert.equal(frame.id, '7');assert.equal(frame.age, 35.2);
    assert.deepEqual(Buffer.from(frame.jpeg), body);
  }
});
test('coalesced network frames remain available for paced display', () => {
  const parser = new MultipartFrames();
  const result = parser.push(Buffer.concat([packet(1, Buffer.from('old')), packet(2, Buffer.from('new'))]));
  assert.equal(result.length,2);assert.equal(result[0].id,'1');
  assert.equal(result[1].id, '2');assert.equal(Buffer.from(result[1].jpeg).toString(), 'new');
});
test('invalid frame sizes and an unbounded header are rejected', () => {
  for (const size of ['0', '-1', 'nan', '9999999']) {
    const data = Buffer.from(`--frame\r\nContent-Length: ${size}\r\nX-Frame-Id: 1\r\nX-Frame-Age-Ms: 1\r\n\r\n`);
    assert.throws(() => new MultipartFrames().push(data));
  }
  assert.throws(() => new MultipartFrames().push(Buffer.alloc(8193)));
});
test('a slow decoder retains at most three pending images and resumes after a decode error', async () => {
  let release;const blocked = new Promise(resolve => { release = resolve; });
  const seen = [], errors = [];
  const decoder = new LatestDecoder(async n => { seen.push(n); if(n===1)await blocked;if(n===3)throw Error('decode'); }, e => errors.push(e));
  decoder.offer(1);decoder.offer(0);decoder.offer(0);decoder.offer(2);decoder.offer(3);
  assert.equal(decoder.pending.length,3);release();
  await new Promise(resolve => setImmediate(resolve));
  decoder.offer(4);await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(seen, [1,0,2,3,4]);assert.equal(errors.length,1);assert.equal(decoder.pending.length,0);assert.equal(decoder.busy,false);
});
