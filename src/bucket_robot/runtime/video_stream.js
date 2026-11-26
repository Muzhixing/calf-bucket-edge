// Transport only. No control commands, target selection, or motion decisions.
const videoStream = (() => {
  class MultipartFrames {
    constructor() { this.buffer = new Uint8Array(0); this.header = null; }
    push(chunk) {
      const joined = new Uint8Array(this.buffer.length + chunk.length);
      joined.set(this.buffer); joined.set(chunk, this.buffer.length); this.buffer = joined;
      if (joined.length > 2 * 1024 * 1024) throw new Error('视频缓存超出限制');
      const frames = [];
      while (true) {
        if (!this.header) {
          let end = -1;
          for (let i = 0; i + 3 < this.buffer.length; i++) {
            if (this.buffer[i] === 13 && this.buffer[i+1] === 10 && this.buffer[i+2] === 13 && this.buffer[i+3] === 10) { end = i; break; }
          }
          if (end < 0) { if (this.buffer.length > 8192) throw new Error('视频帧头无效'); return frames; }
          const text = new TextDecoder().decode(this.buffer.subarray(0, end));
          const field = name => text.match(new RegExp('^' + name + ':\\s*(.+)$', 'im'))?.[1].trim();
          const size = Number(field('Content-Length')), id = field('X-Frame-Id'), age = Number(field('X-Frame-Age-Ms'));
          if (!Number.isSafeInteger(size) || size <= 0 || size > 1024 * 1024 || !/^\d+$/.test(id || '') || !Number.isFinite(age) || age < 0) throw new Error('视频帧信息无效');
          this.header = { size, id, age }; this.buffer = this.buffer.slice(end + 4);
        }
        if (this.buffer.length < this.header.size) return frames;
        frames.push({ jpeg: this.buffer.slice(0, this.header.size), id: this.header.id, age: this.header.age });
        if(frames.length>6)frames.shift();
        this.buffer = this.buffer.slice(this.header.size); this.header = null;
      }
    }
  }
  class LatestDecoder {
    constructor(consume, failed) { this.consume = consume; this.failed = failed; this.pending = []; this.busy = false; }
    offer(frame) { this.pending.push(frame);if(this.pending.length>3)this.pending.shift();if (!this.busy) this.drain(); }
    async drain() {
      this.busy = true;
      try {
        while (this.pending.length) {
          const frame = this.pending.shift();
          try { await this.consume(frame); } catch (error) { this.failed(error); }
        }
      } finally { this.busy = false; }
    }
  }
  return { MultipartFrames, LatestDecoder };
})();
