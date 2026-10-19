// Минимальный PNG-энкодер: 8/16 бит, 1/3/4 канала. Сжатие — нативный CompressionStream('deflate') (zlib).
// Нужен, потому что canvas.toBlob умеет только 8 бит, а карты высот для Unreal/Unity должны быть 16-битными.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

async function zlibDeflate(data: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const cs = new CompressionStream('deflate');
  const writer = cs.writable.getWriter();
  void writer.write(data);
  void writer.close();
  return new Uint8Array(await new Response(cs.readable).arrayBuffer());
}

function chunk(type: string, data: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/**
 * @param samples построчно, каналы подряд (Uint8Array для 8 бит, Uint16Array для 16 бит)
 */
export async function encodePNG(
  width: number,
  height: number,
  channels: 1 | 3 | 4,
  bitDepth: 8 | 16,
  samples: Uint8Array | Uint16Array,
): Promise<Blob> {
  const colorType = channels === 1 ? 0 : channels === 3 ? 2 : 6;
  const bytesPerSample = bitDepth / 8;
  const stride = width * channels * bytesPerSample;
  const raw = new Uint8Array((stride + 1) * height); // перед каждой строкой — байт фильтра (0)
  for (let y = 0; y < height; y++) {
    const row = y * (stride + 1);
    raw[row] = 0;
    const base = y * width * channels;
    if (bitDepth === 8) {
      raw.set((samples as Uint8Array).subarray(base, base + width * channels), row + 1);
    } else {
      const src = samples as Uint16Array;
      let o = row + 1;
      for (let i = 0; i < width * channels; i++) {
        const v = src[base + i];
        raw[o++] = v >> 8; // big-endian
        raw[o++] = v & 0xff;
      }
    }
  }
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = bitDepth;
  ihdr[9] = colorType;
  const idat = await zlibDeflate(raw);
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', new Uint8Array(0))], {
    type: 'image/png',
  });
}
