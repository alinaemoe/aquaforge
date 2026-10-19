// Карты для игровых движков: heightmap (16 бит), splatmap материалов, flowmap, карта эрозии.
import { encodePNG } from './png';

export interface ExportState {
  n: number;
  world: number;
  cell: number;
  terrain: Float32Array;
  water: Float32Array;
  q: Float32Array; // пары (x, y) расхода на клетку
  acc: Float32Array; // накопленное изменение высоты (<0 вымывание, >0 осаждение)
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (a: number, b: number, x: number) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};

function heights16(st: ExportState): Uint16Array {
  const out = new Uint16Array(st.n * st.n);
  for (let i = 0; i < out.length; i++) out[i] = Math.round(clamp01(st.terrain[i]) * 65535);
  return out;
}

/** 16-битный серый PNG — напрямую в Unreal / Unity / Blender. */
export function heightPNG16(st: ExportState): Promise<Blob> {
  return encodePNG(st.n, st.n, 1, 16, heights16(st));
}

/** RAW .r16 (little-endian uint16) — формат импорта Unreal Landscape. */
export function heightR16(st: ExportState): Blob {
  return new Blob([heights16(st).buffer as ArrayBuffer], { type: 'application/octet-stream' });
}

/**
 * Splatmap (RGBA8): R — скала, G — трава/грунт, B — песок, A — снег.
 * Веса считаются так же, как биомный шейдер: по высоте над морем и уклону.
 */
export function splatPNG(st: ExportState, heightScale: number, seaLevel: number): Promise<Blob> {
  const { n, terrain } = st;
  const out = new Uint8Array(n * n * 4);
  const H = (x: number, y: number) => terrain[Math.min(n - 1, Math.max(0, y)) * n + Math.min(n - 1, Math.max(0, x))];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const h = terrain[y * n + x];
      const gx = ((H(x + 1, y) - H(x - 1, y)) * heightScale) / (2 * st.cell);
      const gy = ((H(x, y + 1) - H(x, y - 1)) * heightScale) / (2 * st.cell);
      const ny = 1 / Math.sqrt(1 + gx * gx + gy * gy);
      const slope = 1 - ny;
      const t = clamp01((h - seaLevel) / Math.max(1 - seaLevel, 1e-3));
      const sand = 1 - smooth(0, 0.06, t);
      const rock = Math.max(smooth(0.25, 0.5, t), smooth(0.16, 0.34, slope));
      const snow = smooth(0.78, 0.9, t) * (1 - smooth(0.3, 0.5, slope));
      const grass = Math.max(0, 1 - Math.max(sand, rock, snow));
      const sum = sand + rock + snow + grass || 1;
      const o = (y * n + x) * 4;
      out[o] = Math.round((rock / sum) * 255);
      out[o + 1] = Math.round((grass / sum) * 255);
      out[o + 2] = Math.round((sand / sum) * 255);
      out[o + 3] = Math.round((snow / sum) * 255);
    }
  }
  return encodePNG(n, n, 4, 8, out);
}

/**
 * Flowmap (RGBA8): R,G — направление течения (0.5 = покой, ±1 → 0..255), B — расход, A — глубина воды.
 */
export function flowPNG(st: ExportState): Promise<Blob> {
  const { n, q, water } = st;
  const out = new Uint8Array(n * n * 4);
  for (let i = 0; i < n * n; i++) {
    const qx = q[i * 2], qy = q[i * 2 + 1];
    const d = Math.max(water[i], 0.004);
    let vx = qx / d, vy = qy / d;
    const vl = Math.hypot(vx, vy);
    if (vl > 1.5) { vx *= 1.5 / vl; vy *= 1.5 / vl; }
    out[i * 4] = Math.round(clamp01(0.5 + (0.5 * vx) / 1.5) * 255);
    out[i * 4 + 1] = Math.round(clamp01(0.5 + (0.5 * vy) / 1.5) * 255);
    out[i * 4 + 2] = Math.round(clamp01(Math.hypot(qx, qy) / 0.02) * 255);
    out[i * 4 + 3] = Math.round(clamp01(water[i] / 0.05) * 255);
  }
  return encodePNG(n, n, 4, 8, out);
}

/** Карта эрозии (16 бит): 32768 = без изменений, меньше — вымывание, больше — осаждение (±0.25 высоты). */
export function erosionPNG16(st: ExportState): Promise<Blob> {
  const out = new Uint16Array(st.n * st.n);
  for (let i = 0; i < out.length; i++) out[i] = Math.round(clamp01(0.5 + st.acc[i] * 2) * 65535);
  return encodePNG(st.n, st.n, 1, 16, out);
}

/** Сопроводительный JSON с масштабами — чтобы настроить Landscape в движке. */
export function metaJSON(st: ExportState, heightScale: number): Blob {
  return new Blob(
    [
      JSON.stringify(
        {
          generator: 'AQUAFORGE',
          resolution: st.n,
          worldSizeUnits: st.world,
          heightScaleUnits: heightScale,
          note: 'height16 = normalized height * 65535; world Z extent = heightScaleUnits. Erosion map: 32768 = no change, ±0.25 height range.',
        },
        null,
        2,
      ),
    ],
    { type: 'application/json' },
  );
}
