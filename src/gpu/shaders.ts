// WGSL: процедурная генерация heightmap (compute) и рендер рельефа (vertex+fragment).

export const NOISE_WGSL = /* wgsl */ `
struct Params {
  seed: f32,
  freq: f32,
  octaves: f32,
  lacunarity: f32,
  gain: f32,
  ridge: f32,    // 0..1 доля ridge-шума
  worley: f32,   // 0..1 доля cellular-шума
  warp: f32,     // сила domain warp
  island: f32,   // 0..1 маска острова
  exponent: f32, // степень профиля высот
  size: f32,     // размер текстуры
  _pad: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> outBuf: array<f32>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn hash2(p: vec2i, seed: u32) -> vec2f {
  let a = pcg(bitcast<u32>(p.x) + pcg(bitcast<u32>(p.y) + pcg(seed)));
  let b = pcg(a ^ 0x9e3779b9u);
  return vec2f(f32(a), f32(b)) / 4294967295.0;
}

fn perlin(p: vec2f, seed: u32) -> f32 {
  let i = vec2i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let g00 = hash2(i, seed) * 2.0 - 1.0;
  let g10 = hash2(i + vec2i(1, 0), seed) * 2.0 - 1.0;
  let g01 = hash2(i + vec2i(0, 1), seed) * 2.0 - 1.0;
  let g11 = hash2(i + vec2i(1, 1), seed) * 2.0 - 1.0;
  let n00 = dot(g00, f);
  let n10 = dot(g10, f - vec2f(1.0, 0.0));
  let n01 = dot(g01, f - vec2f(0.0, 1.0));
  let n11 = dot(g11, f - vec2f(1.0, 1.0));
  return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y) * 1.4142;
}

fn worley(p: vec2f, seed: u32) -> f32 {
  let i = vec2i(floor(p));
  let f = fract(p);
  var d = 8.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let o = vec2i(x, y);
      let r = hash2(i + o, seed) + vec2f(o) - f;
      d = min(d, dot(r, r));
    }
  }
  return sqrt(d);
}

// fBm: x = обычный (0..1), y = ridged (0..1), z = cellular fBm (0..1)
fn stack(p0: vec2f, seed: u32) -> vec3f {
  var p = p0;
  var amp = 0.5;
  var sumN = 0.0;
  var sumR = 0.0;
  var sumW = 0.0;
  var norm = 0.0;
  let oct = i32(P.octaves);
  for (var k = 0; k < 10; k++) {
    if (k >= oct) { break; }
    let n = perlin(p, seed + u32(k));
    sumN += n * amp;
    let r = 1.0 - abs(n);
    sumR += r * r * amp;
    sumW += (1.0 - clamp(worley(p, seed + u32(k) * 7u), 0.0, 1.0)) * amp;
    norm += amp;
    p = p * P.lacunarity + vec2f(17.3, 9.1);
    amp *= P.gain;
  }
  return vec3f(sumN / norm * 0.5 + 0.5, sumR / norm, sumW / norm);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = u32(P.size);
  if (gid.x >= size || gid.y >= size) { return; }
  let uv = vec2f(gid.xy) / (P.size - 1.0);
  let seed = u32(P.seed);
  var p = uv * P.freq;

  // domain warp
  if (P.warp > 0.0) {
    let w = vec2f(
      perlin(p * 0.7 + vec2f(5.2, 1.3), seed + 101u),
      perlin(p * 0.7 + vec2f(8.7, 2.8), seed + 202u)
    );
    p += w * P.warp;
  }

  let s = stack(p, seed);
  var h = mix(s.x, s.y, P.ridge);
  h = mix(h, s.z, P.worley * 0.5);

  // маска острова
  let c = uv * 2.0 - 1.0;
  let fall = clamp(1.0 - pow(length(c), 2.5), 0.0, 1.0);
  h = mix(h, h * fall, P.island);

  h = pow(clamp(h, 0.0, 1.0), P.exponent);
  outBuf[gid.y * size + gid.x] = h;
}
`;
