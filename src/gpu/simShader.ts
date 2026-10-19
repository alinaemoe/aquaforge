// Сеточная гидро-эрозия на GPU (виртуальные трубы, Mei et al. 2007 + thermal).
// Все величины — в нормализованных единицах высоты, шаг времени = 1.
// Порядок за шаг: calcFlux → updateWater(+rain) → erode → advect(+tick) → thermal.
//
// Оптимизация: соседи читаются через тайлы в workgroup-памяти (16×16 + halo),
// размер сетки обязан быть кратен 16 (проверок границ потоков нет — барьеры требуют uniform-потока).

/** f16 = true: flux/q/осадок хранятся в half (меньше трафика памяти); иначе f32. */
import { STRATA_WGSL } from './strataShader';

export function simWgsl(f16: boolean): string {
  const prelude = f16
    ? `enable f16;
alias hf = f16;
alias hv2 = vec2<f16>;
alias hv4 = vec4<f16>;
`
    : `alias hf = f32;
alias hv2 = vec2<f32>;
alias hv4 = vec4<f32>;
`;
  return prelude + SIM_BODY + STRATA_WGSL;
}

const SIM_BODY = /* wgsl */ `
struct Sim {
  n: f32,
  rain: f32,       // слой воды за шаг (в среднем)
  evap: f32,       // доля испарения за шаг
  kc: f32,         // множитель ёмкости потока
  ks: f32,         // скорость растворения
  kd: f32,         // скорость осаждения
  talusH: f32,     // макс. устойчивый перепад между соседними клетками
  thermal: f32,    // скорость осыпания
  slopeK: f32,     // перевод перепада высот в тангенс уклона
  flowG: f32,      // "гравитация" в трубах
  sea: f32,
  rainProb: f32,
  rainOn: f32,
  stormPeriod: f32, // шаги между ливнями (0 — без штормов)
  stormDuty: f32,   // доля периода, когда идёт дождь
  _c: f32,
  st: vec4f,       // геология: scale, warp, dipX, dipY
  hard: vec4f,     // податливость пород: глина, песчаник, известняк, базальт
  src: array<vec4f, 8>, // источники рек: клетка x, y, расход, радиус (клетки)
};

@group(0) @binding(0) var<uniform> S: Sim;
@group(0) @binding(1) var<storage, read_write> tA: array<f32>;   // рельеф (основной)
@group(0) @binding(2) var<storage, read_write> tB: array<f32>;   // рельеф (промежуточный)
@group(0) @binding(3) var<storage, read_write> w: array<f32>;    // глубина воды
@group(0) @binding(4) var<storage, read_write> sA: array<hf>;   // осадок
@group(0) @binding(5) var<storage, read_write> sB: array<hf>;
@group(0) @binding(6) var<storage, read_write> flux: array<hv4>; // L, R, T(y-1), B(y+1)
@group(0) @binding(7) var<storage, read_write> q: array<hv2>;    // расход воды через клетку
// acc[0..N*N) — накопленное изменение высоты от эрозии/осаждения (f32 в битах u32),
// acc[N*N] — счётчик шагов (для случайности дождя)
@group(0) @binding(8) var<storage, read_write> acc: array<u32>;

const B: u32 = 16u;          // сторона тайла (рабочей группы)
const T1: u32 = 18u;         // тайл с halo=1
const T2: u32 = 20u;         // тайл с halo=2

var<workgroup> tileF: array<f32, 324>;     // halo 1, f32
var<workgroup> tileV: array<vec4f, 324>;   // halo 1, vec4
var<workgroup> tileS: array<f32, 400>;     // halo 2, f32

fn id(p: vec2i) -> u32 {
  let n = i32(S.n);
  let c = clamp(p, vec2i(0), vec2i(n - 1));
  return u32(c.y * n + c.x);
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let x = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (x >> 22u) ^ x;
}

// индекс клетки внутри тайла halo=1 (lx, ly — локальные координаты 0..15)
fn t1(lx: u32, ly: u32) -> u32 { return (ly + 1u) * T1 + (lx + 1u); }

@compute @workgroup_size(16, 16)
fn calcFlux(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let org = vec2i(wid.xy) * 16 - 1;
  for (var k = li; k < 324u; k += 256u) {
    let j = id(org + vec2i(i32(k % T1), i32(k / T1)));
    tileF[k] = tA[j] + w[j];
  }
  workgroupBarrier();

  let lx = li % B; let ly = li / B;
  let p = vec2i(wid.xy) * 16 + vec2i(i32(lx), i32(ly));
  let i = id(p);
  let k0 = t1(lx, ly);
  let c = tileF[k0];
  let f = vec4f(flux[i]);
  let G = S.flowG;
  var nf = vec4f(
    max(0.0, f.x * 0.98 + G * (c - tileF[k0 - 1u])),
    max(0.0, f.y * 0.98 + G * (c - tileF[k0 + 1u])),
    max(0.0, f.z * 0.98 + G * (c - tileF[k0 - T1])),
    max(0.0, f.w * 0.98 + G * (c - tileF[k0 + T1]))
  );
  let sum = nf.x + nf.y + nf.z + nf.w;
  let d = w[i];
  if (sum > d && sum > 0.0) { nf *= d / sum; }
  flux[i] = hv4(nf);
}

@compute @workgroup_size(16, 16)
fn updateWater(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let org = vec2i(wid.xy) * 16 - 1;
  for (var k = li; k < 324u; k += 256u) {
    tileV[k] = vec4f(flux[id(org + vec2i(i32(k % T1), i32(k / T1)))]);
  }
  workgroupBarrier();

  let lx = li % B; let ly = li / B;
  let p = vec2i(wid.xy) * 16 + vec2i(i32(lx), i32(ly));
  let n = i32(S.n);
  let i = id(p);
  let k0 = t1(lx, ly);
  let f = tileV[k0];
  let fl = tileV[k0 - 1u];
  let fr = tileV[k0 + 1u];
  let fu = tileV[k0 - T1];
  let fd = tileV[k0 + T1];
  // на границе соседей нет — вклад «снаружи» не учитываем
  var inflow = 0.0;
  if (p.x > 0)     { inflow += fl.y; }
  if (p.x < n - 1) { inflow += fr.x; }
  if (p.y > 0)     { inflow += fu.w; }
  if (p.y < n - 1) { inflow += fd.z; }
  var d = w[i] + inflow - (f.x + f.y + f.z + f.w);

  q[i] = hv2(vec2f(0.5 * (f.y + fl.y - f.x - fr.x), 0.5 * (f.w + fu.w - f.z - fd.z)));

  // дождь
  if (S.rainOn > 0.5) {
    let cnt = u32(n) * u32(n);
    let r = f32(pcg(u32(p.x) + pcg(u32(p.y) + pcg(acc[cnt] * 2654435761u + 17u)))) / 4294967295.0;
    var rainMul = 1.0;
    if (S.stormPeriod > 0.5) {
      let ph = f32(acc[cnt] % u32(S.stormPeriod)) / S.stormPeriod;
      rainMul = select(0.0, 1.0 / S.stormDuty, ph < S.stormDuty);
    }
    if (r < S.rainProb) { d += S.rain * rainMul / S.rainProb; }
  }

  // постоянные источники (родники)
  for (var k = 0; k < 8; k++) {
    let sr = S.src[k];
    if (sr.z > 0.0) {
      let dd = length(vec2f(p) - sr.xy);
      if (dd < sr.w) { d += sr.z * (1.0 - dd / sr.w); }
    }
  }

  // сток: края карты и океан забирают воду
  if (p.x == 0 || p.y == 0 || p.x == n - 1 || p.y == n - 1) { d = 0.0; }
  if (tA[i] < S.sea) { d *= 0.7; }
  w[i] = max(d, 0.0);
}

@compute @workgroup_size(16, 16)
fn erode(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let org = vec2i(wid.xy) * 16 - 1;
  for (var k = li; k < 324u; k += 256u) {
    tileF[k] = tA[id(org + vec2i(i32(k % T1), i32(k / T1)))];
  }
  workgroupBarrier();

  let lx = li % B; let ly = li / B;
  let p = vec2i(wid.xy) * 16 + vec2i(i32(lx), i32(ly));
  let i = id(p);
  let k0 = t1(lx, ly);
  var h = tileF[k0];
  var s = f32(sB[i]);

  let gx = (tileF[k0 + 1u] - tileF[k0 - 1u]) * 0.5 * S.slopeK;
  let gy = (tileF[k0 + T1] - tileF[k0 - T1]) * 0.5 * S.slopeK;
  let gl = length(vec2f(gx, gy));
  let sinA = gl / sqrt(1.0 + gl * gl);
  let qm = length(vec2f(q[i]));
  let cap = S.kc * max(sinA, 0.01) * qm;

  if (h < S.sea || w[i] < 1e-4) {
    // океан / пересохшая клетка: весь осадок оседает (дельты)
    h += s;
    s = 0.0;
  } else if (s < cap) {
    let hardMul = S.hard[rockType(strataU(vec2f(p) / (S.n - 1.0), h, S.st))];
    let e = min(S.ks * hardMul * (cap - s), h);
    h -= e;
    s += e;
  } else {
    let dp = S.kd * (s - cap);
    h += dp;
    s -= dp;
  }
  acc[i] = bitcast<u32>(bitcast<f32>(acc[i]) + (h - tileF[k0]));
  tB[i] = h;
  sA[i] = hf(s);
}

@compute @workgroup_size(16, 16)
fn advect(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let org = vec2i(wid.xy) * 16 - 2;
  for (var k = li; k < 400u; k += 256u) {
    tileS[k] = f32(sA[id(org + vec2i(i32(k % T2), i32(k / T2)))]);
  }
  workgroupBarrier();

  let lx = li % B; let ly = li / B;
  let p = vec2i(wid.xy) * 16 + vec2i(i32(lx), i32(ly));
  let i = id(p);
  if (p.x == 0 && p.y == 0) {
    let cnt = u32(S.n) * u32(S.n);
    acc[cnt] = acc[cnt] + 1u;
  }

  var v = vec2f(q[i]) / max(w[i], 0.004);
  let l = length(v);
  if (l > 1.5) { v *= 1.5 / l; }

  let m = S.n - 1.0;
  let pf = clamp(vec2f(p) - v, vec2f(0.0), vec2f(m));
  let f0 = floor(pf);
  let fr = pf - f0;
  let loc = vec2i(f0) - org;           // 0..18
  let b0 = u32(loc.y) * T2 + u32(loc.x);
  let a = tileS[b0];
  let b = tileS[b0 + 1u];
  let c = tileS[b0 + T2];
  let d = tileS[b0 + T2 + 1u];
  sB[i] = hf(mix(mix(a, b, fr.x), mix(c, d, fr.x), fr.y));
  w[i] = w[i] * (1.0 - S.evap);
}

@compute @workgroup_size(16, 16)
fn thermal(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) li: u32,
) {
  let org = vec2i(wid.xy) * 16 - 1;
  for (var k = li; k < 324u; k += 256u) {
    tileF[k] = tB[id(org + vec2i(i32(k % T1), i32(k / T1)))];
  }
  workgroupBarrier();

  let lx = li % B; let ly = li / B;
  let p = vec2i(wid.xy) * 16 + vec2i(i32(lx), i32(ly));
  let k0 = t1(lx, ly);
  let h = tileF[k0];
  let T = S.talusH;
  var dh = 0.0;
  var diff = tileF[k0 - 1u] - h;
  if (diff > T) { dh += S.thermal * (diff - T); } else if (diff < -T) { dh -= S.thermal * (-diff - T); }
  diff = tileF[k0 + 1u] - h;
  if (diff > T) { dh += S.thermal * (diff - T); } else if (diff < -T) { dh -= S.thermal * (-diff - T); }
  diff = tileF[k0 - T1] - h;
  if (diff > T) { dh += S.thermal * (diff - T); } else if (diff < -T) { dh -= S.thermal * (-diff - T); }
  diff = tileF[k0 + T1] - h;
  if (diff > T) { dh += S.thermal * (diff - T); } else if (diff < -T) { dh -= S.thermal * (-diff - T); }
  tA[id(p)] = h + dh;
}
`;
