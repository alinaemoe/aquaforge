// God-mode инструменты: пикинг курсора по рельефу (1 поток) и кисти (compute по всей сетке).

/** Луч из камеры → ячейка рельефа. Результат: out[0] = (cellX, cellY, hit 0|1, высота рельефа 0..1). */
export const PICK_WGSL = /* wgsl */ `
struct Pick {
  ro: vec4f,   // origin
  rd: vec4f,   // direction (normalized)
  p0: vec4f,   // n, heightScale, worldSize, maxT
};
@group(0) @binding(0) var<uniform> P: Pick;
@group(0) @binding(1) var<storage, read> terrain: array<f32>;
@group(0) @binding(2) var<storage, read_write> outHit: array<vec4f>;

fn hAt(x: f32, z: f32) -> f32 {
  let n = P.p0.x;
  let gx = clamp((x / P.p0.z + 0.5) * (n - 1.0), 0.0, n - 1.0);
  let gz = clamp((z / P.p0.z + 0.5) * (n - 1.0), 0.0, n - 1.0);
  let x0 = u32(floor(gx)); let z0 = u32(floor(gz));
  let x1 = min(x0 + 1u, u32(n) - 1u); let z1 = min(z0 + 1u, u32(n) - 1u);
  let fx = gx - floor(gx); let fz = gz - floor(gz);
  let N = u32(n);
  let a = terrain[z0 * N + x0]; let b = terrain[z0 * N + x1];
  let c = terrain[z1 * N + x0]; let d = terrain[z1 * N + x1];
  return mix(mix(a, b, fx), mix(c, d, fx), fz);
}

@compute @workgroup_size(1)
fn pick() {
  let half = P.p0.z * 0.5;
  var t = 0.0;
  var prevT = 0.0;
  var hit = false;
  for (var k = 0; k < 4000; k++) {
    let p = P.ro.xyz + P.rd.xyz * t;
    if (abs(p.x) <= half && abs(p.z) <= half) {
      let y = hAt(p.x, p.z) * P.p0.y;
      if (p.y <= y) { hit = true; break; }
    }
    prevT = t;
    t += 0.3;
    if (t > P.p0.w) { break; }
  }
  if (!hit) { outHit[0] = vec4f(0.0); return; }
  // уточнение бисекцией между prevT и t
  var lo = prevT; var hi = t;
  for (var k = 0; k < 8; k++) {
    let mid = 0.5 * (lo + hi);
    let p = P.ro.xyz + P.rd.xyz * mid;
    if (p.y <= hAt(p.x, p.z) * P.p0.y) { hi = mid; } else { lo = mid; }
  }
  let p = P.ro.xyz + P.rd.xyz * hi;
  let n = P.p0.x;
  let cx = clamp((p.x / P.p0.z + 0.5) * (n - 1.0), 0.0, n - 1.0);
  let cz = clamp((p.z / P.p0.z + 0.5) * (n - 1.0), 0.0, n - 1.0);
  outHit[0] = vec4f(cx, cz, 1.0, hAt(p.x, p.z));
}
`;

/** Кисти. op: 0 дождь, 1 поднять, 2 опустить, 3 плотина (отрезок a→b). */
export const TOOLS_WGSL = /* wgsl */ `
struct Tool {
  c: vec4f,    // центр (клетки x, y), радиус (клетки), сила
  op: vec4f,   // x = операция
  a: vec4f,    // отрезок плотины: ax, ay, bx, by (клетки)
  lvl: vec4f,  // x = уровень гребня (0..1), y = полуширина (клетки), z = n
};
@group(0) @binding(0) var<uniform> T: Tool;
@group(0) @binding(1) var<storage, read_write> tA: array<f32>;
@group(0) @binding(2) var<storage, read_write> w: array<f32>;

@compute @workgroup_size(16, 16)
fn apply(@builtin(global_invocation_id) g: vec3u) {
  let n = u32(T.lvl.z);
  if (g.x >= n || g.y >= n) { return; }
  let p = vec2f(g.xy);
  let i = g.y * n + g.x;
  let op = i32(T.op.x);
  if (op <= 2) {
    let d = length(p - T.c.xy);
    if (d < T.c.z) {
      let k = 1.0 - d / T.c.z;
      let f = k * k;
      if (op == 0) { w[i] = w[i] + T.c.w * f; }
      else if (op == 1) { tA[i] = min(tA[i] + T.c.w * f, 1.0); }
      else { tA[i] = max(tA[i] - T.c.w * f, 0.0); }
    }
  } else {
    let a = T.a.xy;
    let ab = T.a.zw - a;
    let len2 = dot(ab, ab);
    var t = 0.0;
    if (len2 > 0.5) { t = clamp(dot(p - a, ab) / len2, 0.0, 1.0); }
    let dist = length(p - (a + ab * t));
    let hw = T.lvl.y;
    if (dist < hw) {
      let prof = T.lvl.x - max(0.0, dist - hw * 0.5) / (hw * 0.5) * 0.04;
      tA[i] = max(tA[i], prof);
    }
  }
}
`;
