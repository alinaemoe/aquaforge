// Освещение карты высот: мягкие тени от солнца (raymarch по рельефу) + ambient occlusion по соседним высотам.
// Результат: для каждой клетки vec2(тень 0..1, AO 0..1) — читается вершинным шейдером рендера.

export const LIGHT_WGSL = /* wgsl */ `
struct L {
  p: vec4f,    // n, heightScale, world, sea
  sun: vec4f,  // направление на солнце (xyz)
};
@group(0) @binding(0) var<uniform> Lu: L;
@group(0) @binding(1) var<storage, read> hm: array<f32>;
@group(0) @binding(2) var<storage, read_write> outL: array<vec2f>;

// высота поверхности в мировых единицах (не ниже моря) в мировой точке (x, z)
fn hAt(x: f32, z: f32) -> f32 {
  let n = Lu.p.x;
  let gx = clamp((x / Lu.p.z + 0.5) * (n - 1.0), 0.0, n - 1.0);
  let gz = clamp((z / Lu.p.z + 0.5) * (n - 1.0), 0.0, n - 1.0);
  let x0 = u32(floor(gx)); let z0 = u32(floor(gz));
  let N = u32(n);
  let x1 = min(x0 + 1u, N - 1u); let z1 = min(z0 + 1u, N - 1u);
  let fx = gx - floor(gx); let fz = gz - floor(gz);
  let h = mix(mix(hm[z0 * N + x0], hm[z0 * N + x1], fx), mix(hm[z1 * N + x0], hm[z1 * N + x1], fx), fz);
  return max(h, Lu.p.w) * Lu.p.y;
}

@compute @workgroup_size(8, 8)
fn light(@builtin(global_invocation_id) g: vec3u) {
  let N = u32(Lu.p.x);
  if (g.x >= N || g.y >= N) { return; }
  let n = Lu.p.x;
  let world = Lu.p.z;
  let half = world * 0.5;
  let x = (f32(g.x) / (n - 1.0) - 0.5) * world;
  let z = (f32(g.y) / (n - 1.0) - 0.5) * world;
  let hc = hAt(x, z);
  let hs = Lu.p.y;

  // мягкая тень: минимальный «угол просвета» вдоль луча на солнце
  let dir = normalize(Lu.sun.xyz);
  var sh = 1.0;
  var t = 0.4;
  for (var k = 0; k < 64; k++) {
    let p = vec3f(x, hc + 0.06, z) + dir * t;
    if (abs(p.x) > half || abs(p.z) > half || p.y > hs) { break; }
    let d = p.y - hAt(p.x, p.z);
    sh = min(sh, clamp(7.0 * d / t, 0.0, 1.0));
    if (sh < 0.01) { break; }
    t += 0.22 + t * 0.075;
  }
  // солнце за горизонтом — всё в тени
  sh = sh * smoothstep(0.0, 0.05, dir.y);

  // AO: насколько окружающий рельеф выше точки на нескольких радиусах и в 8 направлениях
  var occ = 0.0;
  for (var r = 0; r < 3; r++) {
    let rad = select(select(0.8, 2.2, r == 1), 5.0, r == 2);
    for (var a = 0; a < 8; a++) {
      let ang = f32(a) * 0.785398;
      let hh = hAt(x + cos(ang) * rad, z + sin(ang) * rad);
      occ += clamp((hh - hc) / (rad * 1.6), 0.0, 1.0);
    }
  }
  let ao = 1.0 - 0.7 * (occ / 24.0);
  outL[g.y * N + g.x] = vec2f(sh, ao);
}
`;
