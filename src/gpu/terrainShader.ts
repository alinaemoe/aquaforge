// Рендер рельефа и diorama-стенок. Высота и вода читаются прямо из storage-буферов симуляции.
import { STRATA_WGSL } from './strataShader';

const BODY = /* wgsl */ `
struct U {
  vp: mat4x4f,
  cam: vec4f,   // xyz = позиция камеры
  p0: vec4f,    // x = клеток по стороне, y = масштаб высоты, z = уровень моря (0..1), w = размер мира
  sun: vec4f,   // xyz = направление на солнце
  st: vec4f,    // геология: scale, warp, dipX, dipY
  opts: vec4f,  // x = режим (0 биомы, 1 геология), y = позиция среза по X (мир), z = diorama, w = глубина основания
  brush: vec4f, // курсор: мир x, z, радиус, активен
  seg: vec4f,   // превью плотины: ax, az, bx, bz (мир)
  opts2: vec4f, // x = время, y = оверлей (0 нет, 1 эрозия, 2 скорость, 3 Δвысота), z = превью плотины, w = —
  src: array<vec4f, 8>, // источники рек: мир x, z, активен
  camF: vec4f,  // вперёд, w = aspect
  camR: vec4f,  // вправо
  camU: vec4f,  // вверх, w = tan(fov/2)
  atm: vec4f,   // x = плотность тумана
  skyH: vec4f,  // небо у горизонта
  skyZ: vec4f,  // небо в зените
  palG: vec4f,  // трава rgb, w = снег вкл (0/1)
  palS: vec4f,  // песок rgb
  sunC: vec4f,  // цвет солнца rgb, w = интенсивность
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> terrain: array<f32>;
@group(0) @binding(2) var<storage, read> water: array<f32>;
@group(0) @binding(3) var<storage, read> h0: array<f32>;      // исходный рельеф
@group(0) @binding(4) var<storage, read> acc: array<u32>;     // накопленная эрозия (f32 в битах)
//Q_DECL//
@group(0) @binding(6) var<storage, read> lightBuf: array<vec2f>; // (тень, AO) по клеткам

// Split-view: «до» — исходный рельеф без воды, «после» — текущее состояние
// opts2.w: 0 — как есть, 1 — «до» (h0, без воды), 2 — снимок time-lapse (рельеф из буфера, без воды)
fn isBefore() -> bool { return u.opts2.w > 0.5 && u.opts2.w < 1.5; }
fn noWater() -> bool { return u.opts2.w > 0.5; }
fn Ht(i: u32) -> f32 { if (isBefore()) { return h0[i]; } return terrain[i]; }
fn Wt(i: u32) -> f32 { if (noWater()) { return 0.0; } return water[i]; }

fn cid(c: vec2i) -> u32 {
  let n = i32(u.p0.x);
  let cc = clamp(c, vec2i(0), vec2i(n - 1));
  return u32(cc.y * n + cc.x);
}

// высота поверхности (вода или земля, не ниже моря)
fn S(c: vec2i) -> f32 {
  let i = cid(c);
  return max(Ht(i) + Wt(i), u.p0.z);
}

fn skyColor(dir: vec3f) -> vec3f {
  let d = normalize(dir);
  let t = clamp(d.y, 0.0, 1.0);
  var c = mix(u.skyH.xyz, u.skyZ.xyz, pow(t, 0.5));
  let sd = max(dot(d, normalize(u.sun.xyz)), 0.0);
  c += vec3f(1.0, 0.82, 0.55) * (pow(sd, 600.0) * 6.0 + pow(sd, 14.0) * 0.22);
  return c;
}

// анимированная каустика: сеть ярких линий в нулях интерференции искажённых синусов
// (без делений — устойчива, в отличие от «классической» формулы с сингулярностями)
fn caustic(p: vec2f, t: f32) -> f32 {
  var q = p * 0.85;
  var v = 0.0;
  for (var n = 0; n < 3; n++) {
    let fn_ = f32(n + 1);
    q += vec2f(sin(q.y * 1.1 + t * 0.45 * fn_), cos(q.x * 1.3 - t * 0.38 * fn_)) * 0.7;
    v += abs(sin(q.x * 1.7) * sin(q.y * 1.9));
  }
  v /= 3.0;
  return pow(clamp(1.0 - v * 2.2, 0.0, 1.0), 3.0) * 1.6;
}

fn foamNoise(p: vec2f, t: f32) -> f32 {
  let a = sin(p.x * 1.7 + t * 0.9) * sin(p.y * 1.3 - t * 0.7);
  let b = sin((p.x + p.y) * 2.9 - t * 1.3) * sin((p.x - p.y) * 2.3 + t * 0.8);
  return 0.5 + 0.28 * a + 0.22 * b;
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i), hash21(i + vec2f(1.0, 0.0)), w.x), mix(hash21(i + vec2f(0.0, 1.0)), hash21(i + vec2f(1.0, 1.0)), w.x), w.y);
}

// облака: слой на высоте CLOUD_H, дрейфуют по ветру
const CLOUD_H = 70.0;
fn cloudDens(p: vec2f, t: f32) -> f32 {
  let q = p * 0.018 + vec2f(t * 0.004, t * 0.0015);
  let f = vnoise(q) * 0.55 + vnoise(q * 2.3 + 7.0) * 0.3 + vnoise(q * 5.1 + 3.0) * 0.15;
  return smoothstep(0.42, 0.64, f);
}
// тень облаков: луч к солнцу пересекает слой облаков
fn cloudShade(world: vec3f) -> f32 {
  let L = normalize(u.sun.xyz);
  let q = world.xz + L.xz / max(L.y, 0.08) * max(CLOUD_H - world.y, 0.0);
  return 1.0 - 0.55 * cloudDens(q, u.opts2.x);
}

fn worldUV(p: vec3f) -> vec2f {
  return p.xz / u.p0.w + vec2f(0.5);
}

// ───────────────────────── рельеф ─────────────────────────
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) h: f32,
  @location(2) nrm: vec3f,
  @location(3) depth: f32,
  @location(4) fx: vec4f,   // эрозия, Δвысота, скорость x, y
  @location(5) wd: f32,     // глубина воды симуляции
  @location(6) lt: vec2f,   // тень, AO
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let n = u32(u.p0.x);
  let q = vi / 6u;
  let k = vi % 6u;
  let qx = q % (n - 1u);
  let qy = q / (n - 1u);
  var offs = array<vec2u, 6>(
    vec2u(0u, 0u), vec2u(0u, 1u), vec2u(1u, 0u),
    vec2u(1u, 0u), vec2u(0u, 1u), vec2u(1u, 1u)
  );
  let c = vec2i(vec2u(qx, qy) + offs[k]);
  let i = cid(c);
  let h = Ht(i);
  let surf = S(c);
  let world = u.p0.w;
  let step = world / (u.p0.x - 1.0);
  let x = (f32(c.x) / (u.p0.x - 1.0) - 0.5) * world;
  let z = (f32(c.y) / (u.p0.x - 1.0) - 0.5) * world;

  let dx = (S(c + vec2i(1, 0)) - S(c - vec2i(1, 0))) * u.p0.y;
  let dz = (S(c + vec2i(0, 1)) - S(c - vec2i(0, 1))) * u.p0.y;

  var o: VOut;
  o.world = vec3f(x, surf * u.p0.y, z);
  o.pos = u.vp * vec4f(o.world, 1.0);
  o.h = h;
  o.nrm = normalize(vec3f(-dx, 2.0 * step, -dz));
  o.depth = surf - h;
  var vel = readQ(i) / max(Wt(i), 0.004);
  let vl = length(vel);
  if (vl > 1.5) { vel *= 1.5 / vl; }
  o.fx = vec4f(bitcast<f32>(acc[i]), h - h0[i], vel.x, vel.y);
  if (noWater()) { o.fx = vec4f(0.0); }
  o.wd = Wt(i);
  o.lt = lightBuf[i];
  return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  if (i.world.x > u.opts.y) { discard; }
  let sea = u.p0.z;
  var n = normalize(i.nrm);
  let L = normalize(u.sun.xyz);
  let V = normalize(u.cam.xyz - i.world);
  let uv = worldUV(i.world);
  let strata = rockColor(strataU(uv, i.h, u.st));

  // --- суша ---
  let t = clamp((i.h - sea) / max(1.0 - sea, 0.001), 0.0, 1.0);
  let slope = 1.0 - n.y;
  let sand  = u.palS.xyz;
  let grass = u.palG.xyz;
  let snow  = vec3f(0.95, 0.97, 1.00);
  var col: vec3f;
  if (u.opts.x > 0.5) {
    // режим Geology: чистые цвета пород
    col = strata;
  } else {
    col = mix(sand, grass, smoothstep(0.0, 0.06, t));
    col = mix(col, strata, smoothstep(0.25, 0.5, t));
    col = mix(col, strata, smoothstep(0.16, 0.34, slope));          // обнажения на крутых склонах
    col = mix(col, snow, smoothstep(0.78, 0.9, t) * (1.0 - smoothstep(0.30, 0.5, slope)) * u.palG.w);
  }
  if (i.h < sea) {
    col = mix(sand * 0.6, vec3f(0.12, 0.2, 0.25), clamp((sea - i.h) * 6.0, 0.0, 1.0));
  }

  // --- детали поверхности: шум-альбедо, микро-рельеф, мокрый песок у воды ---
  let rainK = u.cam.w;
  var wetK = 0.0;
  if (i.h >= sea && i.depth < 0.002 && u.opts.x < 0.5) {
    let nz = vnoise(i.world.xz * 2.6) * 0.55 + vnoise(i.world.xz * 8.5) * 0.3 + vnoise(i.world.xz * 26.0) * 0.15;
    col *= 0.78 + 0.44 * nz;
    let bump = vec2f(vnoise(i.world.xz * 11.0 + 13.0), vnoise(i.world.xz * 11.0 + 71.0)) - vec2f(0.5);
    n = normalize(n + vec3f(bump.x, 0.0, bump.y) * 0.45);
    wetK = (1.0 - smoothstep(0.0, 0.016, i.h - sea)) + smoothstep(0.0005, 0.004, i.wd) * 0.7;
    wetK = clamp(wetK + rainK * 0.35, 0.0, 1.0);
    col *= mix(1.0, 0.55, wetK);
  }

  // --- вода ---
  let time = u.opts2.x;
  let depth = max(i.depth, 0.0);
  let alpha = 1.0 - exp(-depth * 60.0);
  let shallow = vec3f(0.12, 0.50, 0.62);
  let deep = vec3f(0.02, 0.10, 0.25);
  let wcol = mix(shallow, deep, clamp(depth * 6.0, 0.0, 1.0));
  // мутная вода быстрых потоков: взвесь окрашивает воду в охристый цвет
  let turb = smoothstep(0.25, 1.1, length(i.fx.zw)) * smoothstep(0.0, 0.02, depth) * (1.0 - clamp(depth * 14.0, 0.0, 0.7));
  let wcolT = mix(wcol, vec3f(0.42, 0.30, 0.16), turb * 0.75);
  col = mix(col, wcolT, alpha * 0.92);

  // рябь: лёгкое возмущение нормали поверхности воды
  var wn = n;
  if (alpha > 0.02) {
    let rp = i.world.xz;
    let g = vec2f(
      cos(rp.x * 1.9 + time * 1.2) * 0.5 + cos((rp.x + rp.y) * 3.3 - time * 1.7) * 0.35 + cos(rp.y * 0.7 + time * 0.6) * 0.3,
      cos(rp.y * 2.1 - time * 1.0) * 0.5 + cos((rp.x - rp.y) * 2.9 + time * 1.5) * 0.35 + cos(rp.x * 0.8 - time * 0.7) * 0.3
    );
    wn = normalize(n + vec3f(g.x, 0.0, g.y) * 0.05);
  }

  let sunCol = u.sunC.xyz * u.sunC.w;
  let ambCol = (mix(u.skyH.xyz, u.skyZ.xyz, 0.5) * 0.9 + vec3f(0.26)) * 1.25;
  let diff = max(dot(wn, L), 0.0) * i.lt.x * cloudShade(i.world);
  let hemi = 0.30 + 0.30 * wn.y;
  var lit = col * (diff * sunCol + hemi * ambCol * (0.30 + 0.70 * i.lt.y));

  // мокрый песок блестит
  if (wetK > 0.0) {
    lit += sunCol * pow(max(dot(reflect(-L, n), V), 0.0), 28.0) * wetK * 0.22 * i.lt.x;
  }
  // блёстки снега: у каждой ячейки своя микро-грань, ловит солнце под определённым углом
  let snowW = smoothstep(0.78, 0.9, t) * (1.0 - smoothstep(0.30, 0.5, slope)) * u.palG.w;
  if (snowW > 0.05 && i.depth < 0.002 && u.opts.x < 0.5) {
    let cell = floor(i.world.xz * 14.0);
    let fc = vec3f(hash21(cell) - 0.5, 1.2, hash21(cell + 17.0) - 0.5);
    let g = pow(max(dot(normalize(L + V), normalize(fc)), 0.0), 380.0);
    let tw = 0.55 + 0.45 * sin(u.opts2.x * 3.0 + hash21(cell + 5.0) * 40.0);
    lit += vec3f(1.0, 0.97, 0.9) * g * tw * 7.0 * snowW * i.lt.x * u.sunC.w;
  }

  // каустика на дне мелководья
  if (depth > 0.0005 && u.atm.y < 0.5) {
    let cau = caustic(i.world.xz, time);
    let k = smoothstep(0.0, 0.012, depth) * exp(-depth * 9.0);
    lit += vec3f(0.5, 0.85, 1.0) * cau * k * 0.45 * alpha;
  }

  // отражение неба (Fresnel) + блик солнца
  let cv = clamp(dot(wn, V), 0.0, 1.0);
  let F = 0.06 + 0.94 * pow(1.0 - cv, 5.0);
  var Rv = reflect(-V, wn);
  Rv.y = abs(Rv.y);
  let deepW = smoothstep(0.004, 0.03, depth);
  lit = mix(lit, skyColor(Rv), clamp(F * deepW * 1.15, 0.0, 0.9));

  // пена: быстрые мелкие потоки и береговая линия моря
  let spd = length(vec2f(i.fx.z, i.fx.w));
  let fn1 = foamNoise(i.world.xz * 2.2, time);
  var foam = smoothstep(0.45, 1.0, spd) * smoothstep(0.003, 0.012, i.wd) * (1.0 - smoothstep(0.03, 0.08, i.wd)) * smoothstep(0.35, 0.75, fn1);
  if (i.h < sea + 0.004 && !noWater()) {
    let sd = sea - i.h;
    let band = 1.0 - smoothstep(0.0, 0.012, abs(sd - 0.004 - 0.003 * sin(time * 1.4 + i.world.x * 0.4 + i.world.z * 0.3)));
    foam += band * smoothstep(0.3, 0.7, fn1) * 0.9;
  }
  lit = mix(lit, vec3f(0.95, 0.98, 1.0), clamp(foam, 0.0, 1.0) * 0.85 * select(1.0, 0.0, u.atm.z > 0.5));

  // круги от капель дождя на воде
  if (rainK > 0.01 && alpha > 0.3) {
    let gp = i.world.xz * 2.4;
    let gc = floor(gp);
    let ph = fract(time * 0.8 + hash21(gc) * 5.0);
    let rr = length(fract(gp) - vec2f(0.5));
    let ring = (1.0 - smoothstep(0.0, 0.05, abs(rr - ph * 0.45))) * (1.0 - ph) * step(0.45, hash21(gc + 3.0));
    lit += vec3f(0.75, 0.85, 1.0) * ring * 0.45 * rainK * alpha;
  }

  // --- аналитические оверлеи ---
  let ov = select(i32(u.opts2.y), 0, noWater());
  if (ov == 1 || ov == 3) {
    let a = tanh((select(i.fx.y, i.fx.x, ov == 1)) * 10.0);
    let hc = select(vec3f(0.15, 0.45, 1.0), vec3f(1.0, 0.18, 0.10), a < 0.0);
    lit = mix(lit * 0.45, hc, clamp(abs(a) * 1.6, 0.0, 0.92));
  } else if (ov == 2) {
    let v = vec2f(i.fx.z, i.fx.w);
    let sp = length(v);
    let wet = smoothstep(0.006, 0.022, i.wd) * clamp(sp * 12.0, 0.0, 1.0);
    let dir = v / max(sp, 1e-4);
    let phase = dot(i.world.xz, dir) * 1.6 - u.opts2.x * (1.5 + sp * 8.0);
    let stripe = smoothstep(0.55, 0.95, fract(phase));
    let ramp = mix(vec3f(0.05, 0.25, 0.7), vec3f(0.9, 1.0, 1.0), clamp(sp / 0.9, 0.0, 1.0));
    lit = mix(lit * 0.5, ramp * (0.55 + 0.9 * stripe), wet);
  }

  // --- курсор кисти ---
  if (u.brush.w > 0.5) {
    let bd = length(i.world.xz - u.brush.xy);
    let ring = 1.0 - smoothstep(0.0, 0.22, abs(bd - u.brush.z));
    lit += vec3f(0.15, 0.85, 1.0) * ring * 0.9;
    lit += vec3f(0.03, 0.12, 0.18) * step(bd, u.brush.z);
  }
  // --- превью плотины ---
  if (u.opts2.z > 0.5) {
    let a = u.seg.xy; let ab = u.seg.zw - a;
    let t = clamp(dot(i.world.xz - a, ab) / max(dot(ab, ab), 1e-4), 0.0, 1.0);
    let sd = length(i.world.xz - (a + ab * t));
    lit += vec3f(1.0, 0.85, 0.2) * (1.0 - smoothstep(0.2, 0.5, sd)) * 0.9;
  }
  // --- источники рек ---
  for (var k = 0; k < 8; k++) {
    let sr = u.src[k];
    if (sr.z > 0.5) {
      let sdist = length(i.world.xz - sr.xy);
      let pulse = 1.3 + 0.35 * sin(u.opts2.x * 3.0 + f32(k));
      lit += vec3f(1.0, 0.55, 0.1) * (1.0 - smoothstep(0.6, 1.0, sdist)) * 0.9;
      lit += vec3f(1.0, 0.7, 0.3) * (1.0 - smoothstep(0.0, 0.18, abs(sdist - pulse))) * 0.7;
    }
  }

  let dist = length(i.world - u.cam.xyz);
  let fog = 1.0 - exp(-dist * u.atm.x);
  lit = mix(lit, u.skyH.xyz * 0.85, fog);
  return vec4f(lit, 1.0);
}

// ───────────────────────── стенки diorama / срез ─────────────────────────
// instance 0: x=-W/2, 1: z=-W/2, 2: z=+W/2, 3: плоскость среза x=cutX
struct WOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) hTop: f32,
  @location(2) @interpolate(flat) nrm: vec3f,
};

@vertex
fn vsWall(@builtin(vertex_index) vi: u32, @builtin(instance_index) wall: u32) -> WOut {
  let n = i32(u.p0.x);
  let c = i32(vi / 2u);
  let top = (vi % 2u) == 0u;
  let W = u.p0.w;
  let t = f32(c) / f32(n - 1);
  var x = 0.0; var z = 0.0; var h = 0.0; var wt = 0.0;
  var nrm = vec3f(1.0, 0.0, 0.0);
  if (wall == 0u) {
    let i = cid(vec2i(0, c));
    x = -0.5 * W; z = (t - 0.5) * W; h = Ht(i); wt = Wt(i); nrm = vec3f(-1.0, 0.0, 0.0);
  } else if (wall == 1u) {
    let i = cid(vec2i(c, 0));
    x = (t - 0.5) * W; z = -0.5 * W; h = Ht(i); wt = Wt(i); nrm = vec3f(0.0, 0.0, -1.0);
  } else if (wall == 2u) {
    let i = cid(vec2i(c, n - 1));
    x = (t - 0.5) * W; z = 0.5 * W; h = Ht(i); wt = Wt(i); nrm = vec3f(0.0, 0.0, 1.0);
  } else {
    x = u.opts.y; z = (t - 0.5) * W;
    let fx = clamp((x / W + 0.5) * f32(n - 1), 0.0, f32(n - 1));
    let ix = i32(floor(fx));
    let fr = fx - floor(fx);
    let a = cid(vec2i(ix, c));
    let b = cid(vec2i(ix + 1, c));
    h = mix(Ht(a), Ht(b), fr);
    wt = mix(Wt(a), Wt(b), fr);
  }
  let surf = max(h + wt, u.p0.z);
  let y = select(-u.opts.w, surf * u.p0.y, top);
  var o: WOut;
  o.world = vec3f(x, y, z);
  o.pos = u.vp * vec4f(o.world, 1.0);
  o.hTop = h;
  o.nrm = nrm;
  return o;
}

@fragment
fn fsWall(i: WOut) -> @location(0) vec4f {
  if (i.world.x > u.opts.y + 0.002) { discard; }
  let hs = u.p0.y;
  let L = normalize(u.sun.xyz);
  let z = i.world.y / hs;
  let uv = worldUV(i.world);
  var col = rockColor(strataU(uv, z, u.st));

  // над землёй, но ниже поверхности — вода
  let groundY = i.hTop * hs;
  if (i.world.y > groundY + 0.02) {
    let d = clamp((u.p0.z * hs - i.world.y) * 0.25 + 0.2, 0.0, 1.0);
    col = mix(vec3f(0.14, 0.55, 0.68), vec3f(0.03, 0.14, 0.30), d);
  }

  let shade = 0.55 + 0.45 * max(dot(i.nrm, L), 0.0);
  let below = clamp(-i.world.y / max(u.opts.w, 0.001), 0.0, 1.0);
  var lit = col * shade * (0.45 + 0.55 * u.sunC.w) * mix(1.0, 0.45, below);
  let dist = length(i.world - u.cam.xyz);
  lit = mix(lit, u.skyH.xyz, 1.0 - exp(-dist * u.atm.x));
  return vec4f(lit, 1.0);
}


// ───────────────────────── деревья (процедурный инстансинг) ─────────────────────────
// Каждый инстанс — кандидат: хэш → клетка; живёт, только если там суша, нет воды, полого и подходит биом.
struct TOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) col: vec3f,
  @location(2) nrm: vec3f,
  @location(3) lt: vec2f,
};

fn hash1(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return f32(x) / 4294967295.0;
}

@vertex
fn vsTree(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> TOut {
  var o: TOut;
  o.pos = vec4f(0.0, 0.0, 2.0, 1.0); // за пределами клипа — инстанс отброшен
  o.world = vec3f(0.0); o.col = vec3f(0.0); o.nrm = vec3f(0.0, 1.0, 0.0); o.lt = vec2f(1.0, 1.0);
  let dens = u.atm.w;
  if (dens <= 0.0) { return o; }
  let n = u.p0.x;
  let fcx = 6.0 + hash1(inst * 4u + 0u) * (n - 13.0);
  let fcy = 6.0 + hash1(inst * 4u + 1u) * (n - 13.0);
  let c = vec2i(vec2f(fcx, fcy));
  let i = cid(c);
  let h = Ht(i);
  let sea = u.p0.z;
  let t = (h - sea) / max(1.0 - sea, 0.001);
  if (h < sea + 0.012 || Wt(i) > 0.0015 || t > 0.7) { return o; }
  let world = u.p0.w;
  let step = world / (n - 1.0);
  let dx = (Ht(cid(c + vec2i(1, 0))) - Ht(cid(c - vec2i(1, 0)))) * u.p0.y;
  let dz = (Ht(cid(c + vec2i(0, 1))) - Ht(cid(c - vec2i(0, 1)))) * u.p0.y;
  let nrm = normalize(vec3f(-dx, 2.0 * step, -dz));
  let slope = 1.0 - nrm.y;
  let x = (fcx / (n - 1.0) - 0.5) * world;
  let z = (fcy / (n - 1.0) - 0.5) * world;
  // рощи: низкочастотная маска
  let grove = 0.5 + 0.5 * sin(x * 0.19 + sin(z * 0.11) * 2.0) * sin(z * 0.17 - cos(x * 0.13) * 2.0);
  let keep = dens * (0.25 + 0.9 * grove) * (1.0 - smoothstep(0.35, 0.65, t)) * (1.0 - smoothstep(0.07, 0.16, slope))
           * smoothstep(0.012, 0.05, h - sea);
  if (hash1(inst * 4u + 2u) > keep) { return o; }
  // немного тоньше у сильно размытых клеток: русло и обнажение породы без деревьев
  if (bitcast<f32>(acc[i]) < -0.25) { return o; }

  let sz = 0.5 + 0.4 * hash1(inst * 4u + 3u);
  let k = vi / 15u;                // 0 нижний конус, 1 верхний
  let j = (vi % 15u) / 3u;         // грань
  let cc = vi % 3u;                // 0 вершина, 1/2 — кольцо
  var y0 = 0.35 * sz; var y1 = 1.9 * sz; var r = 0.62 * sz;
  if (k == 1u) { y0 = 1.05 * sz; y1 = 2.7 * sz; r = 0.42 * sz; }
  let a0 = (f32(j) + select(0.0, 1.0, cc == 2u)) * 1.2566371 + f32(inst % 7u);
  var local = vec3f(0.0, y1, 0.0);
  if (cc != 0u) { local = vec3f(cos(a0) * r, y0, sin(a0) * r); }
  let am = (f32(j) + 0.5) * 1.2566371 + f32(inst % 7u);
  let base = vec3f(x, h * u.p0.y - 0.05, z);
  o.world = base + local;
  o.pos = u.vp * vec4f(o.world, 1.0);
  o.nrm = normalize(vec3f(cos(am), 0.55, sin(am)));
  let v = hash1(inst * 7u + 11u);
  let dry = clamp(1.0 - u.palG.w * 0.0, 0.0, 1.0);
  var g = mix(vec3f(0.05, 0.22, 0.09), vec3f(0.12, 0.34, 0.12), v) * mix(0.75, 1.15, local.y / (2.7 * sz));
  g = mix(g, u.palG.xyz * 0.55, 0.35) * dry;
  o.col = g;
  o.lt = lightBuf[i];
  return o;
}

@fragment
fn fsTree(i: TOut) -> @location(0) vec4f {
  if (i.world.x > u.opts.y) { discard; }
  let L = normalize(u.sun.xyz);
  let sunCol = u.sunC.xyz * u.sunC.w;
  let ambCol = (mix(u.skyH.xyz, u.skyZ.xyz, 0.5) * 0.9 + vec3f(0.26)) * 1.25;
  let nn = normalize(i.nrm);
  let diff = max(dot(nn, L), 0.0) * i.lt.x * cloudShade(i.world);
  var lit = i.col * (diff * sunCol + (0.35 + 0.25 * nn.y) * ambCol * (0.30 + 0.70 * i.lt.y));
  let dist = length(i.world - u.cam.xyz);
  lit = mix(lit, u.skyH.xyz * 0.85, 1.0 - exp(-dist * u.atm.x));
  return vec4f(lit, 1.0);
}

// ───────────────────────── небо ─────────────────────────
struct SkyOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn vsSky(@builtin(vertex_index) vi: u32) -> SkyOut {
  let c = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: SkyOut;
  o.ndc = c * 2.0 - 1.0;
  o.pos = vec4f(o.ndc, 1.0, 1.0);
  return o;
}

@fragment
fn fsSky(i: SkyOut) -> @location(0) vec4f {
  let dir = normalize(u.camF.xyz + u.camR.xyz * (i.ndc.x * u.camU.w * u.camF.w) + u.camU.xyz * (i.ndc.y * u.camU.w));
  var col = skyColor(dir);
  if (dir.y > 0.015 && u.cam.y < CLOUD_H) {
    let tt = (CLOUD_H - u.cam.y) / dir.y;
    let d = cloudDens(u.cam.xz + dir.xz * tt, u.opts2.x);
    let fade = exp(-tt * 0.0035);
    let cc = u.skyH.xyz * 0.5 + u.sunC.xyz * u.sunC.w * 0.5 + vec3f(0.22);
    col = mix(col, cc, d * 0.88 * fade);
  }
  return vec4f(col, 1.0);
}

// ───────────────────────── дождь: полосы вокруг камеры ─────────────────────────
struct ROut {
  @builtin(position) pos: vec4f,
  @location(0) a: f32,
};

@vertex
fn vsRain(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> ROut {
  var o: ROut;
  o.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  o.a = 0.0;
  let rk = u.cam.w;
  if (rk < 0.01) { return o; }
  let fi = f32(inst);
  let ctr = u.cam.xyz + u.camF.xyz * 38.0;
  let rx = (hash21(vec2f(fi, 1.0)) - 0.5) * 90.0;
  let rz = (hash21(vec2f(fi, 2.0)) - 0.5) * 90.0;
  let spd = 38.0 + hash21(vec2f(fi, 3.0)) * 10.0;
  let fall = fract(hash21(vec2f(fi, 4.0)) - u.opts2.x * spd / 80.0);
  let y = fall * 80.0 + ctr.y - 35.0;
  var p = vec3f(ctr.x + rx, y, ctr.z + rz);
  if (vi == 1u) { p += vec3f(0.12, 1.5, 0.05); }
  o.pos = u.vp * vec4f(p, 1.0);
  o.a = rk * select(0.5, 0.0, vi == 1u) * smoothstep(0.0, 0.08, fall) * (1.0 - smoothstep(0.85, 1.0, fall));
  if (hash21(vec2f(fi, 5.0)) > rk) { o.pos = vec4f(0.0, 0.0, 2.0, 1.0); }
  return o;
}

@fragment
fn fsRain(i: ROut) -> @location(0) vec4f {
  let c = mix(vec3f(0.7, 0.8, 1.0), u.sunC.xyz, 0.3) * (0.5 + 0.5 * u.sunC.w);
  return vec4f(c, 0.3 + i.a);
}
`;

/** f16 = true: буфер q хранит пары half (читаем как u32), иначе vec2f. */
export function terrainWgsl(f16: boolean): string {
  const qDecl = f16
    ? `@group(0) @binding(5) var<storage, read> qbuf: array<u32>;
fn readQ(i: u32) -> vec2f { return unpack2x16float(qbuf[i]); }`
    : `@group(0) @binding(5) var<storage, read> qbuf: array<vec2f>;
fn readQ(i: u32) -> vec2f { return qbuf[i]; }`;
  return BODY.replace('//Q_DECL//', qDecl) + STRATA_WGSL;
}
