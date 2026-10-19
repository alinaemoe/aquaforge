// Постпроцесс: bloom (яркие области → 1/4 разрешения → гауссово размытие) + мягкий tonemap + виньетка.
// Все проходы используют один bind group layout: sampler, tex, uniform P, tex2.

export const POST_WGSL = /* wgsl */ `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> P: vec4f;
@group(0) @binding(3) var tex2: texture_2d<f32>;

struct V {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> V {
  let c = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: V;
  o.pos = vec4f(c * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(c.x, 1.0 - c.y);
  return o;
}

// P.xy = размер texel источника, P.z = порог яркости
@fragment
fn bright(i: V) -> @location(0) vec4f {
  let ts = P.xy;
  var c = vec3f(0.0);
  for (var y = -1; y <= 1; y += 2) {
    for (var x = -1; x <= 1; x += 2) {
      c += textureSample(tex, samp, i.uv + vec2f(f32(x), f32(y)) * ts * 1.5).rgb;
    }
  }
  c *= 0.25;
  let l = max(c.r, max(c.g, c.b));
  let k = smoothstep(P.z, P.z + 0.5, l);
  return vec4f(c * k, 1.0);
}

// P.xy = шаг в texel вдоль направления размытия
@fragment
fn blur(i: V) -> @location(0) vec4f {
  var w = array<f32, 5>(0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);
  var c = textureSample(tex, samp, i.uv).rgb * w[0];
  for (var k = 1; k < 5; k++) {
    let o = P.xy * f32(k) * 1.4;
    c += textureSample(tex, samp, i.uv + o).rgb * w[k];
    c += textureSample(tex, samp, i.uv - o).rgb * w[k];
  }
  return vec4f(c, 1.0);
}

// P.x = сила bloom, P.y = сила виньетки
@fragment
fn composite(i: V) -> @location(0) vec4f {
  let s = textureSample(tex, samp, i.uv).rgb;
  let b = textureSample(tex2, samp, i.uv).rgb;
  var c = s + b * P.x;
  // мягкое «плечо» выше 0.8, чтобы блики не клиповались жёстко
  let sh = vec3f(0.8) + 0.2 * (vec3f(1.0) - exp(-(c - vec3f(0.8)) / 0.2));
  c = select(c, sh, c > vec3f(0.8));
  let d = length(i.uv - vec2f(0.5));
  c *= 1.0 - P.y * smoothstep(0.35, 0.85, d);
  return vec4f(c, 1.0);
}
`;
