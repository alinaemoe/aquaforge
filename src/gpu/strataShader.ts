// Слоистая геология: тип породы — аналитическая функция (uv, высота), без доп. буферов.
// Пласты горизонтальны в нормализованной высоте z, искривлены шумом и наклонены (dip).
// Эрозия вскрывает разные слои по мере понижения поверхности.
//
// st = vec4(scale: пластов на единицу высоты, warp: амплитуда изгиба, dipX, dipY)
// Типы: 0 глина (мягкая), 1 песчаник, 2 известняк, 3 базальт (твёрдый).

export const STRATA_WGSL = /* wgsl */ `
fn sh1(x: f32) -> f32 { return fract(sin(x * 12.9898 + 4.1414) * 43758.5453); }
fn sh2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453123); }
fn svn(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(sh2(i), sh2(i + vec2f(1.0, 0.0)), u.x),
             mix(sh2(i + vec2f(0.0, 1.0)), sh2(i + vec2f(1.0, 1.0)), u.x), u.y);
}

// координата внутри стопки пластов: целая часть = номер пласта
fn strataU(uv: vec2f, z: f32, st: vec4f) -> f32 {
  let off = (svn(uv * 3.0) * 0.65 + svn(uv * 8.0) * 0.35 - 0.5) * st.y
          + (uv.x - 0.5) * st.z + (uv.y - 0.5) * st.w;
  return (z + off) * st.x;
}

fn rockType(u: f32) -> u32 {
  let r = sh1(floor(u));
  if (r < 0.30) { return 0u; }
  if (r < 0.62) { return 1u; }
  if (r < 0.84) { return 2u; }
  return 3u;
}

fn rockColor(u: f32) -> vec3f {
  var cols = array<vec3f, 4>(
    vec3f(0.62, 0.29, 0.21),  // глина
    vec3f(0.80, 0.62, 0.38),  // песчаник
    vec3f(0.84, 0.82, 0.72),  // известняк
    vec3f(0.19, 0.19, 0.22)   // базальт
  );
  let t = rockType(u);
  let tint = 0.86 + 0.28 * sh1(floor(u) + 31.7);
  let edge = 0.90 + 0.10 * smoothstep(0.0, 0.18, fract(u)); // тонкая тень на подошве пласта
  return cols[t] * tint * edge;
}
`;
