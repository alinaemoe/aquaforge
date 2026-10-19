// Экспорт 3D-сетки ландшафта: OBJ и бинарный glTF (.glb).
// Единицы — мир приложения (100 × 100 по горизонтали, высота = h × heightScale).

export interface MeshData {
  positions: Float32Array; // xyz
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  vertsPerSide: number;
}

/** Сетка из heightmap n×n, прореженная в `step` раз (1 = полное разрешение). */
export function buildMesh(heights: Float32Array, n: number, step: number, heightScale: number, world: number): MeshData {
  const m = Math.floor((n - 1) / step) + 1;
  const positions = new Float32Array(m * m * 3);
  const normals = new Float32Array(m * m * 3);
  const uvs = new Float32Array(m * m * 2);
  const cell = world / (n - 1);
  const H = (x: number, y: number) => heights[Math.min(n - 1, Math.max(0, y)) * n + Math.min(n - 1, Math.max(0, x))];

  for (let j = 0; j < m; j++) {
    for (let i = 0; i < m; i++) {
      const x = Math.round((i * (n - 1)) / (m - 1));
      const y = Math.round((j * (n - 1)) / (m - 1));
      const v = j * m + i;
      positions[v * 3] = (x / (n - 1) - 0.5) * world;
      positions[v * 3 + 1] = H(x, y) * heightScale;
      positions[v * 3 + 2] = (y / (n - 1) - 0.5) * world;
      const dx = (H(x + step, y) - H(x - step, y)) * heightScale;
      const dz = (H(x, y + step) - H(x, y - step)) * heightScale;
      const nx = -dx, ny = 2 * cell * step, nz = -dz;
      const l = Math.hypot(nx, ny, nz) || 1;
      normals[v * 3] = nx / l;
      normals[v * 3 + 1] = ny / l;
      normals[v * 3 + 2] = nz / l;
      uvs[v * 2] = i / (m - 1);
      uvs[v * 2 + 1] = 1 - j / (m - 1);
    }
  }
  const indices = new Uint32Array((m - 1) * (m - 1) * 6);
  let k = 0;
  for (let j = 0; j < m - 1; j++) {
    for (let i = 0; i < m - 1; i++) {
      const a = j * m + i, b = a + 1, c = a + m, d = c + 1;
      // обход против часовой стрелки при взгляде сверху (+Y)
      indices[k++] = a; indices[k++] = c; indices[k++] = b;
      indices[k++] = b; indices[k++] = c; indices[k++] = d;
    }
  }
  return { positions, normals, uvs, indices, vertsPerSide: m };
}

export function toOBJ(mesh: MeshData): Blob {
  const { positions, normals, uvs, indices } = mesh;
  const parts: string[] = ['# AQUAFORGE terrain\no terrain\n'];
  let buf: string[] = [];
  const flush = () => {
    parts.push(buf.join(''));
    buf = [];
  };
  for (let i = 0; i < positions.length; i += 3) {
    buf.push(`v ${positions[i].toFixed(4)} ${positions[i + 1].toFixed(4)} ${positions[i + 2].toFixed(4)}\n`);
    if (buf.length > 20000) flush();
  }
  for (let i = 0; i < uvs.length; i += 2) {
    buf.push(`vt ${uvs[i].toFixed(5)} ${uvs[i + 1].toFixed(5)}\n`);
    if (buf.length > 20000) flush();
  }
  for (let i = 0; i < normals.length; i += 3) {
    buf.push(`vn ${normals[i].toFixed(4)} ${normals[i + 1].toFixed(4)} ${normals[i + 2].toFixed(4)}\n`);
    if (buf.length > 20000) flush();
  }
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] + 1, b = indices[i + 1] + 1, c = indices[i + 2] + 1;
    buf.push(`f ${a}/${a}/${a} ${b}/${b}/${b} ${c}/${c}/${c}\n`);
    if (buf.length > 20000) flush();
  }
  flush();
  return new Blob(parts, { type: 'text/plain' });
}

export function toGLB(mesh: MeshData): Blob {
  const { positions, normals, uvs, indices } = mesh;
  const pad4 = (n: number) => (n + 3) & ~3;

  const posBytes = positions.byteLength, nrmBytes = normals.byteLength;
  const uvBytes = uvs.byteLength, idxBytes = indices.byteLength;
  const total = posBytes + nrmBytes + uvBytes + idxBytes; // все кратны 4
  const bin = new Uint8Array(pad4(total));
  let off = 0;
  const offs: number[] = [];
  for (const arr of [positions, normals, uvs, indices]) {
    offs.push(off);
    bin.set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength), off);
    off += arr.byteLength;
  }

  let minP = [Infinity, Infinity, Infinity], maxP = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      minP[a] = Math.min(minP[a], positions[i + a]);
      maxP[a] = Math.max(maxP[a], positions[i + a]);
    }
  }

  const json = {
    asset: { version: '2.0', generator: 'AQUAFORGE' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: 'terrain' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0, mode: 4 }] }],
    materials: [{ name: 'terrain', pbrMetallicRoughness: { baseColorFactor: [0.55, 0.5, 0.42, 1], metallicFactor: 0, roughnessFactor: 1 } }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3', min: minP, max: maxP },
      { bufferView: 1, componentType: 5126, count: normals.length / 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: uvs.length / 2, type: 'VEC2' },
      { bufferView: 3, componentType: 5125, count: indices.length, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: offs[0], byteLength: posBytes, target: 34962 },
      { buffer: 0, byteOffset: offs[1], byteLength: nrmBytes, target: 34962 },
      { buffer: 0, byteOffset: offs[2], byteLength: uvBytes, target: 34962 },
      { buffer: 0, byteOffset: offs[3], byteLength: idxBytes, target: 34963 },
    ],
    buffers: [{ byteLength: bin.length }],
  };
  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPadded = new Uint8Array(pad4(jsonBytes.length)).fill(0x20); // пробелы
  jsonPadded.set(jsonBytes);

  const header = new DataView(new ArrayBuffer(12));
  header.setUint32(0, 0x46546c67, true); // 'glTF'
  header.setUint32(4, 2, true);
  header.setUint32(8, 12 + 8 + jsonPadded.length + 8 + bin.length, true);
  const jh = new DataView(new ArrayBuffer(8));
  jh.setUint32(0, jsonPadded.length, true);
  jh.setUint32(4, 0x4e4f534a, true); // 'JSON'
  const bh = new DataView(new ArrayBuffer(8));
  bh.setUint32(0, bin.length, true);
  bh.setUint32(4, 0x004e4942, true); // 'BIN\0'
  return new Blob([header.buffer, jh.buffer, jsonPadded, bh.buffer, bin], { type: 'model/gltf-binary' });
}
