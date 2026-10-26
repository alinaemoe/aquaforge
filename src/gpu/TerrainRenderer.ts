import { NOISE_WGSL } from './shaders';
import { terrainWgsl } from './terrainShader';
import { simWgsl } from './simShader';
import { PICK_WGSL, TOOLS_WGSL } from './toolsShader';
import { POST_WGSL } from './postShader';
import { LIGHT_WGSL } from './lightShader';
import { CLIMATES, type ClimateId } from '../climate';
import { lookAt, mul, perspective, type Vec3 } from './math';
import {
  DEFAULT_GEO,
  DEFAULT_TOOL,
  type GeoParams,
  type Overlay,
  type SimParams,
  type TerrainParams,
  type Tool,
  type ToolParams,
} from '../store';

const GRID = 512; // клеток по стороне
const WORLD = 100; // размер мира в юнитах
const CELL = WORLD / (GRID - 1);
const CELLS = GRID * GRID;
const WG = GRID / 16; // сим-проходы: рабочая группа 16×16
const WG8 = GRID / 8; // генерация шума: 8×8
const FOV = Math.PI / 4;
const MAX_SOURCES = 8;

const TREE_CANDIDATES = 90000; // инстансов-кандидатов; живые выбираются в вершинном шейдере
const SCENE_FMT: GPUTextureFormat = 'rgba16float'; // HDR-сцена для bloom
const UBO_FLOATS = 36 + 12 + 32 + 32 + 4; // базовая часть + src[8] + камера/атмосфера/палитры

const SNAP_MAX = 96; // кадров time-lapse в GPU-кольце (96 × 1 МБ)

const SIM_ENTRIES = ['calcFlux', 'updateWater', 'erode', 'advect', 'thermal'] as const;

interface Hit {
  valid: boolean;
  x: number; // клетка
  y: number;
  h: number; // высота рельефа 0..1
}

export interface Source {
  x: number; // клетка
  y: number;
  rate: number;
}

interface CamKey {
  eye: Vec3;
  target: Vec3;
}

const cellToWorld = (c: number) => (c / (GRID - 1) - 0.5) * WORLD;

export class TerrainRenderer {
  private device!: GPUDevice;
  f16 = false;
  private ctx!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private depthTex?: GPUTexture;

  // буферы симуляции
  private tA!: GPUBuffer;
  private tB!: GPUBuffer;
  private water!: GPUBuffer;
  private sedA!: GPUBuffer;
  private sedB!: GPUBuffer;
  private flux!: GPUBuffer;
  private qBuf!: GPUBuffer;
  private acc!: GPUBuffer; // накопленная эрозия + счётчик шагов
  private h0!: GPUBuffer; // исходный рельеф (для Δвысоты)
  private simUBO!: GPUBuffer;

  private noisePipeline!: GPUComputePipeline;
  private noiseBG!: GPUBindGroup;
  private noiseUBO!: GPUBuffer;
  private simPipelines: GPUComputePipeline[] = [];
  private simBG!: GPUBindGroup;
  private renderPipeline!: GPURenderPipeline;
  private wallPipeline!: GPURenderPipeline;
  private treePipeline!: GPURenderPipeline;
  private rainPipeline!: GPURenderPipeline;
  private rainVis = 0;
  private renderBG!: GPUBindGroup;
  private renderUBO!: GPUBuffer;
  private renderUBO2!: GPUBuffer; // «до» для split-view
  private renderBG2!: GPUBindGroup;
  private renderBGL!: GPUBindGroupLayout;
  private skyPipeline!: GPURenderPipeline;

  // постпроцесс (bloom)
  private sceneTex?: GPUTexture;
  private bloomA?: GPUTexture;
  private bloomB?: GPUTexture;
  private sceneView!: GPUTextureView;
  private postBGL!: GPUBindGroupLayout;
  private postSampler!: GPUSampler;
  private pipeBright!: GPURenderPipeline;
  private pipeBlur!: GPURenderPipeline;
  private pipeComp!: GPURenderPipeline;
  private postUbo: GPUBuffer[] = []; // bright, blurH, blurV, composite
  private postBG: GPUBindGroup[] = [];

  climate: ClimateId = 'temperate';
  /** DEV: флаги отладки шейдера (отключение каустики, пены). */
  dbg: [number, number] = [0, 0];
  fx = { fog: 0.0035, bloom: 0.6, tod: 0.35, trees: 1 };

  // освещение: тени + AO (compute), считаются при смене рельефа или солнца
  private lightBuf!: GPUBuffer;
  private lightBuf0!: GPUBuffer; // для исходного рельефа (Before)
  private lightPipeline!: GPUComputePipeline;
  private lightBG!: GPUBindGroup;
  private lightBG0!: GPUBindGroup;
  private lightUBO!: GPUBuffer;
  private lightDirty = true;
  private lightAge = 0;
  private lastSunKey = '';
  private encodeLightBefore = false;

  // time-lapse
  private snapBuf!: GPUBuffer;
  private snapBGs = new Map<number, GPUBindGroup>();
  private snapStepAt = new Array<number>(SNAP_MAX).fill(0);
  private snapTotal = 0;
  private lastSnapStep = 0;
  timelapse = { on: false, interval: 250 };
  /** Индекс кадра time-lapse для просмотра (0 — самый старый) или null = живой вид. */
  scrub: number | null = null;
  onTimelapse?: (s: { count: number; steps: number[] }) => void;

  // инструменты
  private pickPipeline!: GPUComputePipeline;
  private pickBG!: GPUBindGroup;
  private pickUBO!: GPUBuffer;
  private pickBuf!: GPUBuffer;
  private pickStage!: GPUBuffer;
  private toolPipeline!: GPUComputePipeline;
  private toolBG!: GPUBindGroup;
  private toolUBO!: GPUBuffer;

  private params!: TerrainParams;
  sim!: SimParams;
  geo: GeoParams = DEFAULT_GEO;
  tool: Tool = 'orbit';
  toolParams: ToolParams = DEFAULT_TOOL;
  overlay: Overlay = 'none';
  running = false;
  speed = 5;
  sources: Source[] = [];
  private pendingSteps = 0;
  private effSteps = 1;
  private lastFrameT = performance.now();
  private lastStepsStat = 0;
  private stepsDone = 0;

  // ввод/пикинг
  private hit: Hit = { valid: false, x: 0, y: 0, h: 0 };
  private pickReq: { ro: Vec3; rd: Vec3 } | null = null;
  private pickBusy = false;
  private toolDown = false;
  private damStart: Hit | null = null;
  private pendingDam: { a: Hit; b: Hit } | null = null;

  private raf = 0;
  private disposed = false;

  // орбитальная камера
  private yaw = 0.8;
  private pitch = 0.55;
  private dist = 140;
  private target: Vec3 = [0, 4, 0];
  private yawV = 0;
  private pitchV = 0;
  private zoomV = 0;
  private dragging = false;
  private focusGoal: Vec3 | null = null;

  // cinematic
  cineKeys: CamKey[] = [];
  private cine = { playing: false, t0: 0, duration: 20, progress: 0 };
  private recorder: MediaRecorder | null = null;
  private recChunks: Blob[] = [];
  private recTrack: CanvasCaptureMediaStreamTrack | null = null;
  private onRecorded?: (blob: Blob) => void;
  /** Положение разделителя Before/After (0..1 от ширины экрана) или null, если split выключен. */
  split: number | null = null;

  onFps?: (fps: number, stepsPerSec: number) => void;
  onCine?: (s: { keys: number; playing: boolean; recording: boolean; progress: number }) => void;
  onSteps?: (steps: number) => void;

  constructor(private canvas: HTMLCanvasElement) {}

  async init(initial: TerrainParams, sim: SimParams) {
    if (!navigator.gpu) throw new Error('WebGPU недоступен в этом браузере (нужен Chrome/Edge 113+).');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('Не найден GPU-адаптер WebGPU.');
    this.f16 = adapter.features.has('shader-f16');
    this.device = await adapter.requestDevice({
      requiredFeatures: this.f16 ? (['shader-f16'] as GPUFeatureName[]) : [],
    });
    this.device.lost.then((i) => console.error('GPU device lost:', i.message));

    this.ctx = this.canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });

    this.params = initial;
    this.sim = sim;
    this.createBuffers();
    this.createNoisePipeline();
    this.createSimPipelines();
    this.createToolPipelines();
    this.createLightPipeline();
    this.createRenderPipeline();
    this.createPostPipelines();
    this.attachControls();
    this.resize();
    this.generate(initial);
    this.loop();
  }

  private buf(size: number, extra = 0) {
    return this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra,
    });
  }

  private createBuffers() {
    const f = CELLS * 4;
    this.tA = this.buf(f, GPUBufferUsage.COPY_SRC);
    this.tB = this.buf(f);
    this.water = this.buf(f, GPUBufferUsage.COPY_SRC);
    const h = this.f16 ? 2 : 4; // байт на half-значение
    this.sedA = this.buf(CELLS * h);
    this.sedB = this.buf(CELLS * h);
    this.flux = this.buf(CELLS * h * 4);
    this.qBuf = this.buf(CELLS * h * 2, GPUBufferUsage.COPY_SRC);
    this.acc = this.buf((CELLS + 4) * 4, GPUBufferUsage.COPY_SRC);
    this.h0 = this.buf(f, GPUBufferUsage.COPY_SRC);
    this.lightBuf = this.buf(CELLS * 8);
    this.lightBuf0 = this.buf(CELLS * 8);
    this.snapBuf = this.buf(f * SNAP_MAX);
    this.simUBO = this.device.createBuffer({
      size: (24 + MAX_SOURCES * 4) * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private createNoisePipeline() {
    const module = this.device.createShaderModule({ code: NOISE_WGSL });
    this.noisePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    });
    this.noiseUBO = this.device.createBuffer({
      size: 12 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.noiseBG = this.device.createBindGroup({
      layout: this.noisePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.noiseUBO } },
        { binding: 1, resource: { buffer: this.tA } },
      ],
    });
  }

  private createSimPipelines() {
    const d = this.device;
    const bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4, 5, 6, 7, 8].map((b) => ({
          binding: b,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' as const },
        })),
      ],
    });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const module = d.createShaderModule({ code: simWgsl(this.f16) });
    this.simPipelines = SIM_ENTRIES.map((entryPoint) =>
      d.createComputePipeline({ layout, compute: { module, entryPoint } }),
    );
    this.simBG = d.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.simUBO } },
        { binding: 1, resource: { buffer: this.tA } },
        { binding: 2, resource: { buffer: this.tB } },
        { binding: 3, resource: { buffer: this.water } },
        { binding: 4, resource: { buffer: this.sedA } },
        { binding: 5, resource: { buffer: this.sedB } },
        { binding: 6, resource: { buffer: this.flux } },
        { binding: 7, resource: { buffer: this.qBuf } },
        { binding: 8, resource: { buffer: this.acc } },
      ],
    });
  }

  private createToolPipelines() {
    const d = this.device;
    // пикинг
    this.pickPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: PICK_WGSL }), entryPoint: 'pick' },
    });
    this.pickUBO = d.createBuffer({ size: 12 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pickBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.pickStage = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.pickBG = d.createBindGroup({
      layout: this.pickPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.pickUBO } },
        { binding: 1, resource: { buffer: this.tA } },
        { binding: 2, resource: { buffer: this.pickBuf } },
      ],
    });
    // кисти
    this.toolPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: TOOLS_WGSL }), entryPoint: 'apply' },
    });
    this.toolUBO = d.createBuffer({ size: 16 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.toolBG = d.createBindGroup({
      layout: this.toolPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.toolUBO } },
        { binding: 1, resource: { buffer: this.tA } },
        { binding: 2, resource: { buffer: this.water } },
      ],
    });
  }

  private createRenderPipeline() {
    const d = this.device;
    const module = d.createShaderModule({ code: terrainWgsl(this.f16) });
    const V = GPUShaderStage.VERTEX;
    const bgl = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: V | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4, 5, 6].map((b) => ({
          binding: b,
          visibility: V,
          buffer: { type: 'read-only-storage' as const },
        })),
      ],
    });
    this.renderBGL = bgl;
    const layout = d.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const depthStencil: GPUDepthStencilState = {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    };
    this.renderPipeline = d.createRenderPipeline({
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: SCENE_FMT }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil,
    });
    this.wallPipeline = d.createRenderPipeline({
      layout,
      vertex: { module, entryPoint: 'vsWall' },
      fragment: { module, entryPoint: 'fsWall', targets: [{ format: SCENE_FMT }] },
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
      depthStencil,
    });
    this.treePipeline = d.createRenderPipeline({
      layout,
      vertex: { module, entryPoint: 'vsTree' },
      fragment: { module, entryPoint: 'fsTree', targets: [{ format: SCENE_FMT }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil,
    });
    const blend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    this.rainPipeline = d.createRenderPipeline({
      layout,
      vertex: { module, entryPoint: 'vsRain' },
      fragment: { module, entryPoint: 'fsRain', targets: [{ format: SCENE_FMT, blend }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });
    this.skyPipeline = d.createRenderPipeline({
      layout,
      vertex: { module, entryPoint: 'vsSky' },
      fragment: { module, entryPoint: 'fsSky', targets: [{ format: SCENE_FMT }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
    });
    this.renderUBO = d.createBuffer({
      size: UBO_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.renderUBO2 = d.createBuffer({
      size: UBO_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const mkBG = (ubo: GPUBuffer, light: GPUBuffer) =>
      d.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: ubo } },
          { binding: 1, resource: { buffer: this.tA } },
          { binding: 2, resource: { buffer: this.water } },
          { binding: 3, resource: { buffer: this.h0 } },
          { binding: 4, resource: { buffer: this.acc } },
          { binding: 5, resource: { buffer: this.qBuf } },
          { binding: 6, resource: { buffer: light } },
        ],
      });
    this.renderBG = mkBG(this.renderUBO, this.lightBuf);
    this.renderBG2 = mkBG(this.renderUBO2, this.lightBuf0);
  }

  private createLightPipeline() {
    const d = this.device;
    this.lightPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: LIGHT_WGSL }), entryPoint: 'light' },
    });
    this.lightUBO = d.createBuffer({ size: 8 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const mk = (src: GPUBuffer, out: GPUBuffer) =>
      d.createBindGroup({
        layout: this.lightPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.lightUBO } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: out } },
        ],
      });
    this.lightBG = mk(this.tA, this.lightBuf);
    this.lightBG0 = mk(this.h0, this.lightBuf0);
  }

  /** Параметры солнца по времени суток: направление, цвет/интенсивность, теплота (закатность). */
  private sunState() {
    const tod = this.fx.tod;
    const elDeg = 4 + 46 * Math.sin(Math.PI * tod);
    const el = (elDeg * Math.PI) / 180;
    const az = 2.3 - 2.9 * tod;
    const dir: Vec3 = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
    const sm = (a: number, b: number, x: number) => {
      const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
      return t * t * (3 - 2 * t);
    };
    const warm = 1 - sm(4, 24, elDeg);
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const col: Vec3 = [lerp(1, 1, warm), lerp(0.95, 0.5, warm), lerp(0.85, 0.25, warm)];
    const intensity = lerp(0.7, 1.0, sm(2, 30, elDeg));
    const bright = 0.75 + 0.25 * sm(0, 18, elDeg);
    return { dir, col, intensity, warm, bright };
  }

  /** Пересчитать тени/AO для текущего рельефа (и при необходимости для исходного). */
  private encodeLight(enc: GPUCommandEncoder, withBefore: boolean) {
    const p = this.params;
    const sun = this.sunState();
    this.device.queue.writeBuffer(
      this.lightUBO,
      0,
      new Float32Array([GRID, p.heightScale, WORLD, p.seaLevel, sun.dir[0], sun.dir[1], sun.dir[2], 0]),
    );
    const run = (bg: GPUBindGroup) => {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.lightPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(WG8, WG8);
      pass.end();
    };
    run(this.lightBG);
    if (withBefore) run(this.lightBG0);
    this.lightDirty = false;
    this.lightAge = 0;
  }

  private createPostPipelines() {
    const d = this.device;
    const module = d.createShaderModule({ code: POST_WGSL });
    this.postBGL = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.postBGL] });
    const mk = (entry: string, format: GPUTextureFormat) =>
      d.createRenderPipeline({
        layout,
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: entry, targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
    this.pipeBright = mk('bright', SCENE_FMT);
    this.pipeBlur = mk('blur', SCENE_FMT);
    this.pipeComp = mk('composite', this.format);
    this.postSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    for (let k = 0; k < 4; k++) {
      this.postUbo.push(d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    }
  }

  /** Пересоздать HDR-сцену и bloom-текстуры под размер канваса. */
  private createPostTargets(w: number, h: number) {
    const d = this.device;
    this.sceneTex?.destroy();
    this.bloomA?.destroy();
    this.bloomB?.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const bw = Math.max(1, Math.floor(w / 4));
    const bh = Math.max(1, Math.floor(h / 4));
    this.sceneTex = d.createTexture({ size: [w, h], format: SCENE_FMT, usage });
    this.bloomA = d.createTexture({ size: [bw, bh], format: SCENE_FMT, usage });
    this.bloomB = d.createTexture({ size: [bw, bh], format: SCENE_FMT, usage });
    this.sceneView = this.sceneTex.createView();
    const vs = this.sceneView;
    const va = this.bloomA.createView();
    const vb = this.bloomB.createView();
    const q = d.queue;
    q.writeBuffer(this.postUbo[0], 0, new Float32Array([1 / w, 1 / h, 0.85, 0])); // bright: texel исходника, порог
    q.writeBuffer(this.postUbo[1], 0, new Float32Array([1 / bw, 0, 0, 0])); // blur H
    q.writeBuffer(this.postUbo[2], 0, new Float32Array([0, 1 / bh, 0, 0])); // blur V
    const bg = (tex: GPUTextureView, ubo: GPUBuffer, tex2: GPUTextureView) =>
      d.createBindGroup({
        layout: this.postBGL,
        entries: [
          { binding: 0, resource: this.postSampler },
          { binding: 1, resource: tex },
          { binding: 2, resource: { buffer: ubo } },
          { binding: 3, resource: tex2 },
        ],
      });
    this.postBG = [bg(vs, this.postUbo[0], vs), bg(va, this.postUbo[1], va), bg(vb, this.postUbo[2], vb), bg(vs, this.postUbo[3], va)];
  }

  private encodePost(enc: GPUCommandEncoder, target: GPUTextureView) {
    const run = (pipe: GPURenderPipeline, bg: GPUBindGroup, view: GPUTextureView) => {
      const p = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      p.setPipeline(pipe);
      p.setBindGroup(0, bg);
      p.draw(3);
      p.end();
    };
    const va = this.bloomA!.createView();
    const vb = this.bloomB!.createView();
    run(this.pipeBright, this.postBG[0], va); // сцена → яркие области (1/4)
    for (let k = 0; k < 2; k++) {
      run(this.pipeBlur, this.postBG[1], vb); // A → B (по X)
      run(this.pipeBlur, this.postBG[2], va); // B → A (по Y)
    }
    run(this.pipeComp, this.postBG[3], target); // сцена + bloom → канвас
  }

  /** Перегенерировать рельеф и сбросить симуляцию (вода, осадок, эрозия, счётчик). */
  generate(p: TerrainParams) {
    this.params = p;
    const d = new Float32Array([
      p.seed, p.freq, p.octaves, p.lacunarity,
      p.gain, p.ridge, p.worley, p.warp,
      p.island, p.exponent, GRID, 0,
    ]);
    this.device.queue.writeBuffer(this.noiseUBO, 0, d);

    const enc = this.device.createCommandEncoder();
    for (const b of [this.water, this.sedA, this.sedB, this.flux, this.qBuf, this.acc]) {
      enc.clearBuffer(b);
    }
    const pass = enc.beginComputePass();
    pass.setPipeline(this.noisePipeline);
    pass.setBindGroup(0, this.noiseBG);
    pass.dispatchWorkgroups(WG8, WG8);
    pass.end();
    enc.copyBufferToBuffer(this.tA, 0, this.h0, 0, CELLS * 4);
    this.snapTotal = 0;
    this.lastSnapStep = 0;
    this.scrub = null;
    this.encodeLight(enc, true);
    this.lastSunKey = this.sunKey();
    if (this.timelapse.on) this.takeSnapshot(enc, 0);
    this.device.queue.submit([enc.finish()]);
    this.emitTimelapse();
    this.stepsDone = 0;
    this.pendingSteps = 0;
    this.onSteps?.(0);
  }

  setRenderParams(p: TerrainParams) {
    this.params = p;
  }

  /** Выполнить один шаг симуляции (кнопка Step). */
  stepOnce() {
    this.pendingSteps += 1;
  }

  /** Убрать всю воду, осадок и потоки (рельеф остаётся). */
  clearWater() {
    const enc = this.device.createCommandEncoder();
    for (const b of [this.water, this.sedA, this.sedB, this.flux, this.qBuf]) enc.clearBuffer(b);
    this.device.queue.submit([enc.finish()]);
  }

  clearSources() {
    this.sources = [];
  }

  /** Прогнать N шагов вне цикла кадров (чанками, чтобы не словить таймаут GPU). */
  async runSteps(total: number, chunk = 50) {
    this.writeSimUniform();
    for (let done = 0; done < total && !this.disposed; done += chunk) {
      const n = Math.min(chunk, total - done);
      const enc = this.device.createCommandEncoder();
      this.encodeSteps(enc, n);
      this.stepsDone += n;
      this.maybeSnapshot(enc);
      this.device.queue.submit([enc.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    }
    this.onSteps?.(this.stepsDone);
  }

  /** DEV: время каждого прохода в мкс на диспатч (по 400 диспатчей подряд). */
  async profile(reps = 400): Promise<Record<string, number>> {
    this.writeSimUniform();
    const out: Record<string, number> = {};
    const all = SIM_ENTRIES.map((e, i) => [e, this.simPipelines[i]] as const);
    for (const [name, pl] of all) {
      const run = async (n: number) => {
        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setBindGroup(0, this.simBG);
        pass.setPipeline(pl);
        for (let k = 0; k < n; k++) pass.dispatchWorkgroups(WG, WG);
        pass.end();
        this.device.queue.submit([enc.finish()]);
        await this.device.queue.onSubmittedWorkDone();
      };
      await run(20);
      const t = performance.now();
      await run(reps);
      out[name] = Math.round(((performance.now() - t) * 1000) / reps);
    }
    return out;
  }

  // ───────────── time-lapse ─────────────

  private takeSnapshot(enc: GPUCommandEncoder, step: number) {
    const slot = this.snapTotal % SNAP_MAX;
    enc.copyBufferToBuffer(this.tA, 0, this.snapBuf, slot * CELLS * 4, CELLS * 4);
    this.snapStepAt[slot] = step;
    this.snapTotal++;
    this.lastSnapStep = step;
    this.emitTimelapse();
  }

  private maybeSnapshot(enc: GPUCommandEncoder) {
    if (this.timelapse.on && this.stepsDone - this.lastSnapStep >= this.timelapse.interval) {
      this.takeSnapshot(enc, this.stepsDone);
    }
  }

  /** Включить запись time-lapse: снимок рельефа каждые `interval` шагов (первый — сразу). */
  startTimelapse(interval: number) {
    this.timelapse = { on: true, interval: Math.max(10, interval) };
    this.snapTotal = 0;
    this.scrub = null;
    const enc = this.device.createCommandEncoder();
    this.takeSnapshot(enc, this.stepsDone);
    this.device.queue.submit([enc.finish()]);
  }

  stopTimelapse() {
    this.timelapse.on = false;
  }

  clearTimelapse() {
    this.snapTotal = 0;
    this.scrub = null;
    this.emitTimelapse();
  }

  private sunKey() {
    return `${this.fx.tod.toFixed(4)}|${this.params.heightScale}|${this.params.seaLevel}`;
  }

  private emitTimelapse() {
    const count = Math.min(this.snapTotal, SNAP_MAX);
    const first = this.snapTotal - count;
    const steps: number[] = [];
    for (let k = 0; k < count; k++) steps.push(this.snapStepAt[(first + k) % SNAP_MAX]);
    this.onTimelapse?.({ count, steps });
  }

  private snapBindGroup(k: number): GPUBindGroup {
    const count = Math.min(this.snapTotal, SNAP_MAX);
    const first = this.snapTotal - count;
    const slot = (first + Math.min(k, count - 1)) % SNAP_MAX;
    let bg = this.snapBGs.get(slot);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.renderBGL,
        entries: [
          { binding: 0, resource: { buffer: this.renderUBO } },
          { binding: 1, resource: { buffer: this.snapBuf, offset: slot * CELLS * 4, size: CELLS * 4 } },
          { binding: 2, resource: { buffer: this.water } },
          { binding: 3, resource: { buffer: this.h0 } },
          { binding: 4, resource: { buffer: this.acc } },
          { binding: 5, resource: { buffer: this.qBuf } },
          { binding: 6, resource: { buffer: this.lightBuf } },
        ],
      });
      this.snapBGs.set(slot, bg);
    }
    return bg;
  }

  // ───────────── readback для экспорта ─────────────

  private async readBuf(src: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
    const stage = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, stage, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const out = stage.getMappedRange().slice(0);
    stage.unmap();
    stage.destroy();
    return out;
  }

  /** Состояние для экспорта: рельеф, вода, расход q (развёрнут в f32), накопленная эрозия. */
  async readForExport() {
    const terrain = new Float32Array(await this.readBuf(this.tA, CELLS * 4));
    const water = new Float32Array(await this.readBuf(this.water, CELLS * 4));
    const accRaw = new Float32Array(await this.readBuf(this.acc, CELLS * 4));
    const qRaw = await this.readBuf(this.qBuf, CELLS * (this.f16 ? 4 : 8));
    const q = this.f16 ? decodeF16Pairs(new Uint16Array(qRaw)) : new Float32Array(qRaw);
    return { n: GRID, world: WORLD, terrain, water, acc: accRaw, q, cell: CELL };
  }

  /** DEV: перцентили |накопленной эрозии| и скорости воды — для подбора масштабов оверлеев. */
  async accStats() {
    const stage = this.device.createBuffer({ size: CELLS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.acc, 0, stage, 0, CELLS * 4);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    const abs = Array.from(f, Math.abs).sort((x, y) => x - y);
    const at = (q: number) => abs[Math.floor(q * (abs.length - 1))];
    return { p50: at(0.5), p90: at(0.9), p99: at(0.99), max: abs[abs.length - 1] };
  }

  /** DEV: прочитать буфер воды целиком (глубины) — для проверки инструментов. */
  async readWater(which: 'water' | 'terrain' = 'water'): Promise<Float32Array> {
    const stage = this.device.createBuffer({ size: CELLS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(which === 'water' ? this.water : this.tA, 0, stage, 0, CELLS * 4);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    return f;
  }

  private strataUniform(): number[] {
    const g = this.geo;
    return [g.scale, g.warp, g.dipX, g.dipY];
  }

  private writeSimUniform() {
    const p = this.params;
    const s = this.sim;
    const g = this.geo;
    const hs = p.heightScale;
    const talusH = (Math.tan((s.talus * Math.PI) / 180) * CELL) / hs;
    const rainProb = 0.25;
    const src: number[] = [];
    for (let k = 0; k < MAX_SOURCES; k++) {
      const o = this.sources[k];
      src.push(...(o ? [o.x, o.y, o.rate, 3] : [0, 0, 0, 1]));
    }
    this.device.queue.writeBuffer(
      this.simUBO,
      0,
      new Float32Array([
        GRID, s.rain, s.evap, s.kc, s.ks, s.kd, talusH, s.thermal,
        hs / CELL, s.flowG, p.seaLevel, rainProb, s.rainOn ? 1 : 0, s.stormPeriod, s.stormDuty, 0,
        ...this.strataUniform(),
        ...(g.strataErosion ? [g.hClay, g.hSand, g.hLime, g.hBasalt] : [1, 1, 1, 1]),
        ...src,
      ]),
    );
  }

  private encodeSteps(enc: GPUCommandEncoder, count: number) {
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, this.simBG);
    for (let k = 0; k < count; k++) {
      for (const pl of this.simPipelines) {
        pass.setPipeline(pl);
        pass.dispatchWorkgroups(WG, WG);
      }
    }
    pass.end();
  }

  // ───────────── камера и ввод ─────────────

  /** Текущая поза камеры: кинопролёт (если играет) или орбитальная. */
  private camPose(): CamKey {
    if (this.cine.playing) return this.cinePose(this.cine.progress);
    return { eye: this.eyePos(), target: [...this.target] as Vec3 };
  }

  /** Поза на пути в момент t∈[0,1]: Catmull-Rom по ключам либо автопролёт вокруг острова. */
  private cinePose(t: number): CamKey {
    const k = this.cineKeys;
    if (k.length === 0) {
      const a = this.yaw + t * Math.PI * 2;
      const pitch = 0.42 + 0.22 * Math.sin(t * Math.PI * 2);
      const r = 105 - 18 * Math.sin(t * Math.PI * 2 + 1);
      const cp = Math.cos(pitch);
      return {
        eye: [r * cp * Math.sin(a), 4 + r * Math.sin(pitch), r * cp * Math.cos(a)],
        target: [0, 4, 0],
      };
    }
    if (k.length === 1) return k[0];
    const e = t * t * (3 - 2 * t); // ease in-out
    const u = e * (k.length - 1);
    const i = Math.min(Math.floor(u), k.length - 2);
    const f = u - i;
    const P = (j: number) => k[Math.max(0, Math.min(k.length - 1, j))];
    const cr = (a: number[], b: number[], c: number[], d: number[]) =>
      a.map(
        (_, n) =>
          0.5 *
          (2 * b[n] +
            (-a[n] + c[n]) * f +
            (2 * a[n] - 5 * b[n] + 4 * c[n] - d[n]) * f * f +
            (-a[n] + 3 * b[n] - 3 * c[n] + d[n]) * f * f * f),
      ) as Vec3;
    return {
      eye: cr(P(i - 1).eye, P(i).eye, P(i + 1).eye, P(i + 2).eye),
      target: cr(P(i - 1).target, P(i).target, P(i + 1).target, P(i + 2).target),
    };
  }

  /** Добавить ключевой кадр из текущего вида орбитальной камеры. */
  addKeyframe() {
    this.cineKeys.push({ eye: this.eyePos(), target: [...this.target] as Vec3 });
    this.emitCine();
  }

  clearKeyframes() {
    this.cineKeys = [];
    this.emitCine();
  }

  playCine(duration: number) {
    this.cine = { playing: true, t0: performance.now(), duration: Math.max(1, duration), progress: 0 };
    this.emitCine();
  }

  stopCine() {
    this.cine.playing = false;
    if (this.recorder && this.recorder.state !== 'inactive') this.recorder.stop();
    this.emitCine();
  }

  /** Проиграть путь и записать canvas в WebM; по завершении вызывается onDone(blob). */
  startRecording(duration: number, onDone: (blob: Blob) => void) {
    if (this.recorder) return;
    const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) =>
      MediaRecorder.isTypeSupported(m),
    );
    // captureStream(0) + requestFrame(): по одному кадру видео на каждый отрисованный кадр
    const stream = this.canvas.captureStream(0);
    this.recTrack = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
    this.recChunks = [];
    this.recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 14_000_000 });
    this.recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.recChunks.push(e.data);
    };
    this.recorder.onstop = () => {
      const blob = new Blob(this.recChunks, { type: 'video/webm' });
      this.recorder = null;
      this.recTrack = null;
      this.cine.playing = false;
      this.emitCine();
      this.onRecorded?.(blob);
    };
    this.onRecorded = onDone;
    this.recorder.start(250);
    this.playCine(duration);
  }

  private emitCine() {
    this.onCine?.({
      keys: this.cineKeys.length,
      playing: this.cine.playing,
      recording: !!this.recorder,
      progress: this.cine.progress,
    });
  }

  private eyePos(): Vec3 {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return [
      this.target[0] + this.dist * cp * Math.sin(this.yaw),
      this.target[1] + this.dist * sp,
      this.target[2] + this.dist * cp * Math.cos(this.yaw),
    ];
  }

  /** Луч из камеры через точку экрана (clientX/Y). */
  private rayFromClient(cx: number, cy: number): { ro: Vec3; rd: Vec3 } {
    const r = this.canvas.getBoundingClientRect();
    const nx = ((cx - r.left) / r.width) * 2 - 1;
    const ny = 1 - ((cy - r.top) / r.height) * 2;
    const pose = this.camPose();
    const eye = pose.eye;
    const f = norm3(sub3(pose.target, eye));
    const right = norm3(cross3(f, [0, 1, 0]));
    const up = cross3(right, f);
    const th = Math.tan(FOV / 2);
    const asp = r.width / r.height;
    const rd = norm3([
      f[0] + right[0] * nx * th * asp + up[0] * ny * th,
      f[1] + right[1] * nx * th * asp + up[1] * ny * th,
      f[2] + right[2] * nx * th * asp + up[2] * ny * th,
    ]);
    return { ro: eye, rd };
  }

  private attachControls() {
    const c = this.canvas;
    type Drag = { x: number; y: number; mode: 'orbit' | 'pan' | 'tool' };
    let drag: Drag | null = null;

    c.addEventListener('pointerdown', (e) => {
      const pan = e.button === 1 || (e.shiftKey && e.button === 0) || (this.tool === 'orbit' && e.button === 2);
      let mode: Drag['mode'] = 'orbit';
      if (pan) mode = 'pan';
      else if (this.tool !== 'orbit' && e.button === 0) mode = 'tool';
      drag = { x: e.clientX, y: e.clientY, mode };
      this.dragging = true;
      this.yawV = 0;
      this.pitchV = 0;
      this.focusGoal = null;
      c.setPointerCapture(e.pointerId);
      this.pickReq = this.rayFromClient(e.clientX, e.clientY);
      if (mode === 'tool') this.toolPress();
    });
    c.addEventListener('pointerup', () => {
      if (drag?.mode === 'tool') this.toolRelease();
      drag = null;
      this.dragging = false;
    });
    c.addEventListener('pointerleave', () => {
      if (!drag) this.hit.valid = false;
    });
    c.addEventListener('pointermove', (e) => {
      this.pickReq = this.rayFromClient(e.clientX, e.clientY);
      if (!drag || drag.mode === 'tool') return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      drag.x = e.clientX;
      drag.y = e.clientY;
      if (drag.mode === 'pan') {
        const s = this.dist * 0.0015;
        const rx = Math.cos(this.yaw), rz = -Math.sin(this.yaw);
        const fx = -Math.sin(this.yaw), fz = -Math.cos(this.yaw);
        this.target[0] += -dx * rx * s + dy * fx * s;
        this.target[2] += -dx * rz * s + dy * fz * s;
      } else {
        this.yaw -= dx * 0.005;
        this.pitch = Math.min(1.5, Math.max(0.05, this.pitch + dy * 0.005));
        this.yawV = -dx * 0.005;
        this.pitchV = dy * 0.005;
      }
    });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.zoomV = Math.max(-0.12, Math.min(0.12, this.zoomV + e.deltaY * 0.0003));
    }, { passive: false });
    c.addEventListener('dblclick', () => {
      if (!this.hit.valid) return;
      this.focusGoal = [cellToWorld(this.hit.x), this.hit.h * this.params.heightScale, cellToWorld(this.hit.y)];
    });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    new ResizeObserver(() => this.resize()).observe(c);
  }

  private toolPress() {
    if (!this.hit.valid) return;
    if (this.tool === 'river') {
      const s: Source = { x: this.hit.x, y: this.hit.y, rate: this.toolParams.riverRate };
      if (this.sources.length >= MAX_SOURCES) this.sources.shift();
      this.sources.push(s);
    } else if (this.tool === 'dam') {
      this.damStart = { ...this.hit };
    } else {
      this.toolDown = true;
    }
  }

  private toolRelease() {
    this.toolDown = false;
    if (this.tool === 'dam' && this.damStart) {
      if (this.hit.valid) this.pendingDam = { a: this.damStart, b: { ...this.hit } };
      this.damStart = null;
    }
  }

  private resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width === w && this.canvas.height === h && this.depthTex) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.depthTex?.destroy();
    this.depthTex = this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.createPostTargets(w, h);
  }

  private loop = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    this.frame();
  };

  private frames = 0;
  private lastT = performance.now();
  private lastStatT = 0;

  /** Записать uniform кисти и выполнить её на GPU. */
  private encodeTool(enc: GPUCommandEncoder, op: number, c: Hit, strength: number, a?: Hit, b?: Hit, level = 0) {
    const rc = this.toolParams.radius / CELL;
    this.device.queue.writeBuffer(
      this.toolUBO,
      0,
      new Float32Array([
        c.x, c.y, rc, strength,
        op, 0, 0, 0,
        a?.x ?? 0, a?.y ?? 0, b?.x ?? 0, b?.y ?? 0,
        level, Math.max(2, rc * 0.35), GRID, 0,
      ]),
    );
    const pass = enc.beginComputePass();
    pass.setPipeline(this.toolPipeline);
    pass.setBindGroup(0, this.toolBG);
    pass.dispatchWorkgroups(WG, WG);
    pass.end();
  }

  private stepCamera() {
    if (this.cine.playing) return;
    if (!this.dragging && (Math.abs(this.yawV) > 1e-5 || Math.abs(this.pitchV) > 1e-5)) {
      this.yaw += this.yawV;
      this.pitch = Math.min(1.5, Math.max(0.05, this.pitch + this.pitchV));
      this.yawV *= 0.93;
      this.pitchV *= 0.93;
    }
    if (Math.abs(this.zoomV) > 1e-4) {
      this.dist = Math.min(400, Math.max(20, this.dist * Math.exp(this.zoomV)));
      this.zoomV *= 0.82;
    }
    if (this.focusGoal) {
      let d = 0;
      for (let k = 0; k < 3; k++) {
        const diff = this.focusGoal[k] - this.target[k];
        this.target[k] += diff * 0.14;
        d += Math.abs(diff);
      }
      if (d < 0.02) this.focusGoal = null;
    }
  }

  /** Клетка под курсором: высота (0..1) и мировые координаты — для HUD. */
  hover(): { valid: boolean; h: number; x: number; y: number } {
    return { valid: this.hit.valid, h: this.hit.h, x: this.hit.x, y: this.hit.y };
  }

  private frame() {
    this.resize();
    this.stepCamera();
    const p = this.params;
    const g = this.geo;
    const enc = this.device.createCommandEncoder();

    // --- пикинг курсора (результат приходит с задержкой в 1–2 кадра) ---
    let doPick = false;
    if (this.pickReq && !this.pickBusy) {
      const { ro, rd } = this.pickReq;
      this.pickReq = null;
      this.device.queue.writeBuffer(
        this.pickUBO, 0,
        new Float32Array([ro[0], ro[1], ro[2], 0, rd[0], rd[1], rd[2], 0, GRID, p.heightScale, WORLD, 900]),
      );
      const pp = enc.beginComputePass();
      pp.setPipeline(this.pickPipeline);
      pp.setBindGroup(0, this.pickBG);
      pp.dispatchWorkgroups(1);
      pp.end();
      enc.copyBufferToBuffer(this.pickBuf, 0, this.pickStage, 0, 16);
      doPick = true;
      this.pickBusy = true;
    }

    // --- кисти ---
    const tp = this.toolParams;
    if (this.toolDown && this.hit.valid) {
      if (this.tool === 'rain') this.encodeTool(enc, 0, this.hit, 0.02 * tp.strength);
      else if (this.tool === 'raise') this.encodeTool(enc, 1, this.hit, 0.004 * tp.strength);
      else if (this.tool === 'lower') this.encodeTool(enc, 2, this.hit, 0.004 * tp.strength);
    }
    if (this.pendingDam) {
      const { a, b } = this.pendingDam;
      this.pendingDam = null;
      const level = Math.min(Math.min(a.h, b.h) + tp.damExtra / p.heightScale, 1);
      this.encodeTool(enc, 3, a, 0, a, b, level);
    }

    // --- симуляция ---
    let n = this.pendingSteps;
    this.pendingSteps = 0;
    const nowT = performance.now();
    const dt = nowT - this.lastFrameT;
    this.lastFrameT = nowT;
    if (this.running) {
      // адаптивный бюджет: слайдер = желаемые шаги/кадр, фактические — под ~45–60 FPS
      const target = Math.max(1, this.speed);
      if (dt > 26) this.effSteps = Math.max(1, this.effSteps * 0.85);
      else if (dt < 20) this.effSteps = Math.min(target, this.effSteps * 1.08 + 0.5);
      this.effSteps = Math.min(this.effSteps, target);
      n += Math.max(1, Math.round(this.effSteps));
    }
    if (n > 0) {
      this.writeSimUniform();
      this.encodeSteps(enc, n);
      this.stepsDone += n;
      this.maybeSnapshot(enc);
      this.lightDirty = true;
    }
    if (this.toolDown || this.pendingDam) this.lightDirty = true;
    this.lightAge++;
    const sk = this.sunKey();
    if (sk !== this.lastSunKey) {
      this.lastSunKey = sk;
      this.lightDirty = true;
      this.lightAge = 99;
      this.encodeLightBefore = true;
    }
    if (this.lightDirty && this.lightAge >= 4) {
      this.encodeLight(enc, this.encodeLightBefore);
      this.encodeLightBefore = false;
    }

    // --- кинопролёт ---
    if (this.cine.playing) {
      this.cine.progress = Math.min(1, (performance.now() - this.cine.t0) / (this.cine.duration * 1000));
      if (this.cine.progress >= 1) {
        // последний кадр отрисуем, затем остановим (и запись)
        queueMicrotask(() => this.stopCine());
      }
    }

    // --- рендер ---
    const aspect = this.canvas.width / this.canvas.height;
    const pose = this.camPose();
    const eye = pose.eye;
    const vp = mul(perspective(FOV, aspect, 1, 1200), lookAt(eye, pose.target));
    const cutX = (g.cut - 0.5) * WORLD;
    const brushOn = this.tool !== 'orbit' && this.hit.valid;
    const damPreview = this.tool === 'dam' && this.damStart && this.hit.valid;
    const ovMode = { none: 0, erosion: 1, velocity: 2, diff: 3 }[this.overlay];
    const u = new Float32Array(UBO_FLOATS);
    u.set(vp, 0);
    const rainTarget = this.running && this.sim?.rainOn && this.scrub === null ? 1 : 0;
    this.rainVis += (rainTarget - this.rainVis) * 0.04;
    if (Math.abs(rainTarget - this.rainVis) < 0.005) this.rainVis = rainTarget;
    u.set([eye[0], eye[1], eye[2], this.rainVis], 16);
    u.set([GRID, p.heightScale, p.seaLevel, WORLD], 20);
    const sun = this.sunState();
    u.set([sun.dir[0], sun.dir[1], sun.dir[2], 0], 24);
    u.set(this.strataUniform(), 28);
    u.set([g.view === 'geology' ? 1 : 0, cutX, g.diorama ? 1 : 0, 10], 32);
    u.set(
      [brushOn ? cellToWorld(this.hit.x) : 0, brushOn ? cellToWorld(this.hit.y) : 0, this.tool === 'dam' ? tp.radius * 0.35 : tp.radius, brushOn ? 1 : 0],
      36,
    );
    if (damPreview && this.damStart) {
      u.set([cellToWorld(this.damStart.x), cellToWorld(this.damStart.y), cellToWorld(this.hit.x), cellToWorld(this.hit.y)], 40);
    }
    u.set([performance.now() / 1000, ovMode, damPreview ? 1 : 0, 0], 44);
    for (let k = 0; k < MAX_SOURCES; k++) {
      const o = this.sources[k];
      u.set(o ? [cellToWorld(o.x), cellToWorld(o.y), 1, 0] : [0, 0, 0, 0], 48 + k * 4);
    }
    // камера, атмосфера, палитры
    {
      const f = norm3(sub3(pose.target, eye));
      const right = norm3(cross3(f, [0, 1, 0]));
      const up = cross3(right, f);
      const c = CLIMATES[this.climate];
      u.set([f[0], f[1], f[2], aspect], 80);
      u.set([right[0], right[1], right[2], 0], 84);
      u.set([up[0], up[1], up[2], Math.tan(FOV / 2)], 88);
      const treeK = { temperate: 1, arid: 0.3, monsoon: 1.4 }[this.climate] ?? 1;
      u.set([this.fx.fog, this.dbg[0], this.dbg[1], this.fx.trees * treeK], 92);
      const warmH: Vec3 = [0.76, 0.4, 0.22];
      const warmZ: Vec3 = [0.17, 0.11, 0.3];
      const mixv = (a: number[], b: number[], t: number, k: number) => a.map((v, n) => (v + (b[n] - v) * t) * k);
      u.set([...mixv(c.skyHorizon, warmH, sun.warm * 0.75, sun.bright), 0], 96);
      u.set([...mixv(c.skyZenith, warmZ, sun.warm * 0.6, sun.bright), 0], 100);
      u.set([...sun.col, sun.intensity], 112);
      u.set([...c.grass, c.snow ? 1 : 0], 104);
      u.set([...c.sand, 0], 108);
    }
    const scrubbing = this.scrub !== null && this.snapTotal > 0;
    if (scrubbing) u.set([performance.now() / 1000, 0, 0, 2], 44);
    this.device.queue.writeBuffer(this.renderUBO, 0, u);
    const splitOn = this.split !== null;
    if (splitOn) {
      // «до»: без курсора, источников, превью и оверлея
      const ub = u.slice();
      ub.set([0, 0, 0, 0], 36);
      ub.set([0, 0, 0, 0], 40);
      ub.set([u[44], 0, 0, 1], 44);
      for (let k = 0; k < MAX_SOURCES; k++) ub.set([0, 0, 0, 0], 48 + k * 4);
      this.device.queue.writeBuffer(this.renderUBO2, 0, ub);
    }

    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.sceneView,
        clearValue: { r: 0.05, g: 0.07, b: 0.11, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTex!.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.skyPipeline);
    pass.setBindGroup(0, this.renderBG);
    pass.draw(3);
    const cutActive = g.cut < 0.999;
    const drawScene = (bg: GPUBindGroup) => {
      pass.setPipeline(this.renderPipeline);
      pass.setBindGroup(0, bg);
      pass.draw((GRID - 1) * (GRID - 1) * 6);
      pass.setPipeline(this.treePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(30, TREE_CANDIDATES);
      // стенки: diorama (все 4) или только плоскость среза (инстанс 3)
      if (g.diorama || cutActive) {
        pass.setPipeline(this.wallPipeline);
        pass.setBindGroup(0, bg);
        pass.draw(GRID * 2, g.diorama ? 4 : 1, 0, g.diorama ? 0 : 3);
      }
    };
    const afterBG = scrubbing ? this.snapBindGroup(this.scrub!) : this.renderBG;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    if (splitOn) {
      const sx = Math.min(cw - 1, Math.max(1, Math.round(this.split! * cw)));
      pass.setScissorRect(0, 0, sx, ch);
      drawScene(this.renderBG2); // слева — «до»
      pass.setScissorRect(sx, 0, cw - sx, ch);
      drawScene(afterBG); // справа — «после»
    } else {
      drawScene(afterBG);
    }
    if (this.rainVis > 0.01) {
      pass.setScissorRect(0, 0, this.canvas.width, this.canvas.height);
      pass.setPipeline(this.rainPipeline);
      pass.setBindGroup(0, this.renderBG);
      pass.draw(2, 4000);
    }
    pass.end();
    this.device.queue.writeBuffer(this.postUbo[3], 0, new Float32Array([this.fx.bloom, 0.35, 0, 0]));
    this.encodePost(enc, this.ctx.getCurrentTexture().createView());
    this.device.queue.submit([enc.finish()]);
    this.recTrack?.requestFrame();

    if (doPick) {
      this.pickStage.mapAsync(GPUMapMode.READ).then(() => {
        const a = new Float32Array(this.pickStage.getMappedRange().slice(0));
        this.pickStage.unmap();
        this.hit = { valid: a[2] > 0.5, x: a[0], y: a[1], h: a[3] };
        this.pickBusy = false;
      }).catch(() => {
        this.pickBusy = false;
      });
    }

    this.frames++;
    const now = performance.now();
    if (now - this.lastT >= 500) {
      const sps = Math.round(((this.stepsDone - this.lastStepsStat) * 1000) / (now - this.lastT));
      this.lastStepsStat = this.stepsDone;
      this.onFps?.(Math.round((this.frames * 1000) / (now - this.lastT)), Math.max(0, sps));
      this.frames = 0;
      this.lastT = now;
    }
    if (now - this.lastStatT >= 200) {
      this.onSteps?.(this.stepsDone);
      if (this.cine.playing) this.emitCine();
      this.lastStatT = now;
    }
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    for (const b of [this.tA, this.tB, this.water, this.sedA, this.sedB, this.flux, this.qBuf, this.acc, this.h0]) {
      b?.destroy();
    }
    this.depthTex?.destroy();
    this.device?.destroy();
  }
}

const sub3 = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross3 = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm3 = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** f16 (биты) → f32; пары (x, y) подряд. */
function decodeF16Pairs(h: Uint16Array): Float32Array {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const v = h[i];
    const sign = v & 0x8000 ? -1 : 1;
    const e = (v >> 10) & 0x1f;
    const m = v & 0x3ff;
    out[i] = e === 0 ? sign * Math.pow(2, -14) * (m / 1024) : e === 31 ? (m ? NaN : sign * Infinity) : sign * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return out;
}
