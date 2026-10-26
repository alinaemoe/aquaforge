import { create } from 'zustand';
import type { StoryEvent } from './story';

export interface TerrainParams {
  seed: number;
  freq: number;
  octaves: number;
  lacunarity: number;
  gain: number;
  ridge: number;
  worley: number;
  warp: number;
  island: number;
  exponent: number;
  heightScale: number;
  seaLevel: number;
}

export const DEFAULT_PARAMS: TerrainParams = {
  seed: 7,
  freq: 3,
  octaves: 7,
  lacunarity: 2,
  gain: 0.5,
  ridge: 0.35,
  worley: 0.15,
  warp: 0.6,
  island: 0.6,
  exponent: 1.6,
  heightScale: 28,
  seaLevel: 0.18,
};

export interface SimParams {
  rain: number;
  evap: number;
  kc: number;
  ks: number;
  kd: number;
  talus: number; // градусы
  thermal: number;
  flowG: number;
  rainOn: boolean;
  stormPeriod: number; // шаги между ливнями (0 — нет)
  stormDuty: number; // доля периода с дождём
}

export const DEFAULT_SIM: SimParams = {
  rain: 0.0001,
  evap: 0.015,
  kc: 0.4,
  ks: 0.015,
  kd: 0.1,
  talus: 42,
  thermal: 0.015,
  flowG: 0.2,
  rainOn: true,
  stormPeriod: 0,
  stormDuty: 0.1,
};

export interface GeoParams {
  scale: number;      // пластов на единицу высоты
  warp: number;       // изгиб пластов
  dipX: number;       // наклон пластов
  dipY: number;
  hClay: number;      // податливость пород (множитель эрозии)
  hSand: number;
  hLime: number;
  hBasalt: number;
  strataErosion: boolean;
  view: 'biome' | 'geology';
  diorama: boolean;
  cut: number;        // 0..1 — доля мира, остающаяся слева от плоскости среза (1 = без среза)
}

export const DEFAULT_GEO: GeoParams = {
  scale: 22,
  warp: 0.08,
  dipX: 0.12,
  dipY: 0,
  hClay: 3.0,
  hSand: 1.0,
  hLime: 0.45,
  hBasalt: 0.12,
  strataErosion: true,
  view: 'biome',
  diorama: true,
  cut: 1,
};

export type Tool = 'orbit' | 'rain' | 'raise' | 'lower' | 'river' | 'dam';
export type Overlay = 'none' | 'erosion' | 'velocity' | 'diff';

export interface ToolParams {
  radius: number;     // радиус кисти, юниты мира
  strength: number;   // множитель силы
  riverRate: number;  // расход источника
  damExtra: number;   // превышение гребня плотины над низшей точкой концов, юниты мира
}

export const DEFAULT_TOOL: ToolParams = { radius: 6, strength: 1, riverRate: 0.015, damExtra: 0.5 };

export interface CineState {
  keys: number;
  playing: boolean;
  recording: boolean;
  progress: number;
}

export interface TimelapseState {
  count: number;
  steps: number[];
}

export interface FxParams {
  fog: number; // плотность тумана
  bloom: number; // сила bloom
  tod: number; // время суток 0..1 (рассвет → закат)
  trees: number; // плотность деревьев 0..1.5
}

export const YEARS_PER_STEP = 50;

interface State {
  params: TerrainParams;
  sim: SimParams;
  climate: 'temperate' | 'arid' | 'monsoon';
  fx: FxParams;
  setClimate: (c: 'temperate' | 'arid' | 'monsoon') => void;
  setFx: <K extends keyof FxParams>(k: K, v: FxParams[K]) => void;
  story: StoryEvent[];
  storyOn: boolean;
  addStory: (e: StoryEvent[]) => void;
  clearStory: () => void;
  setStoryOn: (b: boolean) => void;
  tl: TimelapseState;
  tlOn: boolean;
  tlInterval: number;
  scrub: number | null;
  meshRes: number;
  setTl: (t: TimelapseState) => void;
  setTlOn: (b: boolean) => void;
  setTlInterval: (n: number) => void;
  setScrub: (n: number | null) => void;
  setMeshRes: (n: number) => void;
  cine: CineState;
  cineDuration: number;
  simDuringRec: boolean;
  split: number | null;
  setCine: (c: CineState) => void;
  setCineDuration: (n: number) => void;
  setSimDuringRec: (b: boolean) => void;
  setSplit: (v: number | null) => void;
  tool: Tool;
  overlay: Overlay;
  toolParams: ToolParams;
  setTool: (t: Tool) => void;
  setOverlay: (o: Overlay) => void;
  setToolParam: <K extends keyof ToolParams>(k: K, v: ToolParams[K]) => void;
  geo: GeoParams;
  setGeo: <K extends keyof GeoParams>(k: K, v: GeoParams[K]) => void;
  running: boolean;
  speed: number;
  steps: number;
  setSim: <K extends keyof SimParams>(k: K, v: SimParams[K]) => void;
  setRunning: (b: boolean) => void;
  setSpeed: (n: number) => void;
  setSteps: (n: number) => void;
  gpuStatus: 'init' | 'ok' | 'unsupported' | 'error';
  gpuMessage: string;
  fps: number;
  sps: number;
  setParam: <K extends keyof TerrainParams>(k: K, v: TerrainParams[K]) => void;
  randomize: () => void;
  setGpu: (s: State['gpuStatus'], msg?: string) => void;
  setFps: (n: number, sps?: number) => void;
}

export const useStore = create<State>((set) => ({
  params: DEFAULT_PARAMS,
  sim: DEFAULT_SIM,
  geo: DEFAULT_GEO,
  climate: 'temperate',
  fx: { fog: 0.0035, bloom: 0.6, tod: 0.35, trees: 1 },
  setClimate: (climate) => set({ climate }),
  setFx: (k, v) => set((s) => ({ fx: { ...s.fx, [k]: v } })),
  story: [],
  storyOn: false,
  addStory: (e) => set((s) => ({ story: [...s.story, ...e] })),
  clearStory: () => set({ story: [] }),
  setStoryOn: (storyOn) => set({ storyOn }),
  tl: { count: 0, steps: [] },
  tlOn: false,
  tlInterval: 250,
  scrub: null,
  meshRes: 256,
  setTl: (tl) => set({ tl }),
  setTlOn: (tlOn) => set({ tlOn }),
  setTlInterval: (tlInterval) => set({ tlInterval }),
  setScrub: (scrub) => set({ scrub }),
  setMeshRes: (meshRes) => set({ meshRes }),
  cine: { keys: 0, playing: false, recording: false, progress: 0 },
  cineDuration: 20,
  simDuringRec: true,
  split: null,
  setCine: (cine) => set({ cine }),
  setCineDuration: (cineDuration) => set({ cineDuration }),
  setSimDuringRec: (simDuringRec) => set({ simDuringRec }),
  setSplit: (split) => set({ split }),
  tool: 'orbit',
  overlay: 'none',
  toolParams: DEFAULT_TOOL,
  setTool: (tool) => set({ tool }),
  setOverlay: (overlay) => set({ overlay }),
  setToolParam: (k, v) => set((s) => ({ toolParams: { ...s.toolParams, [k]: v } })),
  setGeo: (k, v) => set((s) => ({ geo: { ...s.geo, [k]: v } })),
  running: false,
  speed: 5,
  steps: 0,
  setSim: (k, v) => set((s) => ({ sim: { ...s.sim, [k]: v } })),
  setRunning: (running) => set({ running }),
  setSpeed: (speed) => set({ speed }),
  setSteps: (steps) => set({ steps }),
  gpuStatus: 'init',
  gpuMessage: '',
  fps: 0,
  sps: 0,
  setParam: (k, v) => set((s) => ({ params: { ...s.params, [k]: v } })),
  randomize: () =>
    set((s) => ({ params: { ...s.params, seed: Math.floor(Math.random() * 100000) } })),
  setGpu: (gpuStatus, gpuMessage = '') => set({ gpuStatus, gpuMessage }),
  setFps: (fps, sps = 0) => set({ fps, sps }),
}));
