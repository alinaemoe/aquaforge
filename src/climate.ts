// Климатические пресеты: палитра биомов/неба + параметры гидрологии и пород.
import type { GeoParams, SimParams } from './store';

export type ClimateId = 'temperate' | 'arid' | 'monsoon';

type RGB = [number, number, number];

export interface Climate {
  id: ClimateId;
  label: string;
  hint: string;
  grass: RGB;
  sand: RGB;
  snow: boolean;
  skyHorizon: RGB;
  skyZenith: RGB;
  fog: number;
  sim: Partial<SimParams>;
  geo: Partial<GeoParams>;
}

export const CLIMATES: Record<ClimateId, Climate> = {
  temperate: {
    id: 'temperate',
    label: 'Temperate',
    hint: 'Умеренный: ровный дождь, зелёные склоны, снег на вершинах',
    grass: [0.24, 0.4, 0.18],
    sand: [0.76, 0.7, 0.5],
    snow: true,
    skyHorizon: [0.3, 0.42, 0.58],
    skyZenith: [0.04, 0.09, 0.22],
    fog: 0.0035,
    sim: { rain: 0.0001, evap: 0.015, kc: 0.4, ks: 0.015, kd: 0.1, talus: 42, thermal: 0.015, stormPeriod: 0, stormDuty: 0.1 },
    geo: { scale: 22, hClay: 3.0, hSand: 1.0, hLime: 0.45, hBasalt: 0.12 },
  },
  arid: {
    id: 'arid',
    label: 'Arid / Canyon',
    hint: 'Пустыня: редкие шквальные ливни, быстрая эрозия, крутые стенки каньонов',
    grass: [0.52, 0.42, 0.24],
    sand: [0.86, 0.62, 0.36],
    snow: false,
    skyHorizon: [0.72, 0.5, 0.34],
    skyZenith: [0.1, 0.13, 0.27],
    fog: 0.0042,
    sim: { rain: 0.00006, evap: 0.03, kc: 0.9, ks: 0.05, kd: 0.08, talus: 58, thermal: 0.01, stormPeriod: 1500, stormDuty: 0.08 },
    geo: { scale: 30, hClay: 3.5, hSand: 1.2, hLime: 0.3, hBasalt: 0.08, dipX: 0.05 },
  },
  monsoon: {
    id: 'monsoon',
    label: 'Tropical Monsoon',
    hint: 'Джунгли: постоянный обильный дождь, сглаженные холмы, плотный влажный воздух',
    grass: [0.08, 0.4, 0.12],
    sand: [0.5, 0.46, 0.32],
    snow: false,
    skyHorizon: [0.38, 0.5, 0.48],
    skyZenith: [0.06, 0.15, 0.2],
    fog: 0.0065,
    sim: { rain: 0.00018, evap: 0.012, kc: 0.35, ks: 0.01, kd: 0.15, talus: 36, thermal: 0.03, stormPeriod: 0, stormDuty: 0.1 },
    geo: { scale: 16, hClay: 4.0, hSand: 1.5, hLime: 0.8, hBasalt: 0.3 },
  },
};
