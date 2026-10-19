// Story mode: превращает метрики симуляции в «геологическую хронику».
// Масштаб стилизованный: 1 юнит мира по горизонтали = 250 м (остров ≈ 25 км), 1 единица нормализованной высоты × heightScale юнитов × 40 м.
import type { ExportState } from './export/maps';
import { YEARS_PER_STEP } from './store';

const KM_PER_UNIT = 0.25;
const M_PER_HEIGHT_UNIT = 40;

export interface StoryEvent {
  id: string;
  step: number;
  epoch: number;
  title: string;
  text: string;
}

export interface StoryContext {
  heightScale: number;
  seaLevel: number;
  stormPeriod: number;
}

const DIRS = ['востоке', 'северо-востоке', 'севере', 'северо-западе', 'западе', 'юго-западе', 'юге', 'юго-востоке'];

/** Часть острова по клетке: север — сверху (меньше y). */
function sector(x: number, y: number, n: number): string {
  const ang = Math.atan2(n / 2 - y, x - n / 2); // север = +, восток = +
  const k = Math.round(((ang + Math.PI * 2) % (Math.PI * 2)) / (Math.PI / 4)) % 8;
  return DIRS[k];
}

export class StoryTracker {
  private fired = new Set<string>();
  private lastStormIdx = 0;

  reset() {
    this.fired.clear();
    this.lastStormIdx = 0;
  }

  analyze(st: ExportState, step: number, ctx: StoryContext): StoryEvent[] {
    const out: StoryEvent[] = [];
    const { n, terrain, acc, water, q } = st;
    const epoch = Math.floor(step / 2000) + 1;
    const years = step * YEARS_PER_STEP;
    const fire = (id: string, title: string, text: string) => {
      if (this.fired.has(id)) return;
      this.fired.add(id);
      out.push({ id, step, epoch, title, text });
    };

    const cellKm = st.cell * KM_PER_UNIT;
    const cellKm2 = cellKm * cellKm;
    const vertM = ctx.heightScale * M_PER_HEIGHT_UNIT;
    const sea = ctx.seaLevel;

    let eroded = 0, coastDeposit = 0;
    let deepest = 0, deepestIdx = -1;
    let channelCells = 0;
    let lakeCells = 0, lx = 0, ly = 0;
    for (let i = 0; i < n * n; i++) {
      const land = terrain[i] > sea;
      const a = acc[i];
      if (a < 0) {
        eroded -= a;
        if (land && a < deepest) { deepest = a; deepestIdx = i; }
      } else if (a > 0 && terrain[i] < sea + 0.04) {
        coastDeposit += a;
      }
      if (land) {
        if (Math.hypot(q[i * 2], q[i * 2 + 1]) > 0.008) channelCells++;
        if (water[i] > 0.03 && terrain[i] > sea + 0.01) {
          lakeCells++;
          lx += i % n;
          ly += Math.floor(i / n);
        }
      }
    }
    // Σ(Δh)·(м) · км² = 1e-3 км³ на каждый м·км²
    const erodedKm3 = (eroded * vertM * cellKm2) / 1000;

    /** Из упорядоченного списка порогов сообщает только высший достигнутый, прочие помечает пройденными. */
    const milestone = (value: number, thresholds: number[], idOf: (t: number) => string, make: (t: number) => [string, string]) => {
      let top = -1;
      thresholds.forEach((t, k) => {
        if (value >= t) top = k;
      });
      if (top < 0) return;
      thresholds.slice(0, top).forEach((t) => this.fired.add(idOf(t)));
      const [title, text] = make(thresholds[top]);
      fire(idOf(thresholds[top]), title, text);
    };

    if (step > 0) {
      fire('first-water', 'Первая вода', 'Дожди собираются в ручьи на склонах. Рельеф ещё почти не изменился — но русла уже намечены.');
    }
    milestone(
      channelCells * cellKm,
      [100, 350],
      (t) => `rivers-${t}`,
      (t) =>
        t < 350
          ? ['Ручьи сливаются в реки', `Сеть постоянных водотоков вытянулась примерно на ${Math.round(channelCells * cellKm)} км.`]
          : ['Речная сеть', `Реки образовали разветвлённую систему (≈ ${Math.round(channelCells * cellKm)} км русел). Вода находит самый короткий путь к морю.`],
    );

    if (deepestIdx >= 0) {
      const depthM = -deepest * vertM;
      const where = sector(deepestIdx % n, Math.floor(deepestIdx / n), n);
      milestone(
        depthM,
        [100, 200, 300, 400],
        (m) => `incision-${m}`,
        (m) => [`Врез ${m} м`, `Река прорезала долину глубиной ≈ ${Math.round(depthM)} м на ${where} острова. Мягкие породы вымываются быстрее, твёрдые образуют уступы и пороги.`],
      );
    }

    milestone(
      erodedKm3,
      [1.5, 3, 5, 8],
      (t) => `volume-${t}`,
      (t) => [`Смыто ${t} км³ породы`, `Эрозия унесла с острова около ${erodedKm3.toFixed(1)} км³ породы за ${years.toLocaleString('ru-RU')} лет.`],
    );

    if (lakeCells * cellKm2 >= 1.2) {
      const area = lakeCells * cellKm2;
      const where = sector(lx / lakeCells, ly / lakeCells, n);
      const tier = area >= 6 ? 2 : 1;
      if (tier === 2) this.fired.add('lake-1');
      fire(`lake-${tier}`, tier === 2 ? 'Большое озеро' : 'Озеро', `В котловине на ${where} скопилась вода — озеро ≈ ${area.toFixed(1)} км². Оно служит ловушкой для наносов.`);
    }

    const coastKm3 = (coastDeposit * vertM * cellKm2) / 1000;
    if (coastKm3 >= 0.8 && coastKm3 > erodedKm3 * 0.1) {
      fire('delta', 'Дельта', `Наносы доносятся до моря: у побережья отложилось ≈ ${coastKm3.toFixed(2)} км³ осадков, растёт дельта.`);
    }

    if (ctx.stormPeriod > 0) {
      const idx = Math.floor(step / ctx.stormPeriod);
      if (idx > this.lastStormIdx) {
        for (let k = this.lastStormIdx + 1; k <= idx; k++) {
          fire(`storm-${k}`, `Ливень №${k}`, 'Шквальный дождь: вода сошла со склонов селевым потоком, русла углубились за считанные годы.');
        }
        this.lastStormIdx = idx;
      }
    }

    return out;
  }
}

export function storyToMarkdown(events: StoryEvent[]): string {
  const lines = ['# Геологическая хроника острова', '', '_Масштаб стилизованный: остров ≈ 25 км, пик ≈ 1100 м._', ''];
  for (const e of events) {
    lines.push(`## Эпоха ${e.epoch} · ${(e.step * YEARS_PER_STEP).toLocaleString('ru-RU')} лет — ${e.title}`, '', e.text, '');
  }
  return lines.join('\n');
}
