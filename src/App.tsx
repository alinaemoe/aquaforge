import { useEffect, useRef, useState } from 'react';
import { CloudRain, Dices, Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { useStore, YEARS_PER_STEP, type GeoParams, type Overlay, type SimParams, type TerrainParams, type Tool, type ToolParams } from './store';
import { TerrainRenderer } from './gpu/TerrainRenderer';
import { CLIMATES, type ClimateId } from './climate';
import { StoryTracker, storyToMarkdown } from './story';
import { buildMesh, toGLB, toOBJ } from './export/mesh';
import { erosionPNG16, flowPNG, heightPNG16, heightR16, metaJSON, splatPNG } from './export/maps';
import { saveBlob } from './export/download';

type Slider<T> = { key: keyof T; label: string; min: number; max: number; step: number; digits: number };

const TERRAIN_SLIDERS: Slider<TerrainParams>[] = [
  { key: 'freq', label: 'Frequency', min: 0.5, max: 10, step: 0.1, digits: 1 },
  { key: 'octaves', label: 'Octaves', min: 1, max: 10, step: 1, digits: 0 },
  { key: 'lacunarity', label: 'Lacunarity', min: 1.5, max: 3, step: 0.05, digits: 2 },
  { key: 'gain', label: 'Gain', min: 0.2, max: 0.8, step: 0.01, digits: 2 },
  { key: 'ridge', label: 'Ridge mix', min: 0, max: 1, step: 0.01, digits: 2 },
  { key: 'worley', label: 'Cellular mix', min: 0, max: 1, step: 0.01, digits: 2 },
  { key: 'warp', label: 'Domain warp', min: 0, max: 2, step: 0.01, digits: 2 },
  { key: 'island', label: 'Island mask', min: 0, max: 1, step: 0.01, digits: 2 },
  { key: 'exponent', label: 'Height curve', min: 0.5, max: 3, step: 0.01, digits: 2 },
  { key: 'heightScale', label: 'Height scale', min: 5, max: 60, step: 0.5, digits: 1 },
  { key: 'seaLevel', label: 'Sea level', min: 0, max: 0.6, step: 0.005, digits: 3 },
];

const SIM_SLIDERS: Slider<SimParams>[] = [
  { key: 'rain', label: 'Rain intensity', min: 0, max: 0.003, step: 0.00005, digits: 4 },
  { key: 'evap', label: 'Evaporation', min: 0, max: 0.05, step: 0.001, digits: 3 },
  { key: 'kc', label: 'Carry capacity', min: 0.1, max: 5, step: 0.05, digits: 2 },
  { key: 'ks', label: 'Dissolving', min: 0, max: 0.5, step: 0.005, digits: 3 },
  { key: 'kd', label: 'Deposition', min: 0, max: 0.5, step: 0.005, digits: 3 },
  { key: 'flowG', label: 'Flow speed', min: 0.02, max: 0.25, step: 0.005, digits: 3 },
  { key: 'thermal', label: 'Thermal rate', min: 0, max: 0.12, step: 0.005, digits: 3 },
  { key: 'talus', label: 'Talus angle °', min: 10, max: 70, step: 1, digits: 0 },
  { key: 'stormPeriod', label: 'Storm period (0=off)', min: 0, max: 4000, step: 50, digits: 0 },
  { key: 'stormDuty', label: 'Storm duty', min: 0.02, max: 0.5, step: 0.01, digits: 2 },
];

const GEO_SLIDERS: Slider<GeoParams>[] = [
  { key: 'cut', label: 'Cross-section cut', min: 0.02, max: 1, step: 0.005, digits: 2 },
  { key: 'scale', label: 'Layers per height', min: 4, max: 60, step: 1, digits: 0 },
  { key: 'warp', label: 'Layer waviness', min: 0, max: 0.3, step: 0.005, digits: 3 },
  { key: 'dipX', label: 'Dip X', min: -0.6, max: 0.6, step: 0.01, digits: 2 },
  { key: 'dipY', label: 'Dip Y', min: -0.6, max: 0.6, step: 0.01, digits: 2 },
  { key: 'hClay', label: 'Clay softness', min: 0.02, max: 5, step: 0.02, digits: 2 },
  { key: 'hSand', label: 'Sandstone softness', min: 0.02, max: 5, step: 0.02, digits: 2 },
  { key: 'hLime', label: 'Limestone softness', min: 0.02, max: 5, step: 0.02, digits: 2 },
  { key: 'hBasalt', label: 'Basalt softness', min: 0.02, max: 5, step: 0.02, digits: 2 },
];

const STORY_EVERY = 500; // шагов между анализами хроники

const TOOLS: { id: Tool; label: string; hint: string }[] = [
  { id: 'orbit', label: 'Orbit', hint: 'Камера: ЛКМ — вращение, ПКМ — сдвиг' },
  { id: 'rain', label: 'Rain', hint: 'Зажми ЛКМ — полить область' },
  { id: 'raise', label: 'Raise', hint: 'Зажми ЛКМ — поднять рельеф' },
  { id: 'lower', label: 'Lower', hint: 'Зажми ЛКМ — опустить рельеф' },
  { id: 'river', label: 'River', hint: 'Клик — поставить родник (макс. 8)' },
  { id: 'dam', label: 'Dam', hint: 'Протяни линию ЛКМ — построить плотину' },
];

const OVERLAYS: { id: Overlay; label: string }[] = [
  { id: 'none', label: 'Normal' },
  { id: 'erosion', label: 'Erosion' },
  { id: 'velocity', label: 'Velocity' },
  { id: 'diff', label: 'Δ Height' },
];

const TOOL_SLIDERS: Slider<ToolParams>[] = [
  { key: 'radius', label: 'Brush radius', min: 1, max: 20, step: 0.5, digits: 1 },
  { key: 'strength', label: 'Brush strength', min: 0.1, max: 4, step: 0.1, digits: 1 },
  { key: 'riverRate', label: 'River discharge', min: 0.001, max: 0.03, step: 0.001, digits: 3 },
  { key: 'damExtra', label: 'Dam crest +', min: 0, max: 5, step: 0.1, digits: 1 },
];

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<TerrainRenderer | null>(null);
  const s = useStore();
  const { params, sim, geo, tool, overlay, toolParams, cine, cineDuration, simDuringRec, split, tl, tlOn, tlInterval, scrub, meshRes, climate, fx, story, storyOn, running, speed, steps, gpuStatus, gpuMessage, fps, sps } = s;
  const prev = useRef(params);
  const tracker = useRef(new StoryTracker());
  const storyBusy = useRef(false);
  const lastStoryStep = useRef(0);
  const [toast, setToast] = useState<{ id: string; title: string; text: string } | null>(null);

  useEffect(() => {
    const r = new TerrainRenderer(canvasRef.current!);
    const st = useStore.getState();
    r.onFps = st.setFps;
    r.onSteps = st.setSteps;
    r.onCine = st.setCine;
    r.onTimelapse = st.setTl;
    rendererRef.current = r;
    if (import.meta.env.DEV) (window as unknown as { __aqua: TerrainRenderer }).__aqua = r;
    r.init(st.params, st.sim)
      .then(() => st.setGpu('ok'))
      .catch((e: Error) => {
        console.error(e);
        st.setGpu(navigator.gpu ? 'error' : 'unsupported', e.message);
      });
    return () => {
      r.dispose();
      rendererRef.current = null;
    };
  }, []);

  // параметры рельефа: шум → полная регенерация (со сбросом симуляции)
  useEffect(() => {
    const r = rendererRef.current;
    if (!r || gpuStatus !== 'ok') {
      prev.current = params;
      return;
    }
    const noiseChanged = (Object.keys(params) as (keyof TerrainParams)[]).some(
      (k) => k !== 'heightScale' && k !== 'seaLevel' && params[k] !== prev.current[k],
    );
    if (noiseChanged) r.generate(params);
    else r.setRenderParams(params);
    prev.current = params;
  }, [params, gpuStatus]);

  // параметры симуляции и транспорт
  useEffect(() => {
    const r = rendererRef.current;
    if (r) r.split = split;
  }, [split, gpuStatus]);

  useEffect(() => {
    const r = rendererRef.current;
    if (r) r.scrub = scrub;
  }, [scrub, gpuStatus]);

  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;
    r.climate = climate;
    r.fx = fx;
  }, [climate, fx, gpuStatus]);

  // time-lapse: запись вкл/выкл и интервал
  useEffect(() => {
    const r = rendererRef.current;
    if (!r || gpuStatus !== 'ok') return;
    if (tlOn) r.startTimelapse(tlInterval);
    else r.stopTimelapse();
    // интервал меняем «на лету», не перезапуская запись
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tlOn, gpuStatus]);
  useEffect(() => {
    const r = rendererRef.current;
    if (r) r.timelapse.interval = tlInterval;
  }, [tlInterval]);

  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;
    r.geo = geo;
    r.tool = tool;
    r.overlay = overlay;
    r.toolParams = toolParams;
  }, [geo, tool, overlay, toolParams, gpuStatus]);

  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;
    r.sim = sim;
    r.running = running;
    r.speed = speed;
  }, [sim, running, speed, gpuStatus]);

  const years = steps * YEARS_PER_STEP;
  const epoch = Math.floor(steps / 2000) + 1;
  const ready = gpuStatus === 'ok';
  // Story mode: раз в STORY_EVERY шагов читаем состояние и дописываем хронику
  useEffect(() => {
    const r = rendererRef.current;
    if (!r || gpuStatus !== 'ok') return;
    if (steps < lastStoryStep.current) {
      // рельеф сброшен — начинаем историю заново
      tracker.current.reset();
      lastStoryStep.current = 0;
      s.clearStory();
    }
    if (!storyOn || storyBusy.current || steps - lastStoryStep.current < STORY_EVERY) return;
    storyBusy.current = true;
    lastStoryStep.current = steps;
    void r
      .readForExport()
      .then((st) => {
        const p = useStore.getState().params;
        const sim = useStore.getState().sim;
        const evs = tracker.current.analyze(st, steps, { heightScale: p.heightScale, seaLevel: p.seaLevel, stormPeriod: sim.stormPeriod });
        if (evs.length) {
          s.addStory(evs);
          const last = evs[evs.length - 1];
          setToast({ id: last.id + steps, title: `Эпоха ${last.epoch} · ${last.title}`, text: last.text });
          setTimeout(() => setToast((t) => (t && t.id === last.id + steps ? null : t)), 7000);
        }
      })
      .finally(() => {
        storyBusy.current = false;
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [steps, storyOn, gpuStatus]);

  const jumpToStory = (stepAt: number) => {
    if (!tl.count) return;
    let best = 0;
    tl.steps.forEach((st, i) => {
      if (Math.abs(st - stepAt) < Math.abs(tl.steps[best] - stepAt)) best = i;
    });
    s.setRunning(false);
    s.setScrub(best);
  };
  const applyClimate = (id: ClimateId) => {
    const c = CLIMATES[id];
    s.setClimate(id);
    s.setFx('fog', c.fog);
    (Object.keys(c.sim) as (keyof SimParams)[]).forEach((k) => s.setSim(k, c.sim[k] as never));
    (Object.keys(c.geo) as (keyof GeoParams)[]).forEach((k) => s.setGeo(k, c.geo[k] as never));
  };
  const record = () => {
    const r = rendererRef.current;
    if (!r || cine.recording) return;
    const wasRunning = useStore.getState().running;
    if (simDuringRec) s.setRunning(true);
    r.startRecording(cineDuration, (blob) => {
      s.setRunning(wasRunning);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `aquaforge-${Date.now()}.webm`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    });
  };
  const dragSplit = (e: React.PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    el.setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => s.setSplit(Math.min(0.97, Math.max(0.03, ev.clientX / window.innerWidth)));
    const up = () => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
  };
  const exportAs = async (kind: string) => {
    const r = rendererRef.current;
    if (!r) return;
    const st = await r.readForExport();
    const hs = useStore.getState().params.heightScale;
    const sea = useStore.getState().params.seaLevel;
    const stamp = `seed${params.seed}-${steps}st`;
    const mesh = (n: number) => buildMesh(st.terrain, st.n, Math.max(1, Math.round((st.n - 1) / (n - 1))), hs, st.world);
    switch (kind) {
      case 'png16': saveBlob(await heightPNG16(st), `aquaforge-height16-${stamp}.png`); break;
      case 'r16': saveBlob(heightR16(st), `aquaforge-height-${stamp}.r16`); break;
      case 'obj': saveBlob(toOBJ(mesh(meshRes)), `aquaforge-terrain-${stamp}.obj`); break;
      case 'glb': saveBlob(toGLB(mesh(meshRes)), `aquaforge-terrain-${stamp}.glb`); break;
      case 'splat': saveBlob(await splatPNG(st, hs, sea), `aquaforge-splat-${stamp}.png`); break;
      case 'flow': saveBlob(await flowPNG(st), `aquaforge-flow-${stamp}.png`); break;
      case 'erosion': saveBlob(await erosionPNG16(st), `aquaforge-erosion-${stamp}.png`); break;
      case 'meta': saveBlob(metaJSON(st, hs), `aquaforge-meta-${stamp}.json`); break;
    }
  };
  const [hover, setHover] = useState('');
  useEffect(() => {
    const id = window.setInterval(() => {
      const r = rendererRef.current;
      if (!r) return;
      const h = r.hover();
      const st = useStore.getState();
      const txt = h.valid
        ? `h ≈ ${Math.round(h.h * st.params.heightScale * 40)} м · ${h.h > st.params.seaLevel ? 'суша' : 'море'}`
        : '';
      setHover((p) => (p === txt ? p : txt));
    }, 120);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    const ids: Tool[] = ['orbit', 'rain', 'raise', 'lower', 'river', 'dam'];
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' && (el as HTMLInputElement).type !== 'range' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const st = useStore.getState();
      if (e.code === 'Space') {
        e.preventDefault();
        st.setRunning(!st.running);
      } else if (e.key === 'r' || e.key === 'R') {
        st.setRunning(false);
        rendererRef.current?.generate(st.params);
      } else if (e.key >= '1' && e.key <= '6') {
        st.setTool(ids[+e.key - 1]);
      } else if (e.key === '[') {
        st.setToolParam('radius', Math.max(1, st.toolParams.radius - 1));
      } else if (e.key === ']') {
        st.setToolParam('radius', Math.min(20, st.toolParams.radius + 1));
      } else if (e.key === 'h' || e.key === 'H') {
        document.querySelector('.panel')?.classList.toggle('hidden');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const toggleSection = (e: React.MouseEvent) => {
    const h = e.target as HTMLElement;
    if (h.tagName !== 'H3') return;
    const closed = h.classList.toggle('closed');
    let n = h.nextElementSibling;
    while (n && n.tagName !== 'H3') {
      n.classList.toggle('folded', closed);
      n = n.nextElementSibling;
    }
  };

  const reset = () => {
    s.setRunning(false);
    rendererRef.current?.generate(useStore.getState().params);
  };

  return (
    <div className="app">
      <canvas ref={canvasRef} className="viewport" />

      <aside className="panel left" onClick={toggleSection}>
        <h1>AQUAFORGE</h1>
        <p className="sub">Phase 1 · grid hydraulic erosion</p>

        <h3>Terrain</h3>
        <div className="row">
          <label>Seed</label>
          <input
            type="number"
            value={params.seed}
            onChange={(e) => s.setParam('seed', Math.max(0, Math.floor(+e.target.value || 0)))}
          />
          <button className="icon" title="Случайный seed" onClick={s.randomize}>
            <Dices size={16} />
          </button>
        </div>
        {TERRAIN_SLIDERS.map((sl) => (
          <SliderRow
            key={sl.key}
            sl={sl}
            value={params[sl.key] as number}
            onChange={(v) => s.setParam(sl.key, v as never)}
          />
        ))}

        <h3>Climate</h3>
        <div className="toggles three">
          {(Object.keys(CLIMATES) as ClimateId[]).map((id) => (
            <button key={id} className={climate === id ? 'on' : ''} title={CLIMATES[id].hint} onClick={() => applyClimate(id)}>
              {CLIMATES[id].label.split(' ')[0]}
            </button>
          ))}
        </div>
        <p className="hint tight">{CLIMATES[climate].hint}</p>
        <SliderRow
          sl={{ key: 'tod', label: 'Time of day (sunrise → sunset)', min: 0.03, max: 0.97, step: 0.01, digits: 2 } as Slider<{ tod: number }>}
          value={fx.tod}
          onChange={(v) => s.setFx('tod', v)}
        />
        <SliderRow
          sl={{ key: 'fog', label: 'Fog density', min: 0, max: 0.015, step: 0.0005, digits: 4 } as Slider<{ fog: number }>}
          value={fx.fog}
          onChange={(v) => s.setFx('fog', v)}
        />
        <SliderRow
          sl={{ key: 'trees', label: 'Forest density', min: 0, max: 1.5, step: 0.05, digits: 2 } as Slider<{ trees: number }>}
          value={fx.trees}
          onChange={(v) => s.setFx('trees', v)}
        />
        <SliderRow
          sl={{ key: 'bloom', label: 'Bloom', min: 0, max: 2, step: 0.05, digits: 2 } as Slider<{ bloom: number }>}
          value={fx.bloom}
          onChange={(v) => s.setFx('bloom', v)}
        />

        <h3>God tools</h3>
        <div className="toggles three">
          {TOOLS.map((t) => (
            <button key={t.id} className={tool === t.id ? 'on' : ''} title={t.hint} onClick={() => s.setTool(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        <p className="hint tight">{TOOLS.find((t) => t.id === tool)?.hint}</p>
        {TOOL_SLIDERS.map((sl) => (
          <SliderRow
            key={sl.key}
            sl={sl}
            value={toolParams[sl.key] as number}
            onChange={(v) => s.setToolParam(sl.key, v as never)}
          />
        ))}
        <div className="toggles">
          <button onClick={() => rendererRef.current?.clearSources()}>Clear springs</button>
          <button onClick={() => rendererRef.current?.clearWater()}>Clear water</button>
        </div>

        <h3>Story</h3>
        <div className="toggles">
          <button
            className={storyOn ? 'on' : ''}
            onClick={() => {
              const next = !storyOn;
              s.setStoryOn(next);
              if (next && !tlOn) s.setTlOn(true); // хроника ссылается на кадры time-lapse
            }}
          >
            {storyOn ? '■ Narrating' : '✎ Narrate'}
          </button>
          <button
            disabled={story.length === 0}
            onClick={() => saveBlob(new Blob([storyToMarkdown(story)], { type: 'text/markdown' }), `aquaforge-story-${params.seed}.md`)}
          >
            Export .md
          </button>
        </div>
        <div className="story">
          {story.length === 0 && <p className="hint tight">Включи «Narrate» и запусти симуляцию — события появятся здесь.</p>}
          {[...story].reverse().map((e) => (
            <button key={e.id} className="story-item" onClick={() => jumpToStory(e.step)} title="Перейти к этому моменту (time-lapse)">
              <b>Эпоха {e.epoch} · {(e.step * YEARS_PER_STEP).toLocaleString('ru-RU')} лет</b>
              <span className="st">{e.title}</span>
              <span>{e.text}</span>
            </button>
          ))}
        </div>

        <h3>Time-lapse</h3>
        <div className="toggles">
          <button className={tlOn ? 'on rec' : ''} onClick={() => s.setTlOn(!tlOn)}>
            {tlOn ? '■ Recording' : '● Record'}
          </button>
          <button
            disabled={tl.count === 0}
            onClick={() => {
              s.setScrub(null);
              rendererRef.current?.clearTimelapse();
            }}
          >
            Clear ({tl.count})
          </button>
        </div>
        <SliderRow
          sl={{ key: 'tlInterval', label: 'Snapshot every, steps', min: 20, max: 2000, step: 10, digits: 0 } as Slider<{ tlInterval: number }>}
          value={tlInterval}
          onChange={(v) => s.setTlInterval(v)}
        />
        <p className="hint tight">Снимки рельефа лежат на GPU (до 96). Скруббер появится в нижней панели.</p>

        <h3>Export</h3>
        <div className="toggles">
          <button onClick={() => exportAs('png16')}>Height PNG 16-bit</button>
          <button onClick={() => exportAs('r16')}>Height RAW .r16</button>
          <button onClick={() => exportAs('obj')}>Mesh .obj</button>
          <button onClick={() => exportAs('glb')}>Mesh .glb</button>
          <button onClick={() => exportAs('splat')}>Splatmap</button>
          <button onClick={() => exportAs('flow')}>Flowmap</button>
          <button onClick={() => exportAs('erosion')}>Erosion map</button>
          <button onClick={() => exportAs('meta')}>Scale .json</button>
        </div>
        <SliderRow
          sl={{ key: 'meshRes', label: 'Mesh resolution', min: 65, max: 512, step: 1, digits: 0 } as Slider<{ meshRes: number }>}
          value={meshRes}
          onChange={(v) => s.setMeshRes(v)}
        />

        <h3>Cinematic</h3>
        <div className="toggles">
          <button disabled={cine.recording} onClick={() => rendererRef.current?.addKeyframe()}>
            + Keyframe ({cine.keys})
          </button>
          <button disabled={cine.recording} onClick={() => rendererRef.current?.clearKeyframes()}>
            Clear keys
          </button>
          <button
            disabled={cine.recording}
            className={cine.playing && !cine.recording ? 'on' : ''}
            onClick={() => (cine.playing ? rendererRef.current?.stopCine() : rendererRef.current?.playCine(cineDuration))}
          >
            {cine.playing && !cine.recording ? '■ Stop' : '▶ Preview'}
          </button>
          <button className={cine.recording ? 'on rec' : ''} onClick={() => (cine.recording ? rendererRef.current?.stopCine() : record())}>
            {cine.recording ? `■ ${Math.round(cine.progress * 100)}%` : '● Record WebM'}
          </button>
        </div>
        <SliderRow
          sl={{ key: 'cineDuration', label: 'Duration, s', min: 4, max: 60, step: 1, digits: 0 } as Slider<{ cineDuration: number }>}
          value={cineDuration}
          onChange={(v) => s.setCineDuration(v)}
        />
        <label className="check">
          <input type="checkbox" checked={simDuringRec} onChange={(e) => s.setSimDuringRec(e.target.checked)} />
          Run simulation while recording
        </label>
        <p className="hint tight">Без ключевых кадров — автопролёт вокруг острова. Ключи: выбери вид (Orbit) и жми «+ Keyframe».</p>

        <h3>Overlay</h3>
        <div className="toggles">
          {OVERLAYS.map((o) => (
            <button key={o.id} className={overlay === o.id ? 'on' : ''} onClick={() => s.setOverlay(o.id)}>
              {o.label}
            </button>
          ))}
          <button className={split !== null ? 'on' : ''} onClick={() => s.setSplit(split === null ? 0.5 : null)}>
            Before / After
          </button>
        </div>

        <h3>Geology & view</h3>
        <div className="toggles">
          <button className={geo.view === 'biome' ? 'on' : ''} onClick={() => s.setGeo('view', 'biome')}>Biomes</button>
          <button className={geo.view === 'geology' ? 'on' : ''} onClick={() => s.setGeo('view', 'geology')}>Geology</button>
          <button className={geo.diorama ? 'on' : ''} onClick={() => s.setGeo('diorama', !geo.diorama)}>Diorama</button>
          <button className={geo.strataErosion ? 'on' : ''} onClick={() => s.setGeo('strataErosion', !geo.strataErosion)} title="Мягкие слои размываются быстрее">Hardness</button>
        </div>
        {GEO_SLIDERS.map((sl) => (
          <SliderRow
            key={sl.key}
            sl={sl}
            value={geo[sl.key] as number}
            onChange={(v) => s.setGeo(sl.key, v as never)}
          />
        ))}

        <h3>Hydrology</h3>
        {SIM_SLIDERS.map((sl) => (
          <SliderRow
            key={sl.key}
            sl={sl}
            value={sim[sl.key] as number}
            onChange={(v) => s.setSim(sl.key, v as never)}
          />
        ))}
        <p className="hint">Orbit: ЛКМ — вращение, ПКМ/Shift — сдвиг. С инструментом: ЛКМ — применить, ПКМ — вращение, Shift+ЛКМ — сдвиг. Колесо — зум.</p>
      </aside>

      <footer className="timeline">
        <button className="icon big" disabled={!ready} onClick={() => s.setRunning(!running)} title={running ? 'Pause' : 'Start'}>
          {running ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button className="icon big" disabled={!ready || running} onClick={() => rendererRef.current?.stepOnce()} title="Step">
          <SkipForward size={18} />
        </button>
        <button className="icon big" disabled={!ready} onClick={reset} title="Reset">
          <RotateCcw size={18} />
        </button>
        <button
          className={`icon big ${sim.rainOn ? 'on' : ''}`}
          onClick={() => s.setSim('rainOn', !sim.rainOn)}
          title="Дождь вкл/выкл"
        >
          <CloudRain size={18} />
        </button>
        <div className="speed">
          <label>
            Speed <span>{speed}×</span>
          </label>
          <input type="range" min={1} max={100} step={1} value={speed} onChange={(e) => s.setSpeed(+e.target.value)} />
        </div>
        {tl.count > 0 && (
          <div className="scrub">
            <label>
              Time-lapse{' '}
              <span>
                {scrub === null ? 'LIVE' : `${(tl.steps[Math.min(scrub, tl.count - 1)] * YEARS_PER_STEP).toLocaleString('ru-RU')} лет`}
              </span>
            </label>
            <input
              type="range"
              min={0}
              max={tl.count}
              step={1}
              value={scrub === null ? tl.count : scrub}
              onChange={(e) => {
                const v = +e.target.value;
                if (v >= tl.count) s.setScrub(null);
                else {
                  s.setRunning(false);
                  s.setScrub(v);
                }
              }}
            />
          </div>
        )}
        <div className="stats">
          <div><b>Epoch {epoch}</b></div>
          <div>{years.toLocaleString('ru-RU')} лет · {steps.toLocaleString('ru-RU')} шагов</div>
        </div>
      </footer>

      {split !== null && (
        <div className="split" style={{ left: `${split * 100}%` }} onPointerDown={dragSplit}>
          <span className="lab l">BEFORE</span>
          <span className="lab r">AFTER</span>
          <div className="grip" />
        </div>
      )}

      {toast && (
        <div className="toast" key={toast.id}>
          <b>{toast.title}</b>
          <span>{toast.text}</span>
        </div>
      )}

      {hover && <div className="hover">{hover}</div>}
      <div className="hud">{ready ? `${fps} FPS · ${sps.toLocaleString('ru-RU')} шагов/с · 512²` : ''}</div>

      {(gpuStatus === 'unsupported' || gpuStatus === 'error') && (
        <div className="overlay">
          <h2>WebGPU недоступен</h2>
          <p>{gpuMessage}</p>
        </div>
      )}
    </div>
  );
}

function SliderRow<T>({
  sl,
  value,
  onChange,
}: {
  sl: Slider<T>;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="slider">
      <label>
        {sl.label}
        <span>{value.toFixed(sl.digits)}</span>
      </label>
      <input
        type="range"
        min={sl.min}
        max={sl.max}
        step={sl.step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
      />
    </div>
  );
}
