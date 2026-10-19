# AQUAFORGE

Интерактивная 3D-песочница гидравлической эрозии в браузере (WebGPU + React).

- ТЗ: [docs/TZ.md](docs/TZ.md)
- Статус: **Фазы 0–4, Cinematic, Before/After, полировка и Story mode готовы** — GPU-эрозия (~2500 шагов/с), слои пород, God-tools, оверлеи, time-lapse, запись WebM, экспорт, HDR+bloom, вода, климат-пресеты, геологическая хроника. Сборка: `npm run build` → `dist/` (статический хостинг).

## Запуск

```bash
npm install
npm run dev     # http://localhost:5183 (нужен Chrome/Edge с WebGPU)
```

## Структура

- `src/gpu/shaders.ts` — WGSL: noise compute + terrain render
- `src/gpu/TerrainRenderer.ts` — устройство, пайплайны, камера, цикл кадра
- `src/store.ts` — Zustand-состояние параметров
- `src/App.tsx` — UI-панель
