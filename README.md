# AQUAFORGE

**Hydraulic erosion in your browser: carve a landscape with rain and rivers, then watch it age - all on the GPU.**

AQUAFORGE is an interactive 3D terrain sandbox. A WebGPU compute pipeline simulates water flow,
erosion, sediment transport and thermal creep on a heightfield in real time (~2,500 simulation steps
per second on an integrated AMD RDNA3 GPU), while layered rock, water and lighting are rendered live.
Paint rain, dig rivers, build dams, then scrub through time, cut a cross-section, record a cinematic
fly-through and export the result as heightmaps and meshes.

> Needs a browser with WebGPU (Chrome or Edge).

## Features

- **GPU erosion.** Grid shallow-water simulation (virtual pipes, after Mei et al. 2007) with
  erosion/deposition, sediment advection and thermal erosion - five compute passes per step, workgroup
  shared-memory tiles with halo, f16 storage where the device supports it.
- **Geology.** Four rock types (clay, sandstone, limestone, basalt) as tilted, bent strata computed
  analytically: soft layers wash out, basalt holds. Biome and Geology views, a 4-sided **diorama**
  with exposed layers and a movable **cross-section** plane.
- **God tools.** Rain, raise/lower, **dam** (a line whose breach emerges from the erosion itself) and
  up to 8 **river sources** with adjustable discharge, all working on a paused simulation too.
- **Overlays.** Erosion heat (wash-out vs. deposition), velocity strokes, height difference from the
  original terrain.
- **Time-lapse.** Snapshots go GPU-to-GPU into a 96-frame ring - no CPU readback - and a scrubber moves
  between live and any past year.
- **Before / After.** A draggable split view of the original and the eroded terrain with a shared camera.
- **Cinematic camera.** Keyframed Catmull-Rom flight (or an automatic orbit), recorded to **WebM** frame
  by frame with the simulation optionally running in shot.
- **Climate presets and Story mode.** Arid canyon, tropical monsoon, temperate... and an auto-generated
  "geological chronicle" of what the simulation did.
- **Export.** 16-bit height PNG, RAW `.r16`, `.obj` and `.glb` meshes, splatmap, flowmap, erosion map
  and a scale `.json` for game-engine landscape import.

## Engineering notes

- **Raw WebGPU + WGSL**, no 3D framework: terrain, water, strata, lighting, post (HDR + bloom) and tool
  shaders are written by hand (`src/gpu/`).
- **Performance work is documented:** 900 -> ~2,560 steps/s by tiling into workgroup memory, fusing
  passes (7 -> 5) and moving flux and sediment to f16. The remaining limit is memory bandwidth.
- **GPU picking:** a single-thread raymarch over the terrain reads back 16 bytes with 1-2 frames of
  latency - no CPU copy of the heightfield.
- **React + Zustand** for the UI; simulation parameters flow to uniform buffers without re-rendering.
- Design log and decisions with measurements: [`docs/TZ.md`](docs/TZ.md) (Russian), section 10.

## Run it

```bash
npm install
npm run dev     # http://localhost:5183
npm run build   # static output in dist/
```

## Layout

| Path | What |
|---|---|
| `src/gpu/simShader.ts` | Erosion compute passes (WGSL) |
| `src/gpu/terrainShader.ts`, `strataShader.ts`, `lightShader.ts`, `postShader.ts` | Rendering |
| `src/gpu/toolsShader.ts` | God-tool brushes |
| `src/gpu/TerrainRenderer.ts` | Device, pipelines, camera, frame loop, time-lapse, recording |
| `src/export/` | PNG16, R16, OBJ/GLB, splat/flow/erosion maps |
| `src/store.ts`, `src/App.tsx` | State and UI panels |