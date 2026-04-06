# Density Map Scatter — TT Metal SFPU

This document describes the **density map scatter** programming example: what it implements,
how data flows through host and device, how **accuracy** is measured, and a full record of
the **geometry-path accuracy investigation** (observations, hypotheses tested, what was ruled
out, and what remains open).

---

## Purpose

The example compares a **Tenstorrent SFPU** implementation of DREAMPlace-style **triangle
density map** accumulation against a serial CPU reference on the **same** synthetic workload:

- Movable cells with DREAMPlace semantics: clamped sizes, centering offsets, per-cell ratio
  (here fixed to `1`).
- For each cell, bins in a fixed **impact window** are enumerated (truncating bin indices like
  DREAMPlace).
- Per **(cell, bin)** task, an overlap **area** is computed and scattered into a 2D density map.

Two device variants isolate **"SFPU multiply only"** from **"full overlap geometry on device"**.

---

## CMake targets

| Target | Macros | Role |
|--------|--------|------|
| `metal_example_density_map_scatter` | default | Small regression: 5k cells, 256² bins, 1 core |
| `metal_example_density_map_scatter_benchmark` | `DMS_EXE_BENCHMARK=1` | 200k cells, 512², 1000² domain |
| `metal_example_density_map_scatter_benchmark_large` | `DMS_EXE_BENCHMARK_LARGE=1` | 500k cells, 2048², 3000² domain; **pre-multiply** path |
| `metal_example_density_map_scatter_benchmark_large_geometry` | `DMS_EXE_BENCHMARK_LARGE=1`, `DMS_GEOMETRY_DEVICE=1` | Same large workload; **geometry** path |

Source file: `density_map_scatter.cpp` (single translation unit; behavior selected by macros).

Build all four:
```bash
cd tt-metal
ninja -C build \
  metal_example_density_map_scatter \
  metal_example_density_map_scatter_benchmark \
  metal_example_density_map_scatter_benchmark_large \
  metal_example_density_map_scatter_benchmark_large_geometry
```

---

## Host workflow (`density_map_scatter.cpp`)

1. **Generate cells**
   Uniform position and size; `sx_clamped = max(sx, BIN_SIZE_X)`; `offset = (clamped − raw) / 2`.

2. **Enumerate tasks**
   For each cell, bin index ranges `[bxl_idx, bxh_idx)` × `[byl_idx, byh_idx)` via integer
   truncation matching DREAMPlace. Each valid slot gets `(k, h)` and either:
   - **Pre-multiply:** host pre-computes `t_px`, `t_py` with `max(0, min(…) − max(…))`.
   - **Geometry:** host uploads raw bin edges `t_cxl`, `t_cxr`, `t_bxl`, `t_cyl`, `t_cyr`, `t_byl`.

3. **CPU reference**
   Serial, deterministic loop: `ref_area[i]`, accumulated into `ref_density[bin]`.
   - Pre-multiply: `ref_area = t_px * ratio * t_py`.
   - Geometry: `max(0,px) * max(0,py)` from same min/max formula, matching SFPU clamping.
   - Ratio is always 1.0 in this benchmark.

4. **DRAM buffers**
   Replicated mesh buffers; upload = non-blocking writes + `Finish(cq)`.

5. **Program: reader → compute → writer**
   RISCV_0 pulls tiles DRAM→CB; SFPU kernel runs; RISCV_1 writes per-task area tiles back.

6. **Device kernel timing**
   `EnqueueMeshWorkload` + `Finish(cq)` — host wall-clock. Excludes upload/readback.

7. **Readback + scatter**
   Blocking read of `tt_areas`; host scatters using `t_bin_k`, `t_bin_h` (same as reference).

8. **Accuracy check**
   See [Accuracy metrics](#accuracy-metrics).

---

## Device paths

### Pre-multiply (`overlap_compute.cpp`)

- **Inputs (3):** `px`, `py`, `ratio` tiles.
- **Compute:** One SFPU ternary pass: `area = px * ratio * py`.
- **Status: PASSES** — relative L2 ~1.7e-4, well under the 5% gate.

### Geometry (`overlap_compute_geometry.cpp` + `read_tiles_geometry.cpp`)

- **Inputs (6):** `cxl`, `cxr`, `bxl`, `cyl`, `cyr`, `byl`.
  `ratio` removed from upload — hardcoded `1.0` on device (saves ~200 MB/run).
- **Compute on device:**
  1. `px = clamp(min(cxr, bxl+BIN_SIZE_X) − max(cxl, bxl), 0, ∞)`
  2. `py = clamp(min(cyr, byl+BIN_SIZE_Y) − max(cyl, byl), 0, ∞)`
  3. `area = px * py`
- **JIT defines** (host → compile-time constants in kernel):
  `DMS_NUM_BINS_KERNEL`, `DMS_DOMAIN_WIDTH/HEIGHT`,
  `DMS_BIN_SIZE_X_F`, `DMS_BIN_SIZE_Y_F` (exact float literals, same as host).
- **Status: FAILS** — relative L2 ~0.18, integrated area error ~1.8%.

---

## Accuracy metrics

### Per-task strict check

For each valid task: `|tt_areas[i] − ref_area[i]|` vs `STRICT_EPS = 5e-4` (absolute + relative).
Counts mismatches and prints up to 3 worst samples.

### Density map norms (main gate)

After scattering `tt_areas → tt_density`:

| Metric | Definition | Pass threshold |
|--------|------------|----------------|
| **Relative L2** | `‖tt_density − ref_density‖₂ / ‖ref_density‖₂` | ≤ 5% |
| **Integrated relative error** | `|Σtt − Σref| / max(Σref, 1)` | ≤ 0.2% |

### Observed numbers (500k cells, 2048² grid, seed=2025)

| Path | Relative L2 | Integrated err | Result |
|------|-------------|---------------|--------|
| Pre-multiply | ~1.7e-4 | ~1.3e-4 | ✓ PASS |
| Geometry | ~1.79e-1 | ~1.82e-2 | ✗ FAIL |

---

## Geometry accuracy investigation

### What was confirmed NOT to be the cause

Each hypothesis was tested; outcome shown.

**H1 — JIT `BIN_SIZE` mismatch (device float ≠ host float)**
- **Ruled out.** Host instrumentation printed `DMS_BIN_SIZE_X_F = DMS_BIN_SIZE_Y_F = 1.46484375f`
  (= 3000/2048 exactly), matching the host constexpr. The fallback integer-division path is
  bypassed because `DMS_BIN_SIZE_X_F` / `DMS_BIN_SIZE_Y_F` are always passed explicitly.

**H2 — Host scatter / bin indices wrong**
- **Ruled out.** Pre-multiply uses the **same** host `t_bin_k`, `t_bin_h`, `t_valid` arrays and
  passes. The scatter step is correct for both paths.

**H3 — CPU reference sign-convention mismatch (raw DREAMPlace vs clamped)**
- **Ruled out.** The CPU reference under `DMS_GEOMETRY_DEVICE` already uses
  `max(0,px)*max(0,py)`, matching SFPU's clamp-to-zero. The discrepancy persists.

**H4 — `ratio ≠ 1` on geometry path**
- **Ruled out.** Benchmark uses `ratio = ones`. Ratio was removed from the upload entirely;
  device hardcodes `1.0`. Not the cause for this workload.

**H5 — Stuck device process holding the card lock**
- **Separate issue, resolved.** Multiple dead instances of the geometry binary (and `tt-smi`)
  were holding `CHIP_IN_USE_0_PCIe`. Killed with `pkill -9`. Benchmark now runs cleanly.
  Firmware init failure (`Device 0 init: failed to initialize FW`) occurs when stale processes
  hold the lock; this is not an accuracy bug.

### What was tried (and the outcome)

The investigation focused on `overlap_compute_geometry.cpp`. All attempts below used a **cleared
JIT cache** (`rm -rf ~/.cache/tt-metal-cache/*`) before running.

---

**Attempt 1 — Original kernel (baseline)**

```cpp
// Final area pass:
run_area(/*px=*/3, /*py=*/4, /*area=*/7);
// where run_area used ternary LLK with dummy_in2 == out == dst 7
```

Result: rel L2 = **0.1789**, 8.3M / 13M mismatches, `max|err| = 2.15`.
Worst task: `expected=2.15, tt=0`. Some tasks completely zero, others ~50% off.

Key evidence from host log:
```json
{"sum_ref": 1.80802e7, "sum_tt": 1.78439e7, "sum_ratio": 0.9869}
{"i": 9371, "ref": 2.14577, "tt": 0, "abs_err": 2.14577}
```

**Finding:** The ternary LLK receives `dst_index_in2` and `dst_index_out` as separate arguments;
passing the same tile index for both (`in2 = out = 7`) is potentially undefined behavior on
the hardware. When `out == in2`, the LLK start-sync sets `dst_write_addr` to tile 0
(`_llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0)`) regardless of operand indices —
reading operand tiles relative to that base while writing to `out`. The custom face lambda
still reads from absolute indices (`dst_reg[d * N + i]`), but the LLK hardware state may not
be consistent with that when `in2 == out`.

---

**Attempt 2 — Replace ternary final pass with binary SFPU mul (`mul_binary_tile`)**

Added `#include "api/compute/eltwise_binary_sfpu.h"`, called `mul_binary_tile_init()` before
the loop, then `mul_binary_tile(3, 4, 7)`.

Result: **same numbers** as baseline (rel L2 = 0.1789).

**Finding:** `calculate_sfpu_binary_mul` walks dest registers using `sfpi::dst_reg++` per
iteration (relative stepping), while `px_tile_face`/`py_tile_face` use `dst_reg[d*N + i]`
(absolute index). The binary mul reads the wrong lanes — `px` and `py` were written at
absolute positions but the binary read expects sequential layout. Values come out wrong.
Additionally, `mul_binary_tile_init()` reconfigures SFPU addr-mod registers, which can
interfere with the ternary `run_px`/`run_py` passes called later in the same tile loop.

---

**Attempt 3 — Custom binary SFPU face matching ternary layout**

Replaced final pass with `_llk_math_eltwise_binary_sfpu_params_<true>(mul_px_py_face, ...)`,
where `mul_px_py_face` uses `dst_reg[d*N + i]` — same indexing as `px_tile_face`.
Still called `mul_binary_tile_init()` before the loop.

Result: **same numbers** as baseline.

**Finding:** `mul_binary_tile_init()` → `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`
→ `eltwise_binary_sfpu_configure_addrmod` → reprograms `ADDR_MOD_7`. The ternary start/done
helpers also use `ADDR_MOD_7` via `math::set_addr_mod_base()`. Calling binary init before
ternary passes may corrupt the addr-mod state used by `run_px`/`run_py`. After removing
`mul_binary_tile_init()`, result is still the same — meaning the custom binary face also
doesn't fix things, likely because the binary LLK start/done sequence sets `dst_write_addr`
from `dst_index=0` just like the ternary one.

---

**Attempt 4 — Move px/py to high dest slots (5, 6) to avoid FP32 dest banking**

Hypothesis: in FP32 dest mode on Wormhole, `DEST_REGISTER_HALF_SIZE = BIT32_DEST_REGISTER_HALF_SIZE * 2`;
tiles in slots 3/4 might alias with unpack-destination staging when y-group inputs reload to 0–2.
Tried `px → dst 6`, `py → dst 5`.

Result: **catastrophically worse** — rel L2 = **813**, total area = **11.4 billion** (should be
18 million). Garbage values written.

**Finding:** High dest indices (5, 6) exceed safe tile slots in FP32 half-sync mode.
`get_dest_max_tiles<SyncHalf, FP32, Tile32x32>() = DEST_REGISTER_HALF_SIZE/2 / 64 = 4`.
Only tiles 0–3 are valid in half-sync FP32 mode. Using slots 5 or 6 accesses out-of-bounds
dest memory, producing garbage. Reverted immediately.

---

**Attempt 5 — Py-before-px compute order**

Hypothesis: unpacking y-group (cyl, cyr, byl) into dst 0–2 after px was stored in dst 3 might
corrupt dst 3 through FP32 banking. Tried computing py first (stored to dst 4), then px (dst 3).

Result: **catastrophically worse** — rel L2 = **810**, total area = **11.3 billion**.

**Finding:** Same FP32 dest slot overflow issue. With py-first order the _area_ pass reads py
from dst 4 before px is computed; intermediate state is undefined. Reverted.

---

**Attempt 6 — Ternary area pass with non-aliasing dummy operand (current state)**

Use ternary `run_area(px=3, py=4, dummy=2, out=7)`. Slot 2 holds `byl` after `run_py`;
it is a valid, populated dest tile, different from `out`. The area lambda ignores `dummy`.

```cpp
inline void area_tile_face(
    const uint32_t d_px, const uint32_t d_py, const uint32_t /*d_dummy*/, const uint32_t d_area) {
    for (uint32_t i = 0; i < 8; i++) {
        dst_reg[d_area * N + i] = dst_reg[d_px * N + i] * dst_reg[d_py * N + i];
    }
}
// called as:
run_area(/*px=*/3, /*py=*/4, /*dummy_byl=*/2, /*area=*/7);
```

Result: **rel L2 = 0.1789** — identical to baseline.

**Finding:** Task 0's `expected=0.824359, tt=0.448502` is stable across all attempts. This
determinism suggests `px` or `py` for task 0 is systematically wrong before the multiply,
not that the multiply itself is wrong. The ternary `run_px`/`run_py` faces compute:

```
px = clamp(min(cxr, bxl + BIN_SIZE_X) − max(cxl, bxl), 0, ∞)
```

But `_llk_math_eltwise_ternary_sfpu_start_` always calls:
```cpp
math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
```
i.e., it anchors the **hardware dest write pointer at tile 0**, regardless of the output tile
index `d_px/d_py/d_area`. The face lambda uses `dst_reg[d * N + i]` (SFPI relative to current
write ptr), so `dst_reg[0 * N + i]` is tile 0 relative to the write base — which is tile 0 in
global dest. `dst_reg[1 * N + i]` is tile 1 in global dest, etc.

**The critical question for an expert:** does `_llk_math_eltwise_ternary_sfpu_start_(0)` always
set the base to tile 0 unconditionally, meaning tile-indexed reads (`dst_reg[d*N+i]`) are
absolute and do NOT automatically shift for the output tile? Or is there a face-pointer
hardware register (`DEST_TARGET_REG_CFG_MATH_Offset`) that moves under `TTI_SETRWC` between
faces, and our face callback reads the *wrong* shifted position for intermediate tiles (3, 4)?

The `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` increments happen in the _params_ wrapper
between face calls. If those increments shift the hardware base that `dst_reg[d*N+i]` is
relative to, then **`px`/`py` stored in dst 3/4 after face 0 would be read from the wrong
offset in faces 1–3 of `run_area`**, explaining the systematic partial match (some elements
correct, others wrong in a per-face pattern).

---

### Open questions for SFPU / LLK experts

1. **Is `dst_reg[d*N + i]` in SFPI an absolute address, or relative to the per-face hardware
   dest-write pointer set by `TTI_SETRWC`?**
   If relative: each `TTI_SETRWC ... SET_D` between faces shifts the effective base, and writing
   to dst slot 3 in face 0 does NOT mean reading from dst slot 3 in face 1 with the same index.

2. **What is the correct pattern for a 3-pass SFPU kernel that reads intermediate results
   across sequential ternary calls?** Should intermediate tiles be re-`copy_tile`d from dest
   back to a CB and unpacked fresh for each pass?

3. **What is the maximum safe `dst_tile_index` for `tile_regs_acquire` / ternary passes in
   FP32 half-sync mode on Wormhole B0?**
   Experiment shows tiles 5–7 produce garbage in FP32 mode. Only 0–3 appear stable.

4. **Does calling `_llk_math_eltwise_ternary_sfpu_start_(0)` with hard-coded `0` break
   multi-tile kernels?** The caller passes `dst_index_in0/1/2/out` to the params template but
   start always anchors to 0. If this is intentional, how should custom faces read their
   operands?

---

## Python end-to-end benchmark

`DREAMPlace/scripts/benchmark_sfpu_density_map.py` — runs both large TT binaries as subprocesses,
times CPU OMP path, runs numpy formula accuracy, and prints a combined report.

```bash
cd tt-metal && ninja -C build \
  metal_example_density_map_scatter_benchmark_large \
  metal_example_density_map_scatter_benchmark_large_geometry

source DREAMPlace/venv/bin/activate
cd DREAMPlace
python scripts/benchmark_sfpu_density_map.py
```

Timing summary from last successful run:

| Path | Upload | Kernel | Readback | e2e |
|------|--------|--------|----------|-----|
| CPU OMP | — | 140.8 ms | — | — |
| Pre-multiply | 74.3 ms | **3.9 ms** | 40.4 ms | 118.6 ms |
| Geometry | 148.7 ms | **8.2 ms** | 43.0 ms | 199.8 ms |

Pre-multiply kernel speedup vs CPU: **36.5×**. Geometry: **17.2×** (but accuracy fails).

---

## File map

| Path | Role |
|------|------|
| `density_map_scatter.cpp` | Host: generation, enumeration, CPU ref, buffers, program build, timing, accuracy |
| `CMakeLists.txt` | All four targets and compile definitions |
| `kernels/compute/overlap_compute.cpp` | Pre-multiply SFPU kernel — **passing** |
| `kernels/compute/overlap_compute_geometry.cpp` | Geometry SFPU kernel — **open accuracy bug** |
| `kernels/dataflow/read_tiles.cpp` | Reader for 3 inputs (pre-multiply) |
| `kernels/dataflow/read_tiles_geometry.cpp` | Reader for 6 inputs (geometry) |
| `kernels/dataflow/write_tile.cpp` | Shared output writer |
| `DOCUMENTATION.md` | This file |
| `PERFORMANCE_REPORT.md` | Detailed timing breakdown |
| `PROFILING_REPORT.md` | TT device profiler / Tracy data |
| `REPRODUCE.md` | Step-by-step expert reproduction guide |

---

## Summary of status

| Item | Status |
|------|--------|
| Host enumeration & scatter | ✓ Correct (verified against numpy reference) |
| CPU reference formula | ✓ Correct (verified against DREAMPlace C++ extension) |
| JIT BIN_SIZE token accuracy | ✓ Confirmed exact match to host float |
| Pre-multiply SFPU kernel | ✓ Passes (rel L2 ~1.7e-4) |
| Geometry kernel px/py faces | ? Produces systematic ~50% errors on some tasks |
| Geometry kernel area multiply | ? Appears correct in isolation; suspect px or py is wrong |
| Per-face dst addressing in ternary LLK | ❓ **Root cause candidate — needs Metal/LLK expert** |
