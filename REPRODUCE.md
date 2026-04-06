# Reproducing the Geometry-Path Accuracy Bug

This guide gives an **expert** everything needed to reproduce, explore, and (ideally) fix
the geometry SFPU kernel accuracy issue on a Wormhole B0 / N150 Tenstorrent device.

---

## Hardware and software requirements

| Requirement | Value |
|---|---|
| Device | Tenstorrent Wormhole N150 (or N300) |
| OS | Ubuntu 22.04 LTS |
| Driver | `tt-kmd` ≥ 1.30 installed and loaded |
| TT-Metal | Cloned from https://github.com/tenstorrent/tt-metal |
| Python | 3.10+ with `numpy`, `torch` (for Python benchmark only) |

---

## What you are given (the `Transport/` folder)

```
Transport/
├── benchmark_sfpu_density_map.py        # end-to-end Python benchmark
└── density_map_scatter/
    ├── CMakeLists.txt                   # four build targets
    ├── density_map_scatter.cpp          # host code (single TU, macro-gated)
    ├── DOCUMENTATION.md                 # background + full investigation record
    ├── PERFORMANCE_REPORT.md
    ├── PROFILING_REPORT.md
    ├── REPRODUCE.md                     # ← this file
    └── kernels/
        ├── compute/
        │   ├── overlap_compute.cpp          # pre-multiply path — PASSING ✓
        │   └── overlap_compute_geometry.cpp # geometry path — FAILING ✗ (bug here)
        └── dataflow/
            ├── read_tiles.cpp
            ├── read_tiles_geometry.cpp
            └── write_tile.cpp
```

**How to integrate:** copy `density_map_scatter/` into your tt-metal checkout at:

```
tt-metal/tt_metal/programming_examples/density_map_scatter/
```

Then add it to the parent `CMakeLists.txt` (or build it standalone — see Step 2 below).

---

## Step 1 — Set `TT_METAL_HOME` and check the device

```bash
# Set once; used by the Metal runtime and benchmark script.
export TT_METAL_HOME=/path/to/your/tt-metal

# Check no stale process holds the device:
tt-smi -s
# If any shown, kill them:
sudo pkill -9 -f "metal_example_density_map_scatter"
# Then verify device is free:
tt-smi -s
```

Symptoms of a locked device (not an accuracy bug — fix before running):
- `Device 0 init: failed to initialize FW! Try resetting the board.`
- `CHIP_IN_USE_0_PCIe` in Metal logs.

If locked after kill: `tt-smi -r` does a soft board reset.

---

## Step 2 — Copy sources and build

```bash
# Copy the example into tt-metal:
cp -r density_map_scatter/ $TT_METAL_HOME/tt_metal/programming_examples/

# Build (tt-metal must already be configured; adjust build dir as needed):
cd $TT_METAL_HOME
ninja -C build \
  metal_example_density_map_scatter \
  metal_example_density_map_scatter_benchmark \
  metal_example_density_map_scatter_benchmark_large \
  metal_example_density_map_scatter_benchmark_large_geometry
```

After **any edit** to `overlap_compute_geometry.cpp` or `density_map_scatter.cpp`, clear
the JIT kernel cache before running — otherwise the device runs stale compiled code:

```bash
rm -rf ~/.cache/tt-metal-cache/*
ninja -C $TT_METAL_HOME/build metal_example_density_map_scatter_benchmark_large_geometry
```

---

## Step 3 — Reproduce the passing case (pre-multiply) — sanity check

```bash
cd $TT_METAL_HOME
TT_METAL_HOME=$TT_METAL_HOME \
  ./build/programming_examples/metal_example_density_map_scatter_benchmark_large
```

Expected key output lines:

```
Per-task (vs exact DREAMPlace triangle formula): 119136 mismatches / 13029334 valid  max|err|=2.65e-03
  Relative L2     : 1.7208e-04  (‖TT−ref‖₂ / ‖ref‖₂)
  Total area      : TT=18016498  CPU=18018840  rel_err=1.3e-04
  → PASS
```

This is the **pre-multiply** path: host pre-computes `px`, `py`; device does `area = px * py`.
It always passes. Use it to confirm the device and build are healthy before testing geometry.

---

## Step 4 — Reproduce the failing case (geometry kernel)

```bash
cd $TT_METAL_HOME
TT_METAL_HOME=$TT_METAL_HOME \
  ./build/programming_examples/metal_example_density_map_scatter_benchmark_large_geometry
```

Expected key output lines (current state — **deterministic**):

```
  Task       0: expected=0.824359  tt=0.448502  err=3.76e-01
  Task       1: expected=1.352146  tt=1.413345  err=6.12e-02
Per-task mismatches: 8319909 / 13029334  max|err|=2.1458e+00

  Total area      : TT=17690672.0000  CPU=18018840.0000  rel_err=1.8212e-02
  Relative L2     : 1.7886e-01
FAIL: rel_total=1.8212e-02 (max 2.0e-03)  rel_L2=1.7886e-01 (max 5.0e-02)
```

The binary exits with `abort` (TT_THROW on FAIL). That is expected. The device ran
correctly; the answers are just wrong.

---

## Step 5 — Run the Python benchmark (optional, needs numpy + torch)

```bash
pip install numpy torch   # if not already installed

# Run from anywhere — the script finds binaries via $TT_METAL_HOME or --tt-binary flags:
TT_METAL_HOME=$TT_METAL_HOME python benchmark_sfpu_density_map.py

# Or point to binaries explicitly:
python benchmark_sfpu_density_map.py \
  --tt-binary          $TT_METAL_HOME/build/programming_examples/metal_example_density_map_scatter_benchmark_large \
  --tt-binary-geometry $TT_METAL_HOME/build/programming_examples/metal_example_density_map_scatter_benchmark_large_geometry \
  --skip-accuracy       # skip DREAMPlace C++ extension (not required)
```

The DREAMPlace C++ extension (CPU OMP path) is **optional**. If not installed, the script
skips steps 1–3 and still runs both TT binaries and prints the combined timing/accuracy table.

---

## Workload parameters (compiled in — not configurable at runtime)

| Parameter | Value |
|---|---|
| `NUM_CELLS` | 500,000 |
| `NUM_BINS_X = NUM_BINS_Y` | 2048 |
| `DOMAIN_WIDTH = DOMAIN_HEIGHT` | 3000.0 |
| `BIN_SIZE_X = BIN_SIZE_Y` | 3000 / 2048 = **1.46484375** (exact) |
| `MAX_IMPACT_X = MAX_IMPACT_Y` | 3 |
| `seed` | 2025 |
| Cores used | up to 120 Tensix (auto from mesh) |

To change these, edit the `#define` constants at the top of `density_map_scatter.cpp`,
clear the JIT cache, and rebuild both binaries.

---

## The bug: where to look

### File: `kernels/compute/overlap_compute_geometry.cpp`

The kernel computes three sequential SFPU passes inside a single `tile_regs_acquire()` region:

```
Pass 1 (run_px):   reads cxl(dst0), cxr(dst1), bxl(dst2)  → writes px to dst 3
Pass 2 (run_py):   reads cyl(dst0), cyr(dst1), byl(dst2)  → writes py to dst 4
Pass 3 (run_area): reads px(dst3),  py(dst4),  dummy(dst2) → writes area to dst 7
```

All three use `_llk_math_eltwise_ternary_sfpu_params_<true>()`. Face callbacks use
`dst_reg[d * N + i]` indexing (SFPI).

### The open question

Inside `_llk_math_eltwise_ternary_sfpu_params_`, between face calls, the LLK emits:

```cpp
TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D);   // advance dst write ptr by 8 lanes
```

Is `dst_reg[d * N + i]` in an SFPI lambda:

- **(a) Absolute** — `d=3` always means global dest tile 3, regardless of SETRWC?
- **(b) Relative to current SETRWC write pointer** — after N face increments, `d=3`
  maps to a *different* global tile?

If **(b)**: pass 3 (`run_area`) reads the wrong lanes for `px` and `py`, because the
write pointer was advanced by `run_px` and `run_py`'s face iterations. This precisely
explains the stable per-task error: `task 0: expected=0.824, tt=0.448`.

### What has been tried (all failed — see DOCUMENTATION.md for full record)

| Attempt | Change | Result |
|---|---|---|
| 1 | Replace ternary area pass with `mul_binary_tile` | Same error |
| 2 | Custom binary face using `_llk_math_eltwise_binary_sfpu_params_` | Same error |
| 3 | Move px/py to dst slots 5, 6 | Catastrophic (rel L2 > 800) — FP32 half-sync dest limit is 4 tiles |
| 4 | Compute py before px | Catastrophic (same dest limit issue) |
| 5 | Insert `tile_regs_commit()/wait()` between px and py | **Device hangs** |
| 6 | Non-aliasing dummy in ternary area pass (`in2=2, out=7`) | Same error |

### Working reference (confirms single-pass ternary is fine)

`kernels/compute/overlap_compute.cpp` does **one** ternary pass with dst 0, 1, 2 → 7.
It passes with rel L2 ~1.7e-4. This confirms:
- The ternary LLK infra works.
- The bug is specific to **chaining multiple passes** where pass N reads what pass N-1 wrote.

---

## Suggested expert experiments

**Experiment A — Verify absolute vs relative dst addressing**

Write a minimal kernel: pass 1 writes constant `42.0` to dst 3, pass 2 reads from dst 3
and writes to dst 7. Does `pack_tile(7)` give `42.0`? If yes → addressing is absolute.
If zero/garbage → relative, and all three passes need a dst-reset before reading intermediates.

**Experiment B — CB round-trip for intermediate values**

After `run_px`, pack dst 3 into a scratch CB (`pack_tile(3, scratch_cb)`), then re-unpack
it with `copy_tile` into dst 0 before `run_py`. This makes the addressing question moot and
should give correct results if px was computed correctly.

**Experiment C — Count SETRWC advances in the ternary params wrapper**

In `llk_math_eltwise_ternary_sfpu_params.h`, trace how many `TTI_SETRWC SET_D` instructions
are emitted per call to `_llk_math_eltwise_ternary_sfpu_params_` and whether
`_llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0)` resets the write base each call.

---

## Environment variables for debugging

```bash
# Show which kernel files are compiled and which JIT defines are passed:
export TT_METAL_RISCV_DEBUG_INFO=1

# Verbose device dispatch:
export TT_METAL_LOGGER_LEVEL=DEBUG

# Skip JIT cache entirely (always recompile from source):
export TT_METAL_KERNEL_JIT_DISABLE_CACHE=1
```

---

## Known non-issues — do not re-investigate

| Thing | Evidence it is NOT the bug |
|---|---|
| `BIN_SIZE` host vs JIT float mismatch | Instrumented: `DMS_BIN_SIZE_X_F = 1.46484375f` exact match |
| CPU reference sign convention | Reference uses `max(0,px)*max(0,py)` — gap persists |
| Host scatter / bin indices | Pre-multiply uses identical scatter and passes |
| `ratio ≠ 1` | Ratio removed; hardcoded 1.0 on device; not the cause |
| Stale JIT cache | Cache cleared before every test run |
| Device lock / FW init failure | Operational issue only; fix by killing zombie processes |

---

## Context

This is part of porting DREAMPlace density map computation to Tenstorrent hardware.

- **Pre-multiply path**: ~36× kernel speedup over CPU (3.84 ms vs 140 ms), **passes** accuracy.
- **Geometry path**: ~17× kernel speedup (8.26 ms), **fails** accuracy — the only blocker.

The SFPU architecture and LLK details needed to resolve this are internal to Tenstorrent.
The bug is fully reproducible and deterministic as described above.
