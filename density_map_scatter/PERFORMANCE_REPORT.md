# Performance report: density map scatter — geometry SFPU kernel (TT Trace / Tracy)

**Report path (this file):**  
`tt_metal/programming_examples/density_map_scatter/PERFORMANCE_REPORT.md`

**Date:** 2026-04-05

---

## 1. Where “TT Trace” lives in this codebase

| Piece | Location | Role |
|--------|----------|------|
| Tracy CLI + report driver | [`tools/tracy/__main__.py`](../../tools/tracy/__main__.py) | `python3 -m tracy …` sets `TT_METAL_DEVICE_PROFILER=1`, runs `capture-release`, then post-processes logs |
| Shared Tracy helpers | [`tools/tracy/common.py`](../../tools/tracy/common.py) | Default artifact roots: `generated/profiler`, log/report subdirs |
| Device op / trace merge | [`tools/tracy/process_ops_logs.py`](../../tools/tracy/process_ops_logs.py) | Builds consolidated CSVs under `generated/profiler/reports/` |
| Vendored Tracy fork | [`tt_metal/third_party/tracy/`](../../tt_metal/third_party/tracy/) | Tenstorrent Tracy fork (host + device plumbing) |
| Device zone API (kernels) | [`tt_metal/tools/profiler/kernel_profiler.hpp`](../../tt_metal/tools/profiler/kernel_profiler.hpp) | `DeviceZoneScopedN("name")` for on-device intervals |
| Written docs | [`docs/source/tt-metalium/tools/tracy_profiler.rst`](../../docs/source/tt-metalium/tools/tracy_profiler.rst), [`docs/source/tt-metalium/tools/device_program_profiler.rst`](../../docs/source/tt-metalium/tools/device_program_profiler.rst), [`tech_reports/MetalProfiler/metal-profiler.md`](../../tech_reports/MetalProfiler/metal-profiler.md) | Tracy GUI, device CSV format, build notes |

---

## 2. Geometry-dependent kernel (what you are timing)

Per [DOCUMENTATION.md](./DOCUMENTATION.md), the **geometry** path uses:

- **Compute:** [`kernels/compute/overlap_compute_geometry.cpp`](./kernels/compute/overlap_compute_geometry.cpp) — three SFPU ternary passes (`px`, `py`, `area`).
- **Host binary:** `metal_example_density_map_scatter_benchmark_large_geometry` (500k cells, 2048² bins, JIT `DMS_BIN_SIZE_*_F`).

**Host wall-clock “device kernel” time** (end-to-end workload on device, not launch-only) is already printed in `density_map_scatter.cpp` as the interval between `EnqueueMeshWorkload` and `Finish` on the command queue (see DOCUMENTATION § “Device kernel timing”).

---

## 3. Device profiler instrumentation (for *actual* on-core time in Tracy / CSV)

The geometry compute kernel now opens a **single device zone** for the full compute-kernel execution on each Tensix (one sample per core per program run — avoids millions of per-tile markers overflowing profiler buffers):

- **Zone name:** `DMS_OVERLAP_GEOMETRY_SFPU`
- **File:** [`kernels/compute/overlap_compute_geometry.cpp`](./kernels/compute/overlap_compute_geometry.cpp) (`#include "tools/profiler/kernel_profiler.hpp"` + `DeviceZoneScopedN` at the start of `kernel_main`).

With `TT_METAL_DEVICE_PROFILER=1` (set automatically by `python3 -m tracy -r …`), device timestamps are recorded and merged into the Tracy session and into `profile_log_device.csv` (cycles, core coords, RISC type, zone name). Interpretation: use chip frequency from the CSV header to convert cycles → time.

---

## 4. Command: `python3 -m tracy -v -r -p` and pytest

The workflow you were given:

```bash
python3 -m tracy -v -r -p -m pytest …
```

is aimed at **Python/ttnn tests** (`-m pytest` runs pytest as the profiled module). There is **no pytest** in-tree that runs the **density_map_scatter** C++ executable.

For **this** example, profile the **geometry** binary instead (from `TT_METAL_HOME`):

```bash
cd "$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME/tools:${PYTHONPATH}"
python3 -m tracy -v -r -p \
  "$TT_METAL_HOME/build/programming_examples/metal_example_density_map_scatter_benchmark_large_geometry"
```

Notes:

- `-r` — generate ops / device merged reports after capture.
- `-p` — partial (zone-based) profiling on the Python side when profiling Python; for a native binary the outer driver still enables device profiling via `TT_METAL_DEVICE_PROFILER=1` (see [`tools/tracy/__main__.py`](../../tools/tracy/__main__.py)).
- Optional: `-o /path/to/output` sets `TT_METAL_PROFILER_DIR` so all artifacts go to a chosen folder.

If the program **exits non-zero**, Tracy may still write a partial `.tracy` file; post-processing can miss `profile_log_device.csv` because the device was never closed cleanly.

---

## 5. Artifacts to open after a successful run

Under `${TT_METAL_HOME}/generated/profiler/` (or your `-o` directory):

| Artifact | Path (typical) |
|----------|----------------|
| Tracy session (host + embedded device markers) | `.logs/tracy_profile_log_host.tracy` |
| Raw device profiler CSV | `.logs/profile_log_device.csv` |
| Host-side op export | `.logs/tracy_ops_times.csv`, `.logs/tracy_ops_data.csv` |
| C++ device perf rollup (when enabled) | `.logs/cpp_device_perf_report.csv` |
| Dated merged report folder | `reports/ops_perf_results_*.csv` (exact name includes timestamp; optional `name_append` from `-n`) |

Open `.tracy` in the [Tracy GUI](https://github.com/wolfpld/tracy). Filter zones for `DMS_OVERLAP_GEOMETRY_SFPU` to see the geometry compute kernel on device.

---

## 6. Run on 2026-04-05 (this workspace)

**Result:** Profiling **did not** complete successfully here: both the large geometry binary and the small default binary aborted during device bring-up with:

`Timeout waiting for Ethernet core service remote IO request.`

So **no fresh device CSV or ops_perf row** was produced for this session. The commands and paths above are valid for re-running on a healthy Wormhole system.

**Workaround for timing without Tracy:** when the binary runs successfully, use the printed line:

`Device : kernel (N Tensix cores) X.XXX ms`

That number is the same host-measured window documented in [DOCUMENTATION.md](./DOCUMENTATION.md) (mesh workload + `Finish`).

---

## 7. Summary

- **TT Trace** in this repo is the **Tracy-based** profiler under **`tools/tracy`**, backed by **`tt_metal/third_party/tracy`**, documented in **MetalProfiler** / **device_program_profiler** docs.
- **Geometry kernel** on-device intervals are attributed in device logs / Tracy under zone **`DMS_OVERLAP_GEOMETRY_SFPU`** (instrumentation added in `overlap_compute_geometry.cpp`).
- **Profile the C++ geometry benchmark** with `python3 -m tracy -v -r -p <path/to/metal_example_density_map_scatter_benchmark_large_geometry>`, not `-m pytest`, unless you add a pytest wrapper yourself.
- **This report file** is the single place to return to for paths and commands; successful runs will add concrete numbers under **`generated/profiler/`**.
