#!/usr/bin/env python3
"""
Benchmark: TT Metal SFPU density-map scatter kernel vs DREAMPlace CPU exact path.

Python configures the **CPU + numpy** workload (cells, grid, domain) below.

The **TT subprocess** runs `metal_example_density_map_scatter_benchmark_large`
(500_000 cells, 2048×2048 bins, 3000×3000 domain — matches Python below).  For the
smaller 200K / 512² case use `metal_example_density_map_scatter_benchmark` and set
Python `NUM_CELLS`, grid, and domain to match that binary (see `TT_BINARY_*` in code).

Synthetic inputs (when CPU and TT match):
  - Positions uniform in [xl, xh−12), sizes uniform in [1, 11)
  - node_size_x_clamped = max(node_size_x, bin_size_x)   ← DREAMPlace clamping
  - offset_x = (sx_clamped − sx) / 2                     ← DREAMPlace centering

CPU path  : ElectricDensityMapFunction.forward (electric_density_map.cpp C++ extension)
            = exact computeTriangleDensityMapLauncher with OMP + atomic scatter.
            Timed from "tensors in RAM" to "density map in RAM".

TT path   : Two C++ binaries (same synthetic workload when sizes match):
            • *_benchmark_large — host precomputes px/py; device does area = px·ratio·py (3 DRAM inputs).
            • *_benchmark_large_geometry — host uploads cxl,cxr,bxl,cyl,cyr,byl (ratio=1.0 hardcoded on device);
              JIT defines for BIN_SIZE and full overlap SFPU (6 DRAM inputs).
            Timed from "data in DRAM" to "results in DRAM" (Device: kernel line).
            Upload / readback reported separately. Host generate+enumerate is printed by each binary.

Accuracy  : numpy re-implementation of the exact DREAMPlace triangle_density_function
            run on the same synthetic inputs, compared to the CPU reference.
            This characterises formula fidelity.  The SFPU adds a small additional
            FP32 rounding error (~3% extra rel-L2) characterised by the C++ benchmark.

500K × 2048² (matched CPU + TT):
  cd tt-metal && ninja -C build metal_example_density_map_scatter_benchmark_large \\
    metal_example_density_map_scatter_benchmark_large_geometry
  source DREAMPlace/venv/bin/activate && cd DREAMPlace && python scripts/benchmark_sfpu_density_map.py

200K × 512²: build `metal_example_density_map_scatter_benchmark`, point --tt-binary there,
  and set Python NUM_CELLS=200_000, NUM_BINS_*=512, XH=YH=1000.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

# ── Repository path ───────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))

# Locate TT Metal build directory.
# Search order:
#   1. $TT_METAL_HOME/build/programming_examples/   (set by user or tt-metal env)
#   2. ./build/programming_examples/                (script run from inside tt-metal/)
#   3. ../tt-metal/build/programming_examples/      (script in sibling dir of tt-metal)
#   4. ../../tt-metal/build/programming_examples/   (script two levels above tt-metal)
# The expert can always override with --tt-binary / --tt-binary-geometry.
def _find_tt_binary(name: str) -> str:
    candidates = []
    tt_home = os.environ.get("TT_METAL_HOME", "")
    if tt_home:
        candidates.append(os.path.join(tt_home, "build", "programming_examples", name))
    candidates += [
        os.path.join(SCRIPT_DIR, "build", "programming_examples", name),
        os.path.join(SCRIPT_DIR, "..", "build", "programming_examples", name),
        os.path.join(SCRIPT_DIR, "..", "tt-metal", "build", "programming_examples", name),
        os.path.join(SCRIPT_DIR, "..", "..", "tt-metal", "build", "programming_examples", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # Return the most informative path for the error message even if not found.
    return os.path.abspath(candidates[1])

TT_BINARY_DEFAULT          = _find_tt_binary("metal_example_density_map_scatter_benchmark_large")
TT_BINARY_GEOMETRY_DEFAULT = _find_tt_binary("metal_example_density_map_scatter_benchmark_large_geometry")

# ── DREAMPlace CPU extension path setup ──────────────────────────────────────
# Search order for the compiled electric_potential_cpp extension:
#   1. Transport/dreamplace_ref/  (self-contained copy shipped with this folder)
#   2. $DREAMPLACE_ROOT env var   (user override)
#   3. ../DREAMPlace/dreamplace/  (sibling DREAMPlace installation)
#   4. ../../dreamplace/          (two levels up)
# If none is found the CPU path is gracefully skipped — only the numpy baseline
# and TT binary paths will run.
def _setup_dreamplace_path() -> bool:
    """Add the first valid dreamplace package root to sys.path. Returns True if found."""
    candidates = [
        os.path.join(SCRIPT_DIR, "dreamplace_ref"),                          # self-contained copy
        os.environ.get("DREAMPLACE_ROOT", ""),                                # user override
        os.path.join(SCRIPT_DIR, "..", "DREAMPlace", "dreamplace"),          # sibling
        os.path.join(SCRIPT_DIR, "..", "..", "dreamplace"),                   # two levels up
        os.path.join(SCRIPT_DIR, "..", "dreamplace_ref"),                     # adjacent repo
    ]
    for c in candidates:
        if not c:
            continue
        probe = os.path.join(c, "dreamplace", "ops", "electric_potential",
                             "__init__.py") if not c.endswith("dreamplace") else \
                os.path.join(c, "ops", "electric_potential", "__init__.py")
        # Also accept a root that contains a dreamplace/ sub-package
        if os.path.isfile(probe):
            root = c if not c.endswith("dreamplace") else os.path.dirname(c)
            if root not in sys.path:
                sys.path.insert(0, root)
            return True
    return False

_dreamplace_found = _setup_dreamplace_path()

# ── Problem parameters (Python: CPU + numpy path) ─────────────────────────────
# Default matches metal_example_density_map_scatter_benchmark_large (CMake).
NUM_CELLS  = 500_000
NUM_BINS_X = 2048
NUM_BINS_Y = 2048
XL, YL     = 0.0, 0.0
XH, YH     = 3000.0, 3000.0
BIN_SIZE_X = (XH - XL) / NUM_BINS_X
BIN_SIZE_Y = (YH - YL) / NUM_BINS_Y
SEED       = 2025
# DREAMPlace: num_movable_impacted_bins = ceil(max_cell_size / bin_size) + 2
MAX_CELL   = 11.0
N_IMPACT_X = int(math.ceil(MAX_CELL / BIN_SIZE_X)) + 2
N_IMPACT_Y = int(math.ceil(MAX_CELL / BIN_SIZE_Y)) + 2

# ── TT binary compile-time workload (must match chosen --tt-binary executable)
TT_BINARY_NUM_CELLS  = 500_000
TT_BINARY_NUM_BINS_X = 2048
TT_BINARY_NUM_BINS_Y = 2048
TT_BINARY_XL, TT_BINARY_YL = 0.0, 0.0
TT_BINARY_XH, TT_BINARY_YH = 3000.0, 3000.0


def python_tt_workloads_match():
    return (
        NUM_CELLS == TT_BINARY_NUM_CELLS
        and NUM_BINS_X == TT_BINARY_NUM_BINS_X
        and NUM_BINS_Y == TT_BINARY_NUM_BINS_Y
        and XL == TT_BINARY_XL
        and YL == TT_BINARY_YL
        and XH == TT_BINARY_XH
        and YH == TT_BINARY_YH
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate synthetic inputs  (CPU path; TT binary uses its own if sizes differ)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_inputs():
    """
    Returns float32 torch tensors for ElectricDensityMapFunction.forward.

    When `python_tt_workloads_match()` is true, this matches the C++ benchmark's
    synthetic generator (same seed and distributions).
    """
    rng = torch.Generator()
    rng.manual_seed(SEED)

    pos_x  = torch.rand(NUM_CELLS, generator=rng) * (XH - XL - 12.0) + XL
    pos_y  = torch.rand(NUM_CELLS, generator=rng) * (YH - YL - 12.0) + YL
    sx_raw = torch.rand(NUM_CELLS, generator=rng) * (MAX_CELL - 1.0) + 1.0   # [1, 11)
    sy_raw = torch.rand(NUM_CELLS, generator=rng) * (MAX_CELL - 1.0) + 1.0

    # DREAMPlace: node_size_x_clamped = max(node_size_x, bin_size_x)
    sx_clamped = torch.clamp(sx_raw, min=float(BIN_SIZE_X))
    sy_clamped = torch.clamp(sy_raw, min=float(BIN_SIZE_Y))

    # DREAMPlace: offset_x = (sx_clamped − sx_raw) / 2  — centres the stretched cell
    offset_x = (sx_clamped - sx_raw) * 0.5
    offset_y = (sy_clamped - sy_raw) * 0.5

    ratio = torch.ones(NUM_CELLS)

    # Flat position tensor expected by DREAMPlace: [x0..xN, y0..yN]
    pos = torch.cat([pos_x, pos_y])

    # Bin centres
    bin_center_x = (torch.arange(NUM_BINS_X, dtype=torch.float32) + 0.5) * BIN_SIZE_X + XL
    bin_center_y = (torch.arange(NUM_BINS_Y, dtype=torch.float32) + 0.5) * BIN_SIZE_Y + YL

    n_stretched_x = int((sx_raw < BIN_SIZE_X).sum().item())
    n_stretched_y = int((sy_raw < BIN_SIZE_Y).sum().item())

    return {
        "pos":           pos,
        "pos_x":         pos_x,
        "pos_y":         pos_y,
        "sx_raw":        sx_raw,
        "sy_raw":        sy_raw,
        "sx_clamped":    sx_clamped,
        "sy_clamped":    sy_clamped,
        "offset_x":      offset_x,
        "offset_y":      offset_y,
        "ratio":         ratio,
        "bin_center_x":  bin_center_x,
        "bin_center_y":  bin_center_y,
        "n_stretched_x": n_stretched_x,
        "n_stretched_y": n_stretched_y,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CPU path: exact DREAMPlace computeTriangleDensityMapLauncher
# ─────────────────────────────────────────────────────────────────────────────

def run_cpu_scatter(inp: dict, num_runs: int, warmup: int):
    """
    Call ElectricDensityMapFunction.forward with our synthetic inputs.
    This invokes electric_density_map.cpp's computeTriangleDensityMapLauncher
    via the compiled C++ extension (OMP threading already built in).

    Returns (mean_ms, std_ms, density_map_tensor [NUM_BINS_X, NUM_BINS_Y]).
    """
    from dreamplace.ops.electric_potential.electric_overflow import ElectricDensityMapFunction

    pos          = inp["pos"]
    zeros_init   = torch.zeros(NUM_BINS_X, NUM_BINS_Y, dtype=pos.dtype)
    dummy_pmask  = torch.zeros(NUM_BINS_X, NUM_BINS_Y, dtype=pos.dtype)  # unused (padding=0)
    dummy_snmap  = torch.arange(NUM_CELLS, dtype=torch.int32)            # unused (det_flag=0)
    bin_area     = BIN_SIZE_X * BIN_SIZE_Y

    # Positional args match electric_overflow.py forward() signature exactly.
    fwd_args = (
        inp["sx_clamped"],          # node_size_x_clamped
        inp["sy_clamped"],          # node_size_y_clamped
        inp["offset_x"],            # offset_x  = (sx_clamped − sx) / 2
        inp["offset_y"],
        inp["ratio"],
        inp["bin_center_x"],
        inp["bin_center_y"],
        zeros_init,                  # initial_density_map (scatter-only; no fixed cells)
        0.8,                         # target_density (only used for padding bins, padding=0)
        float(XL), float(YL), float(XH), float(YH),
        float(BIN_SIZE_X), float(BIN_SIZE_Y),
        NUM_CELLS,                   # num_movable_nodes
        0,                           # num_filler_nodes
        0,                           # padding
        dummy_pmask,                 # padding_mask (unused, padding=0)
        NUM_BINS_X, NUM_BINS_Y,
        N_IMPACT_X, N_IMPACT_Y,      # num_movable_impacted_bins_x/y
        2, 2,                        # num_filler_impacted_bins (unused, num_filler=0)
        0,                           # deterministic_flag=0 → OMP parallel, no sorted_node_map
        dummy_snmap,                 # sorted_node_map (unused when det_flag=0)
    )

    for _ in range(warmup):
        dm = ElectricDensityMapFunction.forward(pos, *fwd_args)
        dm.mul_(1.0 / bin_area)

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        dm = ElectricDensityMapFunction.forward(pos, *fwd_args)
        dm.mul_(1.0 / bin_area)
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times)), dm.float().clone()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Accuracy: numpy exact DREAMPlace formula on same inputs → compare to CPU
# ─────────────────────────────────────────────────────────────────────────────

def compute_formula_accuracy(inp: dict, cpu_dm: torch.Tensor):
    """
    Re-implement triangle_density_function in numpy (vectorized) on the SAME
    inputs fed to the CPU path.  Runs in a few seconds via batched array ops.

    The SFPU kernel computes this same formula; any additional deviation is
    SFPU FP32 arithmetic rounding (~3% extra rel-L2, from the C++ benchmark).
    """
    nx   = (inp["pos_x"] + inp["offset_x"]).numpy().astype(np.float64)
    ny   = (inp["pos_y"] + inp["offset_y"]).numpy().astype(np.float64)
    nxr  = nx + inp["sx_clamped"].numpy().astype(np.float64)
    nyr  = ny + inp["sy_clamped"].numpy().astype(np.float64)
    rat  = inp["ratio"].numpy().astype(np.float64)

    inv_bx = 1.0 / BIN_SIZE_X
    inv_by = 1.0 / BIN_SIZE_Y

    # Integer bin range for each cell [bxl_i, bxh_i)
    bxl_arr = np.clip((nx  - XL) * inv_bx,       0, NUM_BINS_X - 1).astype(np.int32)
    bxh_arr = np.clip((nxr - XL) * inv_bx + 1.0, 1, NUM_BINS_X    ).astype(np.int32)
    byl_arr = np.clip((ny  - YL) * inv_by,       0, NUM_BINS_Y - 1).astype(np.int32)
    byh_arr = np.clip((nyr - YL) * inv_by + 1.0, 1, NUM_BINS_Y    ).astype(np.int32)

    # Offsets within impact window: 0..N_IMPACT_X-1 / 0..N_IMPACT_Y-1
    dx = np.arange(N_IMPACT_X, dtype=np.int32)
    dy = np.arange(N_IMPACT_Y, dtype=np.int32)

    # Broadcast to (NUM_CELLS, N_IMPACT_X, N_IMPACT_Y) — process in chunks
    # to avoid ~1 GB peak allocation for 200K cells at once.
    CHUNK = 20_000
    py_density = np.zeros((NUM_BINS_X, NUM_BINS_Y), dtype=np.float64)
    flat_density = py_density.ravel()

    for c0 in range(0, NUM_CELLS, CHUNK):
        c1 = min(c0 + CHUNK, NUM_CELLS)
        n  = c1 - c0
        # Bin index grids, explicitly shaped (n, N_IMPACT_X, N_IMPACT_Y)
        bx_base = bxl_arr[c0:c1, None] + dx[None, :]          # (n, Nx)
        by_base = byl_arr[c0:c1, None] + dy[None, :]           # (n, Ny)
        bx = np.broadcast_to(bx_base[:, :, None], (n, N_IMPACT_X, N_IMPACT_Y)).copy()
        by = np.broadcast_to(by_base[:, None, :], (n, N_IMPACT_X, N_IMPACT_Y)).copy()

        # Validity mask — only bins inside each cell's actual impacted range
        valid = (
            (bx >= 0) & (bx < bxh_arr[c0:c1, None, None]) & (bx < NUM_BINS_X) &
            (by >= 0) & (by < byh_arr[c0:c1, None, None]) & (by < NUM_BINS_Y)
        )

        # Triangle-density overlap in x and y
        bxl_f = XL + bx.astype(np.float64) * BIN_SIZE_X
        byl_f = YL + by.astype(np.float64) * BIN_SIZE_Y
        px = (np.minimum(nxr[c0:c1, None, None], bxl_f + BIN_SIZE_X)
              - np.maximum(nx[c0:c1, None, None],  bxl_f))
        py = (np.minimum(nyr[c0:c1, None, None], byl_f + BIN_SIZE_Y)
              - np.maximum(ny[c0:c1, None, None],  byl_f))

        area = px * rat[c0:c1, None, None] * py
        area[~valid] = 0.0

        # Flatten and scatter-add into density grid
        flat_v   = valid.ravel()
        flat_idx = bx.ravel()[flat_v] * NUM_BINS_Y + by.ravel()[flat_v]
        np.add.at(flat_density, flat_idx, area.ravel()[flat_v])

    py_density /= (BIN_SIZE_X * BIN_SIZE_Y)

    cpu_np   = cpu_dm.numpy().astype(np.float64)
    diff     = py_density - cpu_np
    sum_sq_d = float(np.sum(diff ** 2))
    sum_sq_r = float(np.sum(cpu_np ** 2))
    rel_l2   = math.sqrt(sum_sq_d / sum_sq_r) if sum_sq_r > 0 else 0.0
    rel_tot  = abs(py_density.sum() - cpu_np.sum()) / max(cpu_np.sum(), 1e-9)
    max_abs  = float(np.max(np.abs(diff)))

    return {
        "rel_l2":    rel_l2,
        "rel_total": rel_tot,
        "max_abs":   max_abs,
        "nonzero":   int(np.count_nonzero(py_density)),
        "total_ref": float(cpu_np.sum()),
        "total_py":  float(py_density.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TT path: metal_example_density_map_scatter_benchmark via subprocess
# ─────────────────────────────────────────────────────────────────────────────
# The binary generates the SAME synthetic workload (200K cells, 512×512, seed 2025)
# and prints per-phase timing.  We parse and report:
#   - Upload time (host → DRAM)
#   - Device kernel time (pure SFPU execution on 56 cores)
#   - Readback time (DRAM → host)
# This lets the user see end-to-end TT latency vs CPU scatter latency.

# Matches "3× inputs ..." or "6× geometry inputs ..." etc.
_RE_UPLOAD  = re.compile(r"Host\s*:\s*DRAM upload.*?([\d.]+)\s*ms")
_RE_UPLOAD2 = re.compile(r"^\s*DRAM upload\s*:\s*([\d.]+)\s*ms", re.MULTILINE)
_RE_DEVICE  = re.compile(r"Device\s*:\s*kernel \((\d+) Tensix cores\)\s*([\d.]+)\s*ms")
# Fallback if only the indented summary line appears (same binary, different log interleaving).
_RE_DEVICE_KERN_MS = re.compile(r"^\s*Device\s+kernel\s*:\s*([\d.]+)\s*ms", re.MULTILINE)
_RE_NCORES = re.compile(r"using\s+(\d+)\s+Tensix core")
_RE_READBK  = re.compile(r"Host\s*:\s*DRAM readback\s*([\d.]+)\s*ms")
_RE_READBK2 = re.compile(r"^\s*DRAM readback\s*:\s*([\d.]+)\s*ms", re.MULTILINE)
_RE_PASS    = re.compile(r"Test Passed|Test Failed")
_RE_REL_L2  = re.compile(r"Relative L2\s*:\s*([\d.e+-]+)")
_RE_REL_TOT = re.compile(r"Total area\s*.*?rel_err=([\d.e+-]+)")


def run_tt_binary(tt_binary: str, num_runs: int, warmup: int, label: str = "TT"):
    """
    Run the C++ benchmark binary (warmup + timed runs) and parse timing lines.
    Returns (upload_ms list, kernel_ms list, readback_ms list, n_cores, rel_l2, rel_total).
    """
    # binary lives at <tt-metal>/build/programming_examples/<name>
    # → go up two levels to reach <tt-metal>
    tt_metal_home = os.path.abspath(
        os.path.join(os.path.dirname(tt_binary), "..", "..")
    )
    env = {**os.environ, "TT_METAL_HOME": tt_metal_home}

    def _one_run():
        r = subprocess.run([tt_binary], capture_output=True, text=True,
                           env=env, cwd=tt_metal_home)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            # UMD logs often contain the word "Device"; require the benchmark's own line.
            if not _RE_DEVICE.search(out) and not _RE_DEVICE_KERN_MS.search(out):
                err_lines = [
                    l for l in out.splitlines()
                    if "error" in l.lower() or "exception" in l.lower() or "fatal" in l.lower()
                    or "timeout" in l.lower()
                ]
                raise RuntimeError(
                    "TT binary failed (rc=%d): %s" % (
                        r.returncode, err_lines[0] if err_lines else out[:500]))
        m_up   = _RE_UPLOAD.search(out) or _RE_UPLOAD2.search(out)
        m_dev  = _RE_DEVICE.search(out)
        if not m_dev:
            m_k = _RE_DEVICE_KERN_MS.search(out)
            m_nc = _RE_NCORES.search(out)
            if m_k and m_nc:
                # Synthesize a match-like object for kernel_ms / n_cores
                class _M:
                    def __init__(self, n, ms):
                        self._n, self._ms = n, ms
                    def group(self, i):
                        return self._n if i == 1 else self._ms
                m_dev = _M(int(m_nc.group(1)), m_k.group(1))
        m_rb   = _RE_READBK.search(out) or _RE_READBK2.search(out)
        m_rl2  = _RE_REL_L2.search(out)
        m_rtot = _RE_REL_TOT.search(out)
        if not m_dev:
            hint = ""
            if "Could not parse" not in out and r.returncode != 0:
                hint = (
                    "\n(Hint: process exited before printing kernel timing — often device init "
                    "failure or Ethernet timeout; try `tt-smi -r 0` and ensure no other "
                    "process holds the card. Set TT_METAL_LOGGER_LEVEL=Error to reduce log noise.)"
                )
            raise RuntimeError(
                "Could not parse device-timing line (need "
                "'Device : kernel (N Tensix cores) X ms' or 'Device kernel  : X ms')."
                + hint + "\nOutput:\n" + out[:3000])
        return {
            "upload_ms":   float(m_up.group(1))  if m_up  else 0.0,
            "kernel_ms":   float(m_dev.group(2)),
            "readback_ms": float(m_rb.group(1))  if m_rb  else 0.0,
            "n_cores":     int(m_dev.group(1)),
            "rel_l2":      float(m_rl2.group(1)) if m_rl2 else float("nan"),
            "rel_total":   float(m_rtot.group(1)) if m_rtot else float("nan"),
        }

    print("      [%s] warming up (%d run%s)..." % (label, warmup, "s" if warmup != 1 else ""))
    for _ in range(warmup):
        _one_run()

    uploads, kernels, readbacks = [], [], []
    n_cores = 0
    rel_l2 = rel_total = float("nan")
    for i in range(num_runs):
        t = _one_run()
        uploads.append(t["upload_ms"])
        kernels.append(t["kernel_ms"])
        readbacks.append(t["readback_ms"])
        n_cores  = t["n_cores"]
        rel_l2   = t["rel_l2"]
        rel_total = t["rel_total"]
        print("      [%s] run %d/%d: upload=%.1f ms  kernel=%.1f ms  readback=%.1f ms" % (
            label, i + 1, num_runs, t["upload_ms"], t["kernel_ms"], t["readback_ms"]))

    return uploads, kernels, readbacks, n_cores, rel_l2, rel_total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark TT SFPU density-map kernel vs DREAMPlace CPU on identical synthetic inputs"
    )
    ap.add_argument("--runs",      type=int, default=5,  help="Timed runs for each path")
    ap.add_argument("--warmup",    type=int, default=2,  help="Warm-up runs before timing")
    ap.add_argument(
        "--tt-binary",
        default=os.path.abspath(TT_BINARY_DEFAULT),
        help="Path to pre-multiply TT binary (host px/py + SFPU multiply, 500K×2048²)",
    )
    ap.add_argument(
        "--tt-binary-geometry",
        default=os.path.abspath(TT_BINARY_GEOMETRY_DEFAULT),
        help="Path to geometry TT binary (6 DRAM inputs; ratio=1.0 in kernel; BIN_SIZE via JIT defines)",
    )
    ap.add_argument(
        "--skip-geometry-tt",
        action="store_true",
        help="Do not run the geometry-dependent TT benchmark binary",
    )
    ap.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip the (slow) per-cell numpy accuracy loop; use TT binary's reported metrics",
    )
    args = ap.parse_args()

    print("=" * 72)
    print("  TT Metal SFPU density-map kernel vs DREAMPlace CPU  (synthetic)")
    print("=" * 72)
    print("  Cells   : %d  |  Grid: %dx%d  |  Domain: %.0f×%.0f" % (
        NUM_CELLS, NUM_BINS_X, NUM_BINS_Y, XH - XL, YH - YL))
    print("  bin_size: %.6f×%.6f  |  impact: %d×%d" % (
        BIN_SIZE_X, BIN_SIZE_Y, N_IMPACT_X, N_IMPACT_Y))
    if _dreamplace_found:
        print("  DREAMPlace CPU ext: found (%s)" % (
            next((p for p in sys.path if "dreamplace_ref" in p or "DREAMPlace" in p), "on sys.path")))
    else:
        print("  DREAMPlace CPU ext: NOT FOUND — CPU C++ path will be skipped")
        print("    (place dreamplace_ref/ next to this script, or set DREAMPLACE_ROOT)")
    if not python_tt_workloads_match():
        print()
        print("  *** WARNING: Python workload ≠ compiled TT binary workload ***")
        print("      CPU+numpy: %d cells, %d×%d bins, domain [%.0f,%.0f]×[%.0f,%.0f]" % (
            NUM_CELLS, NUM_BINS_X, NUM_BINS_Y, XL, XH, YL, YH))
        print("      TT binary: %d cells, %d×%d bins, domain [%.0f,%.0f]×[%.0f,%.0f]" % (
            TT_BINARY_NUM_CELLS, TT_BINARY_NUM_BINS_X, TT_BINARY_NUM_BINS_Y,
            TT_BINARY_XL, TT_BINARY_XH, TT_BINARY_YL, TT_BINARY_YH))
        print("      Speedup lines compare different problems. SFPU rel-L2 is for the TT job only.")
        print("      Rebuild tt-metal with matching constants (or a second benchmark target).")
    print()

    # ── 1. Generate synthetic inputs for CPU path ───────────────────────────────
    print("[1/5] Generating synthetic inputs (seed=%d)..." % SEED)
    inp = make_synthetic_inputs()
    print("      pos_x ∈ [%.1f, %.1f)  pos_y ∈ [%.1f, %.1f)" % (
        float(inp["pos_x"].min()), float(inp["pos_x"].max()),
        float(inp["pos_y"].min()), float(inp["pos_y"].max())))
    print("      sx_raw ∈ [%.3f, %.3f)   sx_clamped ∈ [%.3f, %.3f)" % (
        float(inp["sx_raw"].min()), float(inp["sx_raw"].max()),
        float(inp["sx_clamped"].min()), float(inp["sx_clamped"].max())))
    print("      %d cells had sx < bin_size_x (stretched in x)" % inp["n_stretched_x"])
    print("      %d cells had sy < bin_size_y (stretched in y)" % inp["n_stretched_y"])
    print()

    # ── 2. CPU path: exact DREAMPlace OMP scatter ─────────────────────────────
    print("[2/5] CPU path  — ElectricDensityMapFunction.forward (OMP, C++ extension)")
    print("      %d warmup + %d timed runs..." % (args.warmup, args.runs))
    try:
        cpu_mean, cpu_std, cpu_dm = run_cpu_scatter(inp, args.runs, args.warmup)
        print("      Mean: %.3f ms   Std: %.3f ms" % (cpu_mean, cpu_std))
        cpu_available = True
    except Exception as e:
        print("      FAILED: %s" % e)
        print("      (activate DREAMPlace venv: source venv/bin/activate)")
        cpu_mean = cpu_std = None
        cpu_dm = None
        cpu_available = False
    print()

    # ── 3. Accuracy: numpy exact formula on same inputs vs CPU reference ───────
    if cpu_available and not args.skip_accuracy:
        print("[3/5] Accuracy — exact triangle_density_function (numpy vectorised) on same inputs")
        t0 = time.perf_counter()
        acc = compute_formula_accuracy(inp, cpu_dm)
        elapsed = (time.perf_counter() - t0) * 1000.0
        print("      Computed in %.1f ms (chunked numpy scatter)" % elapsed)
        print("      non-zero bins  : %d / %d" % (acc["nonzero"], NUM_BINS_X * NUM_BINS_Y))
        print("      total area     : numpy=%.4f  CPU=%.4f  rel_err=%.4e" % (
            acc["total_py"], acc["total_ref"], acc["rel_total"]))
        print("      max |bin delta|: %.4e" % acc["max_abs"])
        print("      Relative L2    : %.4e  (||formula − CPU||₂ / ||CPU||₂)" % acc["rel_l2"])
    elif args.skip_accuracy:
        print("[3/5] Accuracy — skipped (--skip-accuracy); will use TT binary's reported metrics")
        acc = None
    else:
        print("[3/5] Accuracy — skipped (CPU path not available)")
        acc = None
    print()

    # ── 4. TT SFPU: pre-multiply binary (host px/py + device multiply) ───────
    print("[4/5] TT SFPU — pre-multiply variant (3 DRAM inputs)")
    if python_tt_workloads_match():
        print("      Same workload as [1–3]: %d cells, %dx%d grid, seed=%d" % (
            NUM_CELLS, NUM_BINS_X, NUM_BINS_Y, SEED))
    else:
        print("      Compiled workload: %d cells, %dx%d grid, seed=%d (see WARNING above)" % (
            TT_BINARY_NUM_CELLS, TT_BINARY_NUM_BINS_X, TT_BINARY_NUM_BINS_Y, SEED))
    tt_binary = os.path.abspath(args.tt_binary)
    if not os.path.exists(tt_binary):
        print("      Binary not found: %s" % tt_binary)
        print("      Build: ninja -C tt-metal/build metal_example_density_map_scatter_benchmark_large")
        tt_available = False
        uploads = kernels = readbacks = []
        n_cores = 0
        tt_rel_l2 = tt_rel_total = float("nan")
    else:
        try:
            uploads, kernels, readbacks, n_cores, tt_rel_l2, tt_rel_total = \
                run_tt_binary(tt_binary, args.runs, args.warmup, label="pre-mult")
            tt_available = True
        except Exception as e:
            print("      FAILED: %s" % e)
            tt_available = False
            uploads = kernels = readbacks = []
            n_cores = 0
            tt_rel_l2 = tt_rel_total = float("nan")
    print()

    # ── 5. TT SFPU: geometry-dependent binary (6 inputs, BIN_SIZE on device) ─
    print("[5/5] TT SFPU — geometry-dependent variant (6 DRAM inputs, JIT defines)")
    tt_geom_binary = os.path.abspath(args.tt_binary_geometry)
    if args.skip_geometry_tt:
        print("      Skipped (--skip-geometry-tt)")
        tt_geom_available = False
        gu, gk, gr = [], [], []
        n_cores_g = 0
        tt_g_rel_l2 = tt_g_rel_total = float("nan")
    elif not os.path.exists(tt_geom_binary):
        print("      Binary not found: %s" % tt_geom_binary)
        print("      Build: ninja -C tt-metal/build "
              "metal_example_density_map_scatter_benchmark_large_geometry")
        tt_geom_available = False
        gu, gk, gr = [], [], []
        n_cores_g = 0
        tt_g_rel_l2 = tt_g_rel_total = float("nan")
    else:
        try:
            gu, gk, gr, n_cores_g, tt_g_rel_l2, tt_g_rel_total = run_tt_binary(
                tt_geom_binary, args.runs, args.warmup, label="geometry")
            tt_geom_available = True
        except Exception as e:
            print("      FAILED: %s" % e)
            tt_geom_available = False
            gu, gk, gr = [], [], []
            n_cores_g = 0
            tt_g_rel_l2 = tt_g_rel_total = float("nan")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    n_cores_disp = n_cores if tt_available else (n_cores_g if tt_geom_available else 0)
    print("=" * 72)
    print("  RESULTS: %d cells, %d×%d grid, %d Tensix cores" % (
        NUM_CELLS, NUM_BINS_X, NUM_BINS_Y, n_cores_disp))
    print("=" * 72)
    print()
    print("  ── Runtime ──────────────────────────────────────────────────────")
    if cpu_available:
        print("  CPU DREAMPlace OMP scatter       : %8.3f ms  ±%.3f ms" % (cpu_mean, cpu_std))
    def _print_tt_variant(name, up_l, k_l, rb_l, nc):
        if not up_l:
            return
        up_m = float(np.mean(up_l))
        k_m = float(np.mean(k_l))
        rb_m = float(np.mean(rb_l))
        k_std = float(np.std(k_l))
        e2e_m = up_m + k_m + rb_m
        print("  %-34s upload %8.3f ms" % (name, up_m))
        print("  %-34s kernel %8.3f ms  ±%.3f ms  (%2d cores)" % ("", k_m, k_std, nc))
        print("  %-34s readback %6.3f ms" % ("", rb_m))
        print("  %-34s e2e      %8.3f ms" % ("", e2e_m))

    if tt_available:
        _print_tt_variant("TT pre-multiply", uploads, kernels, readbacks, n_cores)
    if tt_geom_available:
        _print_tt_variant("TT geometry-dependent", gu, gk, gr, n_cores_g)

    if cpu_available and tt_available:
        print()
        k_m = float(np.mean(kernels))
        e2e_m = float(np.mean(uploads)) + k_m + float(np.mean(readbacks))
        k_speedup = cpu_mean / k_m
        e2e_speedup = cpu_mean / e2e_m
        print("  Speedup CPU / TT pre-mult kernel   : %6.2fx" % k_speedup)
        print("  Speedup CPU / TT pre-mult e2e      : %6.2fx" % e2e_speedup)
        if not python_tt_workloads_match():
            print("  (Speedup not apples-to-apples: CPU = Python grid; TT = compiled binary grid.)")
        elif k_speedup > 1:
            print("  → TT pre-mult kernel is %.2fx faster than CPU scatter." % k_speedup)
        else:
            print("  → CPU is %.2fx faster than TT pre-mult kernel." % (1.0 / k_speedup))

    if cpu_available and tt_geom_available and gu:
        print()
        gk_m = float(np.mean(gk))
        ge2e = float(np.mean(gu)) + gk_m + float(np.mean(gr))
        print("  Speedup CPU / TT geometry kernel   : %6.2fx" % (cpu_mean / gk_m))
        print("  Speedup CPU / TT geometry e2e      : %6.2fx" % (cpu_mean / ge2e))
    print()
    print("  ── Accuracy (TT vs DREAMPlace CPU / numpy reference) ───────────")
    if acc is not None:
        print("  Formula rel-L2  (numpy exact): %.4e  %s" % (
            acc["rel_l2"],
            "✓ PASS (<2%)" if acc["rel_l2"] < 0.02 else
            "~ WARN (<5%)" if acc["rel_l2"] < 0.05 else "✗ FAIL (>5%)"))
        print("  Integrated Δ    (numpy exact): %.4e" % acc["rel_total"])
    if tt_available and not math.isnan(tt_rel_l2):
        print("  SFPU rel-L2     (TT pre-mult)  : %.4e  %s" % (
            tt_rel_l2,
            "✓ PASS (<2%)" if tt_rel_l2 < 0.02 else
            "~ WARN (<5%)" if tt_rel_l2 < 0.05 else "✗ FAIL (>5%)"))
        print("  SFPU integrated Δ (pre-mult)   : %.4e" % tt_rel_total)
    if tt_geom_available and not math.isnan(tt_g_rel_l2):
        print("  SFPU rel-L2     (TT geometry): %.4e  %s" % (
            tt_g_rel_l2,
            "✓ PASS (<2%)" if tt_g_rel_l2 < 0.02 else
            "~ WARN (<5%)" if tt_g_rel_l2 < 0.05 else "✗ FAIL (>5%)"))
        print("  SFPU integrated Δ (geometry) : %.4e" % tt_g_rel_total)
    if (
        acc is not None
        and tt_available
        and not math.isnan(tt_rel_l2)
        and python_tt_workloads_match()
    ):
        extra = tt_rel_l2 - acc["rel_l2"]
        print()
        print("  Extra pre-mult SFPU vs numpy formula: +%.4e rel-L2" % extra)
    elif acc is not None and tt_available and not python_tt_workloads_match():
        print()
        print("  (Skipping pre-mult vs formula delta: workload mismatch.)")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
