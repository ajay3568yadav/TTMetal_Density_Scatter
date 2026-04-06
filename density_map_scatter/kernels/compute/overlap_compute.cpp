// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ─────────────────────────────────────────────────────────────────────────────
// overlap_compute.cpp  —  PRE-MULTIPLY PATH  (PASSING ✓)
// ─────────────────────────────────────────────────────────────────────────────
//
// PURPOSE
// -------
// This is the SFPU compute kernel for the "pre-multiply" variant of the density
// map scatter benchmark.  It performs the simplest possible on-device computation:
// a single three-operand SFPU pass computing:
//
//   area = px * ratio * py
//
// The host has already computed px and py using the cell/bin geometry:
//   px = max(0, min(cxr, bxl + BIN_SIZE_X) - max(cxl, bxl))
//   py = max(0, min(cyr, byl + BIN_SIZE_Y) - max(cyl, byl))
//
// This path PASSES accuracy (rel L2 ~1.7e-4) and serves as the reference baseline.
// Compare this file with overlap_compute_geometry.cpp (which FAILS) to understand
// what the geometry kernel is trying to do and why it's harder.
//
// ─────────────────────────────────────────────────────────────────────────────
// TENSTORRENT ARCHITECTURE PRIMER (Wormhole / Tensix core)
// ─────────────────────────────────────────────────────────────────────────────
//
// A Tensix core has THREE sub-processors (TRISCs):
//   TRISC_UNPACK  (RISCV_0 alias: data movement from CB → DST registers)
//   TRISC_MATH    (does the actual SFPU computation on DST registers)
//   TRISC_PACK    (RISCV_1 alias: moves results from DST registers → CB)
//
// In practice this file is compiled ONCE per TRISC, gated by the preprocessor.
// The MATH(x) macro expands x only when compiled for TRISC_MATH; it is a no-op
// on the other two TRISCs.  Similarly, UNPACK(x) and PACK(x) exist for the
// other two sub-processors.
//
// DATA FLOW PER TILE:
//   DRAM → [reader kernel / NOC] → Circular Buffer (CB) → copy_tile → DST registers
//   DST registers → [SFPU math] → DST registers
//   DST registers → pack_tile → CB → [writer kernel / NOC] → DRAM
//
// TILES:
//   A "tile" is a 32×32 block of FP32 values = 1024 floats = 4 KB.
//   Physically it is stored as four 8×8 "faces".  The SFPU processes one face
//   (64 elements) at a time, 8 vFloat lanes wide × 8 iterations.
//   N = 32 below is the number of vFloat lanes per tile row (= 32×1 in RC mode).
//
// DST REGISTERS:
//   The destination register file holds multiple tiles.  In FP32 half-sync mode
//   on Wormhole, only tiles 0–3 are reliably accessible (4 × 32×32 FP32 = 16 KB).
//   Tile slots above 3 produce garbage in this mode — do not use them (see
//   overlap_compute_geometry.cpp Attempt 4 for evidence).
//
// CIRCULAR BUFFERS (CBs):
//   On-chip L1 SRAM staging areas.  Indexed by CBIndex::c_N.
//   cb_wait_front(cb, n)  — blocks until n tiles are available to read.
//   cb_reserve_back(cb, n) — blocks until n slots are free to write.
//   cb_push_back / cb_pop_front — mark tiles as produced/consumed.
//
// TILE REGISTER LIFECYCLE:
//   tile_regs_acquire()  — take exclusive ownership of DST register file.
//   copy_tile(cb, slot, dst) — unpack tile from CB slot into DST register.
//   [SFPU math on DST]
//   tile_regs_commit()   — signal math is done; pack can start reading DST.
//   tile_regs_wait()     — pack waits until DST is ready to read.
//   pack_tile(dst, cb)   — write DST tile to output CB.
//   tile_regs_release()  — release ownership of DST so next iteration can start.
//
// ─────────────────────────────────────────────────────────────────────────────
// CIRCULAR BUFFER LAYOUT (this kernel)
// ─────────────────────────────────────────────────────────────────────────────
//
//   c_0  = px    — x-overlap (pre-computed by host, clamped ≥ 0)
//   c_1  = py    — y-overlap (pre-computed by host, clamped ≥ 0)
//   c_2  = ratio — per-cell scaling factor (= 1.0 in this benchmark)
//   c_16 = area  — OUTPUT: area = px * ratio * py
//
// DST REGISTER SLOT ASSIGNMENT:
//   dst 0 ← px
//   dst 1 ← py
//   dst 2 ← ratio
//   dst 7 ← area (output; written by SFPU, read by pack)
//
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
// llk_math_eltwise_ternary_sfpu_params.h provides the template:
//   _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(face_fn, in0, in1, in2, out, vmode)
// It iterates over the 4 faces of a 32×32 tile, calling face_fn(in0, in1, in2, out)
// for each face.  Between faces it emits TTI_SETRWC to advance the hardware
// dest write pointer by 8 lanes.
#include "llk_math_eltwise_ternary_sfpu_params.h"
#endif

// N = 32: number of vFloat lanes per tile row.
// In SFPI, dst_reg[tile * N + i] accesses lane i of the given tile.
// A tile face is 8×8 = 64 values, processed as 8 iterations of 8-wide vFloat.
static constexpr uint32_t N = 32;

// ─────────────────────────────────────────────────────────────────────────────
// area_tile_face: computes area = px * ratio * py for one 8-wide face.
//
// Called by _llk_math_eltwise_ternary_sfpu_params_ four times (once per face).
// Between calls, the LLK emits TTI_SETRWC(SET_D) to advance the write pointer
// by 8 lanes so the next face starts at the correct offset.
//
// Arguments are DST tile indices (not CB indices):
//   d_px    — tile slot containing px values
//   d_py    — tile slot containing py values
//   d_ratio — tile slot containing ratio values
//   d_area  — tile slot to write area into
//
// dst_reg[d * N + i] is SFPI syntax for reading/writing DST register lane i
// of tile d.  This indexing style is used consistently in this file and in
// overlap_compute_geometry.cpp.
//
// v_if / v_endif are SFPI predicated-execution macros (hardware scalar compare +
// lane mask).  Not needed here since px, py, ratio are all already non-negative.
// ─────────────────────────────────────────────────────────────────────────────
#ifdef TRISC_MATH
inline void area_tile_face(
    const uint32_t d_px, const uint32_t d_py, const uint32_t d_ratio, const uint32_t d_area) {
    for (uint32_t i = 0; i < 8; i++) {
        vFloat px    = dst_reg[d_px    * N + i];
        vFloat py    = dst_reg[d_py    * N + i];
        vFloat ratio = dst_reg[d_ratio * N + i];
        dst_reg[d_area * N + i] = px * ratio * py;
    }
}
#endif  // TRISC_MATH

// ─────────────────────────────────────────────────────────────────────────────
// run_area: thin wrapper that invokes the ternary LLK for one tile.
//
// _llk_math_eltwise_ternary_sfpu_params_<true>(...) signature:
//   face_fn  — called once per face with (in0_tile, in1_tile, in2_tile, out_tile)
//   a, b, c  — DST tile indices for in0, in1, in2
//   o        — DST tile index for output
//   vmode    — VectorMode::RC = row+column = process all 4 faces
//
// The <true> template arg enables the "approximate" path (no special handling
// needed here; it selects the SFPU instruction set variant).
//
// The MATH(...) macro ensures this only executes on the TRISC_MATH sub-processor.
// ─────────────────────────────────────────────────────────────────────────────
inline void run_area(uint32_t a, uint32_t b, uint32_t c, uint32_t o) {
    MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
        area_tile_face, a, b, c, o, static_cast<int>(VectorMode::RC)));
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_main: called once per Tensix core.
//
// Each core processes a slice of the total work (n_tiles tiles).
// tile_start is the global tile index offset so all cores read from
// the correct position in the shared DRAM buffers.
// ─────────────────────────────────────────────────────────────────────────────
void kernel_main() {
    // n_tiles: how many area tiles this core must process (may differ by ±1 across cores).
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Circular buffer indices — must match host-side CB setup in density_map_scatter.cpp.
    constexpr auto cb_px    = tt::CBIndex::c_0;
    constexpr auto cb_py    = tt::CBIndex::c_1;
    constexpr auto cb_ratio = tt::CBIndex::c_2;
    constexpr auto cb_area  = tt::CBIndex::c_16;  // output CB; c_16 avoids conflict with input CBs

    // init_sfpu: configures SFPU data format and addressing for this CB pair.
    // Must be called once before any SFPU operations.
    init_sfpu(cb_px, cb_area);

    // pack_reconfig_data_format: tells the packer what data format the output CB uses.
    // Required after init_sfpu when the output CB differs from the first input CB.
    pack_reconfig_data_format(cb_area);

    for (uint32_t ti = 0; ti < n_tiles; ti++) {

        // ── WAIT: block until the reader kernel (RISCV_0) has produced 1 tile
        //    in each input CB.  The reader and compute run concurrently; this
        //    handshake is the synchronisation point between them.
        cb_wait_front(cb_px,    1);
        cb_wait_front(cb_py,    1);
        cb_wait_front(cb_ratio, 1);

        // ── ACQUIRE: take exclusive ownership of the DST register file.
        //    No other kernel can touch DST until tile_regs_release() is called.
        tile_regs_acquire();

        // ── UNPACK: copy each input tile from its CB into a DST register slot.
        //    copy_tile_init(cb) — reconfigures the unpacker for this CB's format.
        //    copy_tile(cb, cb_slot, dst_slot) — DMA from CB to DST register.
        //    cb_slot=0 because each CB holds exactly 1 tile at a time here.
        copy_tile_init(cb_px);
        copy_tile(cb_px, 0, /*dst=*/0);   // px   → DST[0]

        copy_tile_init(cb_py);
        copy_tile(cb_py, 0, /*dst=*/1);   // py   → DST[1]

        copy_tile_init(cb_ratio);
        copy_tile(cb_ratio, 0, /*dst=*/2); // ratio → DST[2]

        // ── SFPU MATH: area = px * ratio * py
        //    Writes result into DST slot 7.  Using slot 7 (not 3 or 4) avoids
        //    clobbering the input slots in case the LLK read order matters.
        run_area(/*px=*/0, /*py=*/1, /*ratio=*/2, /*area=*/7);

        // ── COMMIT/WAIT: signal math is complete and wait until the packer
        //    is ready to read DST.  These two calls are always paired.
        tile_regs_commit();
        tile_regs_wait();

        // ── PACK: write the result from DST slot 7 into the output CB.
        cb_reserve_back(cb_area, 1);      // wait for a free slot in the output CB
        pack_tile(/*dst=*/7, cb_area);    // DST[7] → cb_area
        cb_push_back(cb_area, 1);         // signal the writer kernel (RISCV_1) that a tile is ready

        // ── POP: release the input CB slots so the reader can refill them.
        cb_pop_front(cb_px,    1);
        cb_pop_front(cb_py,    1);
        cb_pop_front(cb_ratio, 1);

        // ── RELEASE: give up DST ownership so the next iteration can acquire it.
        tile_regs_release();
    }
}
