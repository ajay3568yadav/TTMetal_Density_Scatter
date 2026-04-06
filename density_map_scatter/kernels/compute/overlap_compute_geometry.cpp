// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ─────────────────────────────────────────────────────────────────────────────
// overlap_compute_geometry.cpp  —  GEOMETRY PATH  (FAILING ✗)
// ─────────────────────────────────────────────────────────────────────────────
//
// PURPOSE
// -------
// This is the SFPU compute kernel for the "geometry-dependent" variant.
// Unlike overlap_compute.cpp (which receives pre-computed px/py from the host),
// this kernel receives raw bin/cell edges and computes the x- and y-overlaps
// entirely on the Tensix SFPU, using BIN_SIZE from JIT compile-time constants:
//
//   Pass 1 (run_px):
//     bxh = bxl + BIN_SIZE_X           — right edge of this bin
//     r   = min(cxr, bxh)              — right edge of overlap
//     l   = max(cxl, bxl)              — left edge of overlap
//     px  = clamp(r - l, 0, ∞)         — x-overlap width, clamped to ≥ 0
//
//   Pass 2 (run_py):
//     byh = byl + BIN_SIZE_Y
//     r   = min(cyr, byh)
//     l   = max(cyl, byl)
//     py  = clamp(r - l, 0, ∞)         — y-overlap height, clamped to ≥ 0
//
//   Pass 3 (run_area):
//     area = px * py                   — ratio=1.0 hardcoded; not uploaded
//
// ─────────────────────────────────────────────────────────────────────────────
// ACCURACY STATUS:  ✗ FAILS
// ─────────────────────────────────────────────────────────────────────────────
//
// Benchmark result (500k cells, 2048² grid, seed=2025 — deterministic):
//   rel L2            : 0.17886  (gate: ≤ 0.05)
//   integrated rel err: 0.01821  (gate: ≤ 0.002)
//   sample errors     : task 0: expected=0.824, tt=0.448
//                       task 9371: expected=2.146, tt=0.000
//
// The pre-multiply path (overlap_compute.cpp) — which receives the same per-task
// area via px/py inputs — passes with rel L2 ~1.7e-4.  So the bug is entirely
// in the geometry computation in THIS file.
//
// ─────────────────────────────────────────────────────────────────────────────
// THE OPEN BUG — SFPI DST ADDRESSING ACROSS SEQUENTIAL TERNARY PASSES
// ─────────────────────────────────────────────────────────────────────────────
//
// This kernel chains THREE sequential calls to _llk_math_eltwise_ternary_sfpu_params_
// inside a single tile_regs_acquire() region.  Each call processes 4 faces of one tile.
//
// INTENDED FLOW:
//   run_px  → writes px to DST slot 3
//   run_py  → writes py to DST slot 4
//   run_area→ reads px from DST 3, py from DST 4 → writes area to DST 7
//
// THE QUESTION: inside each call to _llk_math_eltwise_ternary_sfpu_params_, the LLK
// emits TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D) between face iterations to advance
// the hardware dest write pointer by 8 lanes.  After 4 faces × 8 lanes = 32 lanes,
// the pointer has advanced by one full tile width.
//
// It is UNRESOLVED whether dst_reg[d * N + i] in an SFPI lambda is:
//   (a) ABSOLUTE — d=3 always means global DST tile 3, independent of SETRWC.
//   (b) RELATIVE — d=3 means "3 tiles ahead of the current SETRWC base pointer",
//       so after run_px has advanced the base by 1 tile, d=3 in run_area reads from
//       global tile 3+1=4, NOT the tile where px was stored.
//
// If (b), the chained passes read wrong data, explaining the stable per-task errors.
// This is the most likely root cause but has not been confirmed by hardware docs
// or assembly inspection.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT HAS BEEN TRIED (ALL FAILED — see REPRODUCE.md for full record)
// ─────────────────────────────────────────────────────────────────────────────
//
//  Attempt 1 — Original: ternary area pass with in2==out (alias)
//    Using the same slot for the dummy third operand and the output is unsafe
//    on this LLK — changed to use byl in slot 2 as a real non-aliased dummy.
//    Result: no change in accuracy (error was not caused by aliasing).
//
//  Attempt 2 — Replace area pass with stock mul_binary_tile()
//    The standard binary multiply from eltwise_binary_sfpu.h uses a DIFFERENT
//    dest walk (sfpi::dst_reg++ per lane, not d*N+i absolute indexing).  This
//    is layout-incompatible with how run_px/run_py wrote their results.
//    Result: identical wrong output.
//
//  Attempt 3 — Custom binary face using _llk_math_eltwise_binary_sfpu_params_
//    Custom face using dst_reg[d*N+i] style to match run_px/run_py layout,
//    but calling mul_binary_tile_init() first (which reprograms ADDR_MOD_7,
//    corrupting the addr-mod state needed by ternary passes).
//    Result: identical wrong output.
//    LESSON: do NOT call mul_binary_tile_init() in the same tile loop as
//    ternary passes — it corrupts SFPU address-modulator register ADDR_MOD_7.
//
//  Attempt 4 — Move px/py to high DST slots (5, 6)
//    Hypothesis: slots 3/4 might alias with unpack staging in FP32 half-sync mode.
//    Result: catastrophic — rel L2 > 800, total area ~11 billion (should be ~18 million).
//    LESSON: in FP32 half-sync mode on Wormhole, only DST slots 0–3 are valid.
//    Slots ≥ 4 read from out-of-bounds dest memory and produce garbage.
//    This is confirmed by get_dest_max_tiles<SyncHalf, FP32, Tile32x32>() = 4.
//
//  Attempt 5 — Compute py before px
//    Hypothesis: loading y inputs overwrites x inputs in slots 0–2, corrupting px.
//    Result: catastrophic for the same reason as Attempt 4.
//
//  Attempt 6 — Insert tile_regs_commit()/tile_regs_wait() between run_px and run_py
//    Hypothesis: "flushing" px before the y pass would preserve it.
//    Result: DEVICE HANGS. Splitting an acquire region with a commit/wait between
//    SFPU passes is not a valid pattern on this hardware.
//
// ─────────────────────────────────────────────────────────────────────────────
// SUGGESTED EXPERT EXPERIMENTS (see REPRODUCE.md for full details)
// ─────────────────────────────────────────────────────────────────────────────
//
//  Experiment A — Verify absolute vs relative DST addressing
//    Write a minimal kernel: pass 1 stores constant 42.0 to dst 3; pass 2 reads
//    dst 3 and writes to dst 7; pack dst 7.  If output == 42.0, addressing is
//    absolute and the bug is elsewhere.  If output is 0 or garbage, addressing
//    is relative and the intermediate values are being read from the wrong slot.
//
//  Experiment B — CB round-trip for intermediate values
//    After run_px: pack dst 3 → scratch CB, then copy_tile scratch → dst 0.
//    Then run_py using dst 0 for px input instead of dst 3.  This sidesteps
//    the SETRWC question entirely and tests whether the px value itself is correct.
//
//  Experiment C — Count SETRWC advances in the ternary params wrapper
//    In llk_math_eltwise_ternary_sfpu_params.h, confirm whether
//    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(0) resets the write
//    base to tile 0 at the start of EACH call (making addressing absolute) or
//    only once at the first call (making addressing accumulate across calls).
//
// ─────────────────────────────────────────────────────────────────────────────
// CIRCULAR BUFFER LAYOUT (this kernel)
// ─────────────────────────────────────────────────────────────────────────────
//
//   INPUTS  (produced by read_tiles_geometry.cpp, consumed here):
//     c_0 = cxl  — left edge of cell in x
//     c_1 = cxr  — right edge of cell in x
//     c_2 = bxl  — left edge of bin in x  (bxh = bxl + BIN_SIZE_X on device)
//     c_3 = cyl  — left edge of cell in y
//     c_4 = cyr  — right edge of cell in y
//     c_5 = byl  — left edge of bin in y  (byh = byl + BIN_SIZE_Y on device)
//   OUTPUT:
//     c_16 = area (per-task overlap area; consumed by write_tile.cpp)
//
// DST REGISTER SLOT ASSIGNMENT (within one tile_regs_acquire() region):
//   After x-unpack:  0=cxl  1=cxr  2=bxl
//   After run_px:    0=cxl  1=cxr  2=bxl  3=px   (assuming absolute addressing)
//   After y-unpack:  0=cyl  1=cyr  2=byl  3=px   (0–2 overwritten; 3 preserved?)
//   After run_py:    0=cyl  1=cyr  2=byl  3=px  4=py
//   After run_area:  7=area  (reads from 3 and 4)
//
// Slot 7 is used for area to match the working pre-multiply kernel (overlap_compute.cpp).
//
// ─────────────────────────────────────────────────────────────────────────────
// JIT COMPILE-TIME CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
//
// BIN_SIZE_X and BIN_SIZE_Y are NOT runtime arguments — they are baked into the
// RISC-V binary at kernel JIT compile time.  The host passes them as:
//   DMS_BIN_SIZE_X_F = 1.46484375f  (= 3000.0 / 2048, exact binary fraction)
//   DMS_BIN_SIZE_Y_F = 1.46484375f
//
// These are confirmed to match the host-side float by instrumentation (not the bug).
// The fallback path (integer domain/bins division) is only for other configurations.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#ifdef TRISC_MATH
// See the long comment in overlap_compute.cpp for what this header provides.
#include "llk_math_eltwise_ternary_sfpu_params.h"
#endif

// ── BIN_SIZE constants ────────────────────────────────────────────────────────
// Preferred path: host passes exact float literals matching the enumeration code.
// DMS_BIN_SIZE_X_F = "1.46484375f"  — passed via JIT define as a float token,
// NOT as a string; the C preprocessor substitutes it directly into the constexpr.
#if defined(DMS_BIN_SIZE_X_F) && defined(DMS_BIN_SIZE_Y_F)
static constexpr float BIN_SIZE_X = DMS_BIN_SIZE_X_F;
static constexpr float BIN_SIZE_Y = DMS_BIN_SIZE_Y_F;
#else
// Fallback: compute from integer domain/bins — may lose precision vs host float.
// This path is only taken if the host did not pass DMS_BIN_SIZE_*_F defines.
#ifndef DMS_NUM_BINS_KERNEL
#define DMS_NUM_BINS_KERNEL 256
#endif
#ifndef DMS_DOMAIN_WIDTH
#define DMS_DOMAIN_WIDTH 1000
#endif
#ifndef DMS_DOMAIN_HEIGHT
#define DMS_DOMAIN_HEIGHT DMS_DOMAIN_WIDTH
#endif
static constexpr float BIN_SIZE_X =
    static_cast<float>(DMS_DOMAIN_WIDTH) / static_cast<float>(DMS_NUM_BINS_KERNEL);
static constexpr float BIN_SIZE_Y =
    static_cast<float>(DMS_DOMAIN_HEIGHT) / static_cast<float>(DMS_NUM_BINS_KERNEL);
#endif

// N = 32 vFloat lanes per tile row.  A tile has 32×32 = 1024 elements total,
// split into 4 faces of 8×8 = 64 elements each.  The face loop runs 8 iterations
// of 8-wide vFloat (i = 0..7).
static constexpr uint32_t N = 32;

// ─────────────────────────────────────────────────────────────────────────────
// px_tile_face: compute x-overlap for one 8×8 face of the tile.
//
// This implements the clamped DREAMPlace overlap formula for x:
//   bxh = bxl + BIN_SIZE_X            — derive right edge of bin on device
//   r   = min(cxr, bxh)               — rightmost valid overlap edge
//   l   = max(cxl, bxl)               — leftmost valid overlap edge
//   px  = clamp(r - l, 0, ∞)          — overlap width; negative → zero
//
// The v_if / v_endif blocks are SFPI predicated execution: they compile to
// hardware scalar-compare + lane-mask instructions (not branching).  All 8 lanes
// execute both sides; the mask selects which lanes take the "if" path.
//
// dst_reg[d * N + i]:
//   d = tile index (0–3 reliably in FP32 half-sync mode on Wormhole)
//   i = lane within the face (0–7)
//   N = 32 = lanes per full tile row
//   Combined: absolute position d*32 + i in the DST register array.
//
// THE OPEN QUESTION: after the ternary LLK advances the SETRWC pointer between
// faces, does d*N+i still map to the same physical DST location, or does it
// shift relative to the new base?  See the file header for full discussion.
// ─────────────────────────────────────────────────────────────────────────────
#ifdef TRISC_MATH
inline void px_tile_face(
    const uint32_t d_cxl, const uint32_t d_cxr, const uint32_t d_bxl, const uint32_t d_px) {
    for (uint32_t i = 0; i < 8; i++) {
        vFloat bxl = dst_reg[d_bxl * N + i];
        vFloat bxh = bxl + BIN_SIZE_X;    // right edge of this bin (compile-time constant added)

        // r = min(cxr, bxh) — right side of overlap
        vFloat cxr = dst_reg[d_cxr * N + i];
        vFloat r   = cxr;
        v_if (bxh < cxr) { r = bxh; }    // if bin ends before cell, clip to bin right edge
        v_endif;

        // l = max(cxl, bxl) — left side of overlap
        vFloat l   = bxl;
        vFloat cxl = dst_reg[d_cxl * N + i];
        v_if (cxl > bxl) { l = cxl; }    // if cell starts after bin left, clip to cell left edge
        v_endif;

        // px = clamp(r - l, 0) — overlap width; negative means no overlap
        vFloat px = r - l;
        v_if (px < sfpi::vConst0) { px = sfpi::vConst0; }  // hardware zero constant
        v_endif;

        dst_reg[d_px * N + i] = px;       // store result in the px tile slot
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// py_tile_face: compute y-overlap for one face (symmetric with px_tile_face).
// ─────────────────────────────────────────────────────────────────────────────
inline void py_tile_face(
    const uint32_t d_cyl, const uint32_t d_cyr, const uint32_t d_byl, const uint32_t d_py) {
    for (uint32_t i = 0; i < 8; i++) {
        vFloat byl = dst_reg[d_byl * N + i];
        vFloat byh = byl + BIN_SIZE_Y;

        vFloat cyr = dst_reg[d_cyr * N + i];
        vFloat r   = cyr;
        v_if (byh < cyr) { r = byh; }
        v_endif;

        vFloat l   = byl;
        vFloat cyl = dst_reg[d_cyl * N + i];
        v_if (cyl > byl) { l = cyl; }
        v_endif;

        vFloat py = r - l;
        v_if (py < sfpi::vConst0) { py = sfpi::vConst0; }
        v_endif;

        dst_reg[d_py * N + i] = py;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// area_tile_face: compute area = px * py for one face.
//
// d_dummy is the third ternary LLK operand (required by the template interface
// but not used in the computation).  It MUST be a different slot from d_area —
// passing in2 == out to the ternary LLK is unsafe on this hardware.
// We pass the byl slot (2) which is already populated and distinct from area (7).
//
// IMPORTANT — things that DO NOT work here (see file header for full history):
//   • Do NOT call mul_binary_tile_init() in the same tile loop.  It reprograms
//     SFPU ADDR_MOD_7, corrupting the addr-mod state used by ternary passes.
//   • Do NOT use dst_reg++ style (stock binary mul) — it walks DST sequentially
//     and is layout-incompatible with the d*N+i absolute indexing used here.
// ─────────────────────────────────────────────────────────────────────────────
inline void area_tile_face(
    const uint32_t d_px, const uint32_t d_py, const uint32_t /*d_dummy*/, const uint32_t d_area) {
    for (uint32_t i = 0; i < 8; i++) {
        dst_reg[d_area * N + i] = dst_reg[d_px * N + i] * dst_reg[d_py * N + i];
    }
}
#endif  // TRISC_MATH

// ─────────────────────────────────────────────────────────────────────────────
// run_px / run_py / run_area: wrappers calling _llk_math_eltwise_ternary_sfpu_params_.
//
// Each call processes a full 32×32 tile = 4 faces × 8 lanes each.
// VectorMode::RC = "row and column" = process all four faces of the tile.
//
// AFTER run_px:  px should be in DST slot o (=3 in kernel_main).
// AFTER run_py:  py should be in DST slot o (=4 in kernel_main).
// AFTER run_area: area is in DST slot d_area (=7 in kernel_main).
//
// Whether "should be in DST slot 3" is actually true across sequential LLK calls
// is the open question — see the file header.
// ─────────────────────────────────────────────────────────────────────────────
inline void run_px(uint32_t a, uint32_t b, uint32_t c, uint32_t o) {
    MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
        px_tile_face, a, b, c, o, static_cast<int>(VectorMode::RC)));
}

inline void run_py(uint32_t a, uint32_t b, uint32_t c, uint32_t o) {
    MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
        py_tile_face, a, b, c, o, static_cast<int>(VectorMode::RC)));
}

// d_dummy_tile must be a real populated DST slot that is different from d_area.
// After run_py, slot 2 still holds byl which satisfies this constraint.
inline void run_area(uint32_t d_px, uint32_t d_py, uint32_t d_dummy_tile, uint32_t d_area) {
    MATH(_llk_math_eltwise_ternary_sfpu_params_<true>(
        area_tile_face, d_px, d_py, d_dummy_tile, d_area, static_cast<int>(VectorMode::RC)));
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_main
// ─────────────────────────────────────────────────────────────────────────────
void kernel_main() {
    // DeviceZoneScopedN emits a Tracy/profiler marker visible in the TT device
    // profiler report (PROFILING_REPORT.md).  One zone per kernel_main() call.
    DeviceZoneScopedN("DMS_OVERLAP_GEOMETRY_SFPU");

    // n_tiles: number of task-tiles this core handles.
    // Each tile holds up to 32×32 = 1024 per-task values (many tiles are partial).
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Input CBs — one per geometry input field.
    // All CBs carry FP32 tiles.  Naming matches the field names in density_map_scatter.cpp.
    constexpr auto cb_cxl  = tt::CBIndex::c_0;  // cell x-left
    constexpr auto cb_cxr  = tt::CBIndex::c_1;  // cell x-right
    constexpr auto cb_bxl  = tt::CBIndex::c_2;  // bin x-left  (bxh = bxl + BIN_SIZE_X on device)
    constexpr auto cb_cyl  = tt::CBIndex::c_3;  // cell y-left
    constexpr auto cb_cyr  = tt::CBIndex::c_4;  // cell y-right
    constexpr auto cb_byl  = tt::CBIndex::c_5;  // bin y-left  (byh = byl + BIN_SIZE_Y on device)
    // c_6 (ratio) was removed — ratio is hardcoded 1.0, saving ~200 MB DRAM upload.
    constexpr auto cb_area = tt::CBIndex::c_16; // output: per-task area = px * py

    // Configure SFPU data format and packer for this CB pair.
    init_sfpu(cb_cxl, cb_area);
    pack_reconfig_data_format(cb_area);

    for (uint32_t ti = 0; ti < n_tiles; ti++) {

        // ── WAIT for reader kernel (RISCV_0) to produce one tile in each input CB.
        cb_wait_front(cb_cxl, 1);
        cb_wait_front(cb_cxr, 1);
        cb_wait_front(cb_bxl, 1);
        cb_wait_front(cb_cyl, 1);
        cb_wait_front(cb_cyr, 1);
        cb_wait_front(cb_byl, 1);

        // ── ACQUIRE exclusive DST ownership for this tile iteration.
        tile_regs_acquire();

        // ── UNPACK x-group: load cxl, cxr, bxl into DST slots 0, 1, 2.
        //    copy_tile_init must be called before each copy_tile to reconfigure
        //    the unpacker for the source CB's data format.
        copy_tile_init(cb_cxl);
        copy_tile(cb_cxl, 0, /*dst=*/0);  // cxl → DST[0]
        copy_tile_init(cb_cxr);
        copy_tile(cb_cxr, 0, /*dst=*/1);  // cxr → DST[1]
        copy_tile_init(cb_bxl);
        copy_tile(cb_bxl, 0, /*dst=*/2);  // bxl → DST[2]

        // ── SFPU pass 1: compute px from {cxl, cxr, bxl}, write to DST[3].
        //    After this call, DST[3] should hold px (if addressing is absolute).
        //    DST[0–2] are reused for y-group inputs next and will be overwritten.
        run_px(/*cxl=*/0, /*cxr=*/1, /*bxl=*/2, /*px=*/3);

        // ── UNPACK y-group: overwrite DST slots 0, 1, 2 with cyl, cyr, byl.
        //    DST[3] (px) and DST[4] (unused so far) are NOT touched by copy_tile.
        copy_tile_init(cb_cyl);
        copy_tile(cb_cyl, 0, /*dst=*/0);  // cyl → DST[0]  (cxl overwritten)
        copy_tile_init(cb_cyr);
        copy_tile(cb_cyr, 0, /*dst=*/1);  // cyr → DST[1]  (cxr overwritten)
        copy_tile_init(cb_byl);
        copy_tile(cb_byl, 0, /*dst=*/2);  // byl → DST[2]  (bxl overwritten)

        // ── SFPU pass 2: compute py from {cyl, cyr, byl}, write to DST[4].
        //    DST[2] (byl) remains intact after this and is used as the dummy
        //    operand in run_area below.
        run_py(/*cyl=*/0, /*cyr=*/1, /*byl=*/2, /*py=*/4);

        // ── SFPU pass 3: area = DST[3] * DST[4], write to DST[7].
        //    CRITICAL: DST[3] and DST[4] must still hold the px and py values
        //    written by run_px and run_py above.  Whether they do depends on
        //    whether the SETRWC advances in those two calls invalidated those
        //    slots — this is the unresolved open question.
        //
        //    DO NOT insert tile_regs_commit()/tile_regs_wait() between these passes.
        //    Splitting an acquire region with commit/wait between SFPU passes
        //    hangs the device (confirmed experimentally).
        run_area(/*px=*/3, /*py=*/4, /*dummy_byl=*/2, /*area=*/7);

        // ── COMMIT/WAIT: signal math complete; wait for pack to be ready.
        tile_regs_commit();
        tile_regs_wait();

        // ── PACK: write area from DST[7] to the output CB.
        cb_reserve_back(cb_area, 1);
        pack_tile(/*dst=*/7, cb_area);
        cb_push_back(cb_area, 1);  // signals write_tile.cpp that a result is ready

        // ── POP all input CBs so the reader can refill them.
        cb_pop_front(cb_cxl, 1);
        cb_pop_front(cb_cxr, 1);
        cb_pop_front(cb_bxl, 1);
        cb_pop_front(cb_cyl, 1);
        cb_pop_front(cb_cyr, 1);
        cb_pop_front(cb_byl, 1);

        // ── RELEASE DST so the next iteration can acquire it.
        tile_regs_release();
    }
}
