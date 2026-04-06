// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ─────────────────────────────────────────────────────────────────────────────
// read_tiles_geometry.cpp  —  READER KERNEL, GEOMETRY PATH
// ─────────────────────────────────────────────────────────────────────────────
//
// PURPOSE
// -------
// This is the data-movement (RISCV_0) kernel for the geometry-dependent variant
// of the density map scatter benchmark.
//
// Instead of the 3 pre-computed inputs (px, py, ratio) used by read_tiles.cpp,
// this kernel reads 6 raw bin/cell edge values per task from DRAM and fills 6
// Circular Buffers for the compute kernel (overlap_compute_geometry.cpp).
//
// The compute kernel then derives px, py, and area entirely on the SFPU.
// ratio is hardcoded 1.0 in the compute kernel — no ratio buffer is needed here,
// saving ~200 MB of DRAM bandwidth per run (1 out of 4 buffers eliminated).
//
// INPUTS (from DRAM, FP32, one value per task, tile-packed):
//   cxl — cell x left edge
//   cxr — cell x right edge
//   bxl — bin x left edge  (bxh = bxl + BIN_SIZE_X is computed on device)
//   cyl — cell y left edge
//   cyr — cell y right edge
//   byl — bin y left edge  (byh = byl + BIN_SIZE_Y is computed on device)
//
// WHY BIN_SIZE IS NOT IN THESE BUFFERS:
//   All tasks in a given benchmark run use the same BIN_SIZE_X and BIN_SIZE_Y
//   (= 3000/2048 = 1.46484375 in the large benchmark).  Rather than uploading
//   a constant tile, BIN_SIZE is encoded as a JIT compile-time constant in the
//   compute kernel (DMS_BIN_SIZE_X_F / DMS_BIN_SIZE_Y_F defines).
//   The host confirms the JIT constant matches its own float using instrumentation.
//
// ─────────────────────────────────────────────────────────────────────────────
// ARCHITECTURE NOTE
// ─────────────────────────────────────────────────────────────────────────────
//
// This kernel is structurally identical to read_tiles.cpp, just with 6 buffers
// instead of 3.  The same TensorAccessor chaining pattern is used — see
// read_tiles.cpp for the detailed primer on TensorAccessor, NOC DMA, and CBs.
//
// PERFORMANCE NOTE:
//   6 DRAM inputs vs 3 means roughly 2× more DRAM bandwidth.
//   Measured upload time: ~150 ms (1.2 GB) vs ~49 ms (0.6 GB) for pre-multiply.
//   The kernel is still DRAM-bandwidth-bound, not compute-bound.
//   Device kernel time: 8.26 ms (geometry) vs 3.84 ms (pre-multiply).
//
// ─────────────────────────────────────────────────────────────────────────────
// RUNTIME ARGUMENTS (set by host in density_map_scatter.cpp):
//   [0] addr_cxl   — DRAM base of cell x-left buffer
//   [1] addr_cxr   — DRAM base of cell x-right buffer
//   [2] addr_bxl   — DRAM base of bin x-left buffer
//   [3] addr_cyl   — DRAM base of cell y-left buffer
//   [4] addr_cyr   — DRAM base of cell y-right buffer
//   [5] addr_byl   — DRAM base of bin y-left buffer
//   [6] n_tiles    — number of tiles this core must read
//   [7] tile_start — global tile index of this core's first tile
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>

void kernel_main() {
    // ── Runtime arguments ────────────────────────────────────────────────────
    const uint32_t addr_cxl   = get_arg_val<uint32_t>(0);
    const uint32_t addr_cxr   = get_arg_val<uint32_t>(1);
    const uint32_t addr_bxl   = get_arg_val<uint32_t>(2);
    const uint32_t addr_cyl   = get_arg_val<uint32_t>(3);
    const uint32_t addr_cyr   = get_arg_val<uint32_t>(4);
    const uint32_t addr_byl   = get_arg_val<uint32_t>(5);
    const uint32_t n_tiles    = get_arg_val<uint32_t>(6);
    const uint32_t tile_start = get_arg_val<uint32_t>(7);

    // ── Circular buffer indices (must match compute kernel and host setup) ────
    // These correspond to the six geometry inputs in overlap_compute_geometry.cpp.
    constexpr uint32_t cb_cxl = tt::CBIndex::c_0;
    constexpr uint32_t cb_cxr = tt::CBIndex::c_1;
    constexpr uint32_t cb_bxl = tt::CBIndex::c_2;
    constexpr uint32_t cb_cyl = tt::CBIndex::c_3;
    constexpr uint32_t cb_cyr = tt::CBIndex::c_4;
    constexpr uint32_t cb_byl = tt::CBIndex::c_5;

    // All CBs carry FP32 tiles; tile size is 32×32×4 = 4096 bytes.
    const uint32_t tile_bytes = get_tile_size(cb_cxl);

    // ── TensorAccessors: chained compile-time arg slots (a0 → a1 → ... → a5).
    //    Each accessor consumes a fixed number of compile-time arg slots.
    //    next_compile_time_args_offset() returns the next free slot so they
    //    don't collide.  Changing the order here would break the chain.
    constexpr auto a0 = TensorAccessorArgs<0>();
    const auto in_cxl = TensorAccessor(a0, addr_cxl, tile_bytes);

    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const auto in_cxr = TensorAccessor(a1, addr_cxr, tile_bytes);

    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    const auto in_bxl = TensorAccessor(a2, addr_bxl, tile_bytes);

    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto in_cyl = TensorAccessor(a3, addr_cyl, tile_bytes);

    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    const auto in_cyr = TensorAccessor(a4, addr_cyr, tile_bytes);

    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    const auto in_byl = TensorAccessor(a5, addr_byl, tile_bytes);

    // ── Main loop: read one tile from each of the 6 DRAM buffers per iteration.
    for (uint32_t i = 0; i < n_tiles; i++) {

        // Reserve one slot in every input CB.
        // If the compute kernel is slower than the reader, this blocks here
        // (back-pressure from the CB depth limit).
        cb_reserve_back(cb_cxl, 1);
        cb_reserve_back(cb_cxr, 1);
        cb_reserve_back(cb_bxl, 1);
        cb_reserve_back(cb_cyl, 1);
        cb_reserve_back(cb_cyr, 1);
        cb_reserve_back(cb_byl, 1);

        // L1 write pointers for each reserved CB slot.
        const uint32_t ptr_cxl = get_write_ptr(cb_cxl);
        const uint32_t ptr_cxr = get_write_ptr(cb_cxr);
        const uint32_t ptr_bxl = get_write_ptr(cb_bxl);
        const uint32_t ptr_cyl = get_write_ptr(cb_cyl);
        const uint32_t ptr_cyr = get_write_ptr(cb_cyr);
        const uint32_t ptr_byl = get_write_ptr(cb_byl);

        // g = global tile index — unique to this core's slice of the work.
        // All six inputs use the same tile index g (one task = one position
        // in every input buffer).
        const uint32_t g = tile_start + i;

        // Issue 6 non-blocking NOC DMA reads in parallel.
        // The NOC can pipeline transfers from different DRAM banks simultaneously.
        noc_async_read_tile(g, in_cxl, ptr_cxl);
        noc_async_read_tile(g, in_cxr, ptr_cxr);
        noc_async_read_tile(g, in_bxl, ptr_bxl);
        noc_async_read_tile(g, in_cyl, ptr_cyl);
        noc_async_read_tile(g, in_cyr, ptr_cyr);
        noc_async_read_tile(g, in_byl, ptr_byl);

        // Wait until all 6 DMA transfers are complete in L1.
        // The compute kernel's cb_wait_front() calls will not proceed until
        // cb_push_back() is called below.
        noc_async_read_barrier();

        // Mark all 6 CB slots as filled — wakes the compute kernel.
        cb_push_back(cb_cxl, 1);
        cb_push_back(cb_cxr, 1);
        cb_push_back(cb_bxl, 1);
        cb_push_back(cb_cyl, 1);
        cb_push_back(cb_cyr, 1);
        cb_push_back(cb_byl, 1);
    }
}
