// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ─────────────────────────────────────────────────────────────────────────────
// write_tile.cpp  —  WRITER KERNEL  (shared by both pre-multiply and geometry paths)
// ─────────────────────────────────────────────────────────────────────────────
//
// PURPOSE
// -------
// This is the data-movement (RISCV_1 / TRISC_PACK) kernel responsible for
// streaming computed per-task area tiles back from the compute kernel's output
// Circular Buffer (c_16) into DRAM.
//
// It is used by BOTH the pre-multiply path (overlap_compute.cpp) and the
// geometry path (overlap_compute_geometry.cpp) — the output format is the same
// in both cases: a flat tile-indexed FP32 array of per-task areas.
//
// DATA FLOW:
//   Compute kernel (SFPU on TRISC_MATH)
//     → pack_tile(7, cb_area)   [packs DST register into CB slot]
//     → cb_push_back(cb_area)   [signals writer that tile is ready]
//   This kernel:
//     → cb_wait_front(cb_area)  [blocks until compute pushes a tile]
//     → get_read_ptr(cb_area)   [L1 address of the filled slot]
//     → noc_async_write_tile()  [DMA from L1 to DRAM]
//     → noc_async_write_barrier()  [wait until DMA completes]
//     → cb_pop_front(cb_area)   [release the CB slot for reuse]
//
// AFTER ALL TILES ARE WRITTEN:
//   The host reads the DRAM output buffer (tt_areas) and scatters it into the
//   2D density map using the precomputed (t_bin_k, t_bin_h) indices.
//
// ─────────────────────────────────────────────────────────────────────────────
// RUNTIME ARGUMENTS (set by host in density_map_scatter.cpp):
//   [0] out_addr   — DRAM base address of the output area buffer
//   [1] n_tiles    — number of area tiles this core must write
//   [2] tile_start — global tile index of this core's first tile
//                    (ensures each core writes to its own non-overlapping
//                    slice of the shared output buffer)
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>

void kernel_main() {
    // ── Runtime arguments from the host ──────────────────────────────────────
    const uint32_t out_addr   = get_arg_val<uint32_t>(0);  // DRAM output buffer base
    const uint32_t n_tiles    = get_arg_val<uint32_t>(1);  // tiles to write on this core
    const uint32_t tile_start = get_arg_val<uint32_t>(2);  // this core's global tile offset

    // cb_area = c_16: the output CB written by pack_tile() in the compute kernel.
    // Using c_16 (not c_0–c_5) avoids any index conflict with input CBs.
    constexpr uint32_t cb_area = tt::CBIndex::c_16;

    // tile_bytes: size of one FP32 tile = 32×32×4 = 4096 bytes.
    const uint32_t tile_bytes = get_tile_size(cb_area);

    // TensorAccessor for the output buffer.
    // TensorAccessorArgs<0> uses compile-time arg slot 0 (only one output buffer).
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    // ── Main loop: one iteration = write one area tile to DRAM ───────────────
    for (uint32_t i = 0; i < n_tiles; i++) {

        // Block until the compute kernel has produced one tile in cb_area.
        // This is the producer-consumer synchronisation point between
        // TRISC_MATH (compute) and RISCV_1 (this writer).
        cb_wait_front(cb_area, 1);

        // Get the L1 address of the tile waiting in the CB.
        const uint32_t rd_ptr = get_read_ptr(cb_area);

        // Compute the global tile index and issue a non-blocking NOC write
        // from L1 to the correct position in the DRAM output buffer.
        // tile_start + i ensures each core writes to its own slice of tt_areas[].
        noc_async_write_tile(tile_start + i, out, rd_ptr);

        // Wait until the DMA write has left L1 and arrived in DRAM.
        // This is required before we can safely release the CB slot.
        noc_async_write_barrier();

        // Release the CB slot back to the compute kernel so it can be reused
        // for the next area tile.  Without this, the compute kernel's
        // cb_reserve_back(cb_area) would block indefinitely.
        cb_pop_front(cb_area, 1);
    }
}
