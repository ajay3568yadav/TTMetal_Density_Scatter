// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ─────────────────────────────────────────────────────────────────────────────
// read_tiles.cpp  —  READER KERNEL, PRE-MULTIPLY PATH
// ─────────────────────────────────────────────────────────────────────────────
//
// PURPOSE
// -------
// This is the data-movement (RISCV_0 / TRISC_UNPACK) kernel for the pre-multiply
// variant of the density map scatter benchmark.
//
// It reads 3 DRAM input buffers tile-by-tile and pushes each tile into its
// corresponding Circular Buffer (CB), where the compute kernel (overlap_compute.cpp)
// will pick it up via copy_tile.
//
// INPUTS (from DRAM, flat tile-addressed buffers):
//   px    — x-overlap per task (pre-computed by host, FP32, clamped ≥ 0)
//   py    — y-overlap per task (pre-computed by host, FP32, clamped ≥ 0)
//   ratio — per-cell scaling factor (= 1.0 in this benchmark)
//
// Each input is a flat 1D array of FP32 tiles indexed by global tile number.
// Multiple Tensix cores share the same DRAM buffers; each core reads a
// non-overlapping slice [tile_start, tile_start + n_tiles).
//
// ─────────────────────────────────────────────────────────────────────────────
// TT METAL DATA MOVEMENT PRIMER
// ─────────────────────────────────────────────────────────────────────────────
//
// CIRCULAR BUFFERS (CBs):
//   On-chip L1 SRAM ring buffers.  The reader fills them; the compute kernel
//   drains them.  cb_reserve_back() blocks until a free slot is available.
//   cb_push_back() marks the slot as filled and wakes the consumer.
//
// NOC (Network-on-Chip):
//   The path from DRAM to L1.  noc_async_read_tile() issues a non-blocking
//   DMA transfer.  noc_async_read_barrier() waits until ALL outstanding
//   async reads have landed in L1.  Only then is it safe to push to the CB.
//
// TensorAccessor / TensorAccessorArgs:
//   A compile-time + runtime pair that encodes how to compute the DRAM address
//   of tile number g.  "Compile-time args" are encoded into the RISC-V binary
//   at JIT time (stride, base, interleave mode, etc.).  TensorAccessorArgs<N>
//   is a counter that chains multiple accessors without index collisions.
//   The chaining ensures accessor for px doesn't use the same compile-time
//   arg slots as the accessor for py or ratio.
//
// TILE SIZE:
//   Each FP32 tile is 32×32 = 1024 float32 values = 4096 bytes.
//   get_tile_size(cb) returns this in bytes; all CBs in one kernel use the same
//   tile size.
//
// ─────────────────────────────────────────────────────────────────────────────
// RUNTIME ARGUMENTS (set by host in density_map_scatter.cpp):
//   [0] addr_px    — DRAM base address of the px buffer
//   [1] addr_py    — DRAM base address of the py buffer
//   [2] addr_ratio — DRAM base address of the ratio buffer
//   [3] n_tiles    — number of tiles this core must read  (= work_per_core)
//   [4] tile_start — global tile index of this core's first tile
//
// COMPILE-TIME ARGUMENTS (encoded in the JIT binary via TensorAccessorArgs):
//   Chained for px, py, ratio — controls DRAM addressing (interleave, stride).
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>

void kernel_main() {
    // ── Runtime arguments from the host ──────────────────────────────────────
    const uint32_t addr_px    = get_arg_val<uint32_t>(0);  // DRAM base of px buffer
    const uint32_t addr_py    = get_arg_val<uint32_t>(1);  // DRAM base of py buffer
    const uint32_t addr_ratio = get_arg_val<uint32_t>(2);  // DRAM base of ratio buffer
    const uint32_t n_tiles    = get_arg_val<uint32_t>(3);  // tiles to process on this core
    const uint32_t tile_start = get_arg_val<uint32_t>(4);  // first global tile index

    // ── Circular buffer indices (must match compute kernel and host setup) ────
    constexpr uint32_t cb_px    = tt::CBIndex::c_0;
    constexpr uint32_t cb_py    = tt::CBIndex::c_1;
    constexpr uint32_t cb_ratio = tt::CBIndex::c_2;

    // tile_bytes: number of bytes per FP32 tile (32×32×4 = 4096).
    // Used by TensorAccessor to stride correctly through the DRAM buffer.
    const uint32_t tile_bytes = get_tile_size(cb_px);

    // ── Build TensorAccessors (compile-time chaining) ─────────────────────────
    // TensorAccessorArgs<0> starts counting compile-time arg slots from 0.
    // Each accessor for (px, py, ratio) is given a non-overlapping slot range.
    // next_compile_time_args_offset() returns the first free slot after a0.
    constexpr auto a0 = TensorAccessorArgs<0>();
    const auto in_px    = TensorAccessor(a0, addr_px,    tile_bytes);

    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const auto in_py    = TensorAccessor(a1, addr_py,    tile_bytes);

    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    const auto in_ratio = TensorAccessor(a2, addr_ratio, tile_bytes);

    // ── Main loop: one iteration = one tile for all three inputs ─────────────
    for (uint32_t i = 0; i < n_tiles; i++) {

        // Reserve one write slot in each CB.
        // Blocks if the compute kernel hasn't consumed the previous tile yet.
        cb_reserve_back(cb_px,    1);
        cb_reserve_back(cb_py,    1);
        cb_reserve_back(cb_ratio, 1);

        // get_write_ptr returns the L1 address of the reserved CB slot.
        // This is the destination for the NOC DMA transfer.
        const uint32_t ptr_px    = get_write_ptr(cb_px);
        const uint32_t ptr_py    = get_write_ptr(cb_py);
        const uint32_t ptr_ratio = get_write_ptr(cb_ratio);

        // g = global tile index — each core reads a different slice of DRAM.
        // tile_start ensures no two cores read the same tile.
        const uint32_t g = tile_start + i;

        // Issue three non-blocking DMA reads in parallel.
        // The NOC can overlap multiple transfers from the same or different DRAM banks.
        noc_async_read_tile(g, in_px,    ptr_px);
        noc_async_read_tile(g, in_py,    ptr_py);
        noc_async_read_tile(g, in_ratio, ptr_ratio);

        // Wait until ALL three DMA transfers have completed into L1.
        // Without this barrier, the data at ptr_* may not be valid yet when we push.
        noc_async_read_barrier();

        // Mark the three CB slots as filled.
        // The compute kernel's cb_wait_front() calls will unblock after this.
        cb_push_back(cb_px,    1);
        cb_push_back(cb_py,    1);
        cb_push_back(cb_ratio, 1);
    }
}
