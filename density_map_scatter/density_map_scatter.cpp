// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Density Map Scatter — TT Metal SFPU vs exact DREAMPlace CPU reference
//
// CPU reference reproduces computeTriangleDensityMapLauncher exactly:
//   node_x       = x[i] + offset_x[i]          (offset = (sx_clamped−sx)/2 when stretched)
//   sx_clamped   = max(node_size_x[i], bin_size_x)   (DREAMPlace node_size_x_clamped)
//   bin_index_xl = int((node_x - xl)             * inv_bin_size_x)
//   bin_index_xh = int((node_x + sx_clamped - xl)* inv_bin_size_x) + 1
//   triangle_density_function(node_x, sx_clamped, xl, k, bin_size_x)
//       = min(node_x + sx_clamped, xl + k*bin_size_x + bin_size_x)
//         - max(node_x, xl + k*bin_size_x)        ← NO max(0,...) clamp
//   area = px * ratio * py
//   atomic_add(buf_map[k*num_bins_y + h], area)
//
// CMake builds:
//   metal_example_density_map_scatter                      — small regression (5k cells, 256², 1 core)
//   metal_example_density_map_scatter_benchmark            — 200k cells, 512², 1000² domain
//   metal_example_density_map_scatter_benchmark_large      — 500k cells, 2048², host px/py + SFPU multiply
//   metal_example_density_map_scatter_benchmark_large_geometry — same workload, geometry SFPU (BIN_SIZE on chip)

#include <fmt/ostream.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using namespace tt::tt_metal;


#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

#ifndef DMS_EXE_BENCHMARK
#define DMS_EXE_BENCHMARK 0
#endif
#ifndef DMS_EXE_BENCHMARK_LARGE
#define DMS_EXE_BENCHMARK_LARGE 0
#endif
#ifndef DMS_GEOMETRY_DEVICE
#define DMS_GEOMETRY_DEVICE 0
#endif

#if DMS_EXE_BENCHMARK_LARGE
// 2048² grid with cells [1,11): need bin_size large enough that impact window stays 10×10
// (1000² / 2048² would be 25×25 per cell → ~10 GB DRAM; 3000² keeps ~1.5 GB).
static constexpr int      NUM_CELLS            = 500000;
static constexpr int      NUM_BINS_X           = 2048;
static constexpr int      NUM_BINS_Y           = 2048;
static constexpr int      MAX_IMPACT_X         = 10;  // ceil(11 / (3000/2048)) + 2
static constexpr int      MAX_IMPACT_Y         = 10;
static constexpr uint32_t DMS_MAX_WORKER_CORES = 56;
static constexpr float    XL                   = 0.0f;
static constexpr float    YL                   = 0.0f;
static constexpr float    XH                   = 3000.0f;
static constexpr float    YH                   = 3000.0f;
#elif DMS_EXE_BENCHMARK
static constexpr int      NUM_CELLS            = 200000;
static constexpr int      NUM_BINS_X           = 512;
static constexpr int      NUM_BINS_Y           = 512;
static constexpr int      MAX_IMPACT_X         = 8;   // ceil(11/1.953)+2 = 8
static constexpr int      MAX_IMPACT_Y         = 8;
static constexpr uint32_t DMS_MAX_WORKER_CORES = 56;
static constexpr float    XL                   = 0.0f;
static constexpr float    YL                   = 0.0f;
static constexpr float    XH                   = 1000.0f;
static constexpr float    YH                   = 1000.0f;
#else
static constexpr int      NUM_CELLS            = 5000;
static constexpr int      NUM_BINS_X           = 256;
static constexpr int      NUM_BINS_Y           = 256;
static constexpr int      MAX_IMPACT_X         = 4;
static constexpr int      MAX_IMPACT_Y         = 4;
static constexpr uint32_t DMS_MAX_WORKER_CORES = 1;
static constexpr float    XL                   = 0.0f;
static constexpr float    YL                   = 0.0f;
static constexpr float    XH                   = 1000.0f;
static constexpr float    YH                   = 1000.0f;
#endif
static constexpr float BIN_SIZE_X   = (XH - XL) / static_cast<float>(NUM_BINS_X);
static constexpr float BIN_SIZE_Y   = (YH - YL) / static_cast<float>(NUM_BINS_Y);
static constexpr float INV_BIN_X    = 1.0f / BIN_SIZE_X;
static constexpr float INV_BIN_Y    = 1.0f / BIN_SIZE_Y;
static constexpr int   MAX_BINS_CELL = MAX_IMPACT_X * MAX_IMPACT_Y;

static constexpr uint32_t TILE_ELEMS = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
static constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(float);

// CPU reference for geometry mode clamps px, py to >= 0 to match device SFPU
// (DREAMPlace triangle_density_function can yield a negative px*py*ratio product).

int main() {
    bool pass = true;

    try {
        // ── 1. Generate synthetic movable cells ───────────────────────────────
        // node_size_x_clamped = max(node_size_x, bin_size_x); offset = (clamped−raw)/2
        // (same as DREAMPlace / benchmark_sfpu_density_map.py for apples-to-apples).
        auto t_gen0 = std::chrono::steady_clock::now();
        std::mt19937 rng(2025);
        std::uniform_real_distribution<float> pos_dist(XL, XH - 12.0f);
        std::uniform_real_distribution<float> size_dist(1.0f, 11.0f);

        struct Cell {
            float x, y;             // raw position (x_tensor, y_tensor in DREAMPlace)
            float sx, sy;           // raw size (node_size_x_tensor)
            float sx_clamped;       // max(sx, BIN_SIZE_X)  = node_size_x_clamped_tensor
            float sy_clamped;       // max(sy, BIN_SIZE_Y)
            float offset_x;         // (sx_clamped − sx) / 2
            float offset_y;
            float ratio;            // ratio_tensor
        };
        std::vector<Cell> cells(NUM_CELLS);
        for (auto& c : cells) {
            c.x         = pos_dist(rng);
            c.y         = pos_dist(rng);
            c.sx        = size_dist(rng);
            c.sy        = size_dist(rng);
            c.sx_clamped = std::max(c.sx, BIN_SIZE_X);   // DREAMPlace clamping
            c.sy_clamped = std::max(c.sy, BIN_SIZE_Y);
            c.offset_x   = (c.sx_clamped - c.sx) * 0.5f;
            c.offset_y   = (c.sy_clamped - c.sy) * 0.5f;
            c.ratio     = 1.0f;
        }

        // ── 2. Enumerate (cell, bin) tasks — matching DREAMPlace bin-range ────
        // DREAMPlace launcher lines 277-291:
        //   node_x       = x[i] + offset_x[i]
        //   node_size_x  = node_size_x_clamped[i]
        //   bin_index_xl = int((node_x - xl) * inv_bin_size_x)
        //   bin_index_xh = int((node_x + node_size_x - xl) * inv_bin_size_x) + 1
        const int raw_tasks = NUM_CELLS * MAX_BINS_CELL;
        const int num_tiles =
            (raw_tasks + static_cast<int>(TILE_ELEMS) - 1) / static_cast<int>(TILE_ELEMS);
        const int num_tasks = num_tiles * static_cast<int>(TILE_ELEMS);

#if DMS_GEOMETRY_DEVICE
        // Geometry TT path: 6 inputs (cxl,cxr,bxl,cyl,cyr,byl); device adds BIN_SIZE for bxh/byh.
        // ratio is always 1.0 in this benchmark, so it is hardcoded in the kernel (saves 200 MB upload).
        std::vector<float> t_cxl(num_tasks, 0.0f);
        std::vector<float> t_cxr(num_tasks, 0.0f);
        std::vector<float> t_bxl(num_tasks, 0.0f);
        std::vector<float> t_cyl(num_tasks, 0.0f);
        std::vector<float> t_cyr(num_tasks, 0.0f);
        std::vector<float> t_byl(num_tasks, 0.0f);
#else
        // Pre-multiply path: host px/py; device area = px * ratio * py only.
        std::vector<float> t_px   (num_tasks, 0.0f);
        std::vector<float> t_py   (num_tasks, 0.0f);
        std::vector<float> t_ratio(num_tasks, 0.0f);
#endif
        std::vector<int>   t_bin_k(num_tasks, 0);
        std::vector<int>   t_bin_h(num_tasks, 0);
        std::vector<bool>  t_valid(num_tasks, false);

        for (int ci = 0; ci < NUM_CELLS; ci++) {
            const Cell& c  = cells[ci];
            const float nx = c.x + c.offset_x;
            const float ny = c.y + c.offset_y;
            const float nxr = nx + c.sx_clamped;
            const float nyr = ny + c.sy_clamped;

            // Exact DREAMPlace bin-range (truncation toward zero = floor for >=0)
            int bxl_idx = std::max(0, static_cast<int>((nx  - XL) * INV_BIN_X));
            int bxh_idx = std::min(NUM_BINS_X, static_cast<int>((nxr - XL) * INV_BIN_X) + 1);
            int byl_idx = std::max(0, static_cast<int>((ny  - YL) * INV_BIN_Y));
            int byh_idx = std::min(NUM_BINS_Y, static_cast<int>((nyr - YL) * INV_BIN_Y) + 1);

            int local_k = 0;
            for (int k = bxl_idx; k < bxh_idx && local_k < MAX_IMPACT_X; k++, local_k++) {
                int local_h = 0;
                for (int h = byl_idx; h < byh_idx && local_h < MAX_IMPACT_Y; h++, local_h++) {
                    int base = ci * MAX_BINS_CELL + local_k * MAX_IMPACT_Y + local_h;
                    const float bxl = XL + static_cast<float>(k) * BIN_SIZE_X;
                    const float byl = YL + static_cast<float>(h) * BIN_SIZE_Y;
#if DMS_GEOMETRY_DEVICE
                    t_cxl[base] = nx;
                    t_cxr[base] = nxr;
                    t_bxl[base] = bxl;
                    t_cyl[base] = ny;
                    t_cyr[base] = nyr;
                    t_byl[base] = byl;
                    // ratio = 1.0 always; hardcoded in kernel — no buffer needed
#else
                    const float bxh = bxl + BIN_SIZE_X;
                    const float byh = byl + BIN_SIZE_Y;
                    t_px   [base] = std::max(0.0f, std::min(nxr, bxh) - std::max(nx,  bxl));
                    t_py   [base] = std::max(0.0f, std::min(nyr, byh) - std::max(ny,  byl));
                    t_ratio[base] = c.ratio;
#endif
                    t_bin_k[base] = k;
                    t_bin_h[base] = h;
                    t_valid[base] = true;
                }
            }
        }
        auto t_gen1 = std::chrono::steady_clock::now();

        int total_valid_tasks = 0;
        for (int i = 0; i < NUM_CELLS * MAX_BINS_CELL; i++) {
            if (t_valid[i]) total_valid_tasks++;
        }

        // ── 3. CPU reference — exact DREAMPlace serial path (for accuracy) ────
        // Serial (deterministic) for accuracy comparison; no float re-ordering.
        auto t_cpu_serial0 = std::chrono::steady_clock::now();
        std::vector<float> ref_density(NUM_BINS_X * NUM_BINS_Y, 0.0f);
        std::vector<float> ref_area   (num_tasks, 0.0f);
        for (int i = 0; i < NUM_CELLS * MAX_BINS_CELL; i++) {
            if (!t_valid[i]) continue;
#if DMS_GEOMETRY_DEVICE
            // Match device SFPU: clamp px, py to >= 0 (dreamplace_triangle_area can be negative).
            float px = std::min(t_cxr[i], t_bxl[i] + static_cast<float>(BIN_SIZE_X))
                     - std::max(t_cxl[i], t_bxl[i]);
            float py = std::min(t_cyr[i], t_byl[i] + static_cast<float>(BIN_SIZE_Y))
                     - std::max(t_cyl[i], t_byl[i]);
            ref_area[i] = std::max(0.f, px) * std::max(0.f, py);  // ratio = 1.0 always
#else
            ref_area[i] = t_px[i] * t_ratio[i] * t_py[i];
#endif
            ref_density[t_bin_k[i] * NUM_BINS_Y + t_bin_h[i]] += ref_area[i];
        }
        auto t_cpu_serial1 = std::chrono::steady_clock::now();

        fmt::print(
            "Setup  : {} cells, {} valid tasks ({} tiles) on {}×{} grid\n",
            NUM_CELLS, total_valid_tasks, num_tiles, NUM_BINS_X, NUM_BINS_Y);
        fmt::print(
            "         bin_size={:.6f}×{:.6f}  sx_clamped applied ({} cells had sx<bin_size)\n",
            BIN_SIZE_X, BIN_SIZE_Y,
            [&]{ int n=0; for(auto& c:cells) if(c.sx<BIN_SIZE_X||c.sy<BIN_SIZE_Y) n++; return n; }());
        fmt::print(
            "Host   : generate+enumerate {:.3f} ms\n",
            std::chrono::duration<double, std::milli>(t_gen1 - t_gen0).count());
        // NOTE: CPU ref here is serial (single-threaded). For exact DREAMPlace OMP timing
        // use scripts/benchmark_sfpu_density_map.py which calls ElectricDensityMapFunction.forward
        // directly — that already uses OMP compiled into the DREAMPlace C++ extension.
        fmt::print(
            "CPU ref: serial exact triangle formula {:.3f} ms  (see Python script for OMP timing)\n",
            std::chrono::duration<double, std::milli>(t_cpu_serial1 - t_cpu_serial0).count());
#if DMS_GEOMETRY_DEVICE
        fmt::print(
            "TT path : geometry-dependent SFPU (6 DRAM inputs; ratio=1.0 hardcoded; BIN_SIZE from JIT defines)\n");
        fmt::print(
            "         host BIN_SIZE_X=Y={:.9f} (passed to JIT as DMS_BIN_SIZE_*_F + int domain/bins)\n",
            BIN_SIZE_X);
#else
        fmt::print(
            "TT path : host px/py precompute + SFPU multiply (3 DRAM inputs)\n");
#endif

        // ── 5. Device setup ───────────────────────────────────────────────────
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        const uint32_t dram_size = static_cast<uint32_t>(num_tiles) * TILE_BYTES;
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size   = TILE_BYTES,
            .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig dram_rep{.size = dram_size};

#if DMS_GEOMETRY_DEVICE
        // 6 input buffers: ratio is hardcoded 1.0 in the kernel — no DRAM allocation needed.
        auto buf_cxl  = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_cxr  = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_bxl  = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_cyl  = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_cyr  = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_byl  = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_area = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());

#else
        auto buf_px    = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_py    = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_ratio = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
        auto buf_area  = distributed::MeshBuffer::create(dram_rep, dram_cfg, mesh_device.get());
#endif

        // ── 6. Multi-core program setup ───────────────────────────────────────
        CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
        const uint32_t grid_cores   = static_cast<uint32_t>(compute_grid.x) * static_cast<uint32_t>(compute_grid.y);
        const uint32_t worker_cores = std::min(DMS_MAX_WORKER_CORES, std::max(1u, grid_cores));

        CoreRangeSet active_workers;
        if (worker_cores == 1) {
            active_workers = CoreRangeSet(CoreRange({0, 0}, {0, 0}));
        } else {
            active_workers = num_cores_to_corerangeset(worker_cores, compute_grid, true);
        }

        auto [num_cores_used, all_cores, core_group_1, core_group_2, work1, work2] =
            split_work_to_cores(active_workers, static_cast<uint32_t>(num_tiles), true);

        fmt::print(
            "Device : compute grid {}×{}  using {} Tensix core(s) for {} tiles "
            "(work_per_core {} / {})\n",
            compute_grid.x, compute_grid.y, num_cores_used, num_tiles, work1, work2);

        Program program = CreateProgram();
        constexpr uint32_t TILES_PER_CB  = 2;
        const uint32_t     cb_bytes      = TILES_PER_CB * TILE_BYTES;

        auto make_fp32_cb = [&](tt::CBIndex idx) {
            CreateCircularBuffer(
                program, all_cores,
                CircularBufferConfig(cb_bytes, {{idx, tt::DataFormat::Float32}})
                    .set_page_size(idx, TILE_BYTES));
        };
#if DMS_GEOMETRY_DEVICE
        make_fp32_cb(tt::CBIndex::c_0);   // cxl
        make_fp32_cb(tt::CBIndex::c_1);   // cxr
        make_fp32_cb(tt::CBIndex::c_2);   // bxl
        make_fp32_cb(tt::CBIndex::c_3);   // cyl
        make_fp32_cb(tt::CBIndex::c_4);   // cyr
        make_fp32_cb(tt::CBIndex::c_5);   // byl
        // c_6 (ratio) removed — hardcoded 1.0 in kernel
        make_fp32_cb(tt::CBIndex::c_16);  // area (output)

        std::vector<uint32_t> reader_ct;
        TensorAccessorArgs(*buf_cxl->get_backing_buffer()).append_to(reader_ct);
        TensorAccessorArgs(*buf_cxr->get_backing_buffer()).append_to(reader_ct);
        TensorAccessorArgs(*buf_bxl->get_backing_buffer()).append_to(reader_ct);
        TensorAccessorArgs(*buf_cyl->get_backing_buffer()).append_to(reader_ct);
        TensorAccessorArgs(*buf_cyr->get_backing_buffer()).append_to(reader_ct);
        TensorAccessorArgs(*buf_byl->get_backing_buffer()).append_to(reader_ct);

        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "density_map_scatter/kernels/dataflow/read_tiles_geometry.cpp",
            all_cores,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_0,
                .noc          = NOC::RISCV_0_default,
                .compile_args = reader_ct});
#else
        make_fp32_cb(tt::CBIndex::c_0);   // px
        make_fp32_cb(tt::CBIndex::c_1);   // py
        make_fp32_cb(tt::CBIndex::c_2);   // ratio
        make_fp32_cb(tt::CBIndex::c_16);  // area (output)

        std::vector<uint32_t> reader_ct;
        TensorAccessorArgs(*buf_px->get_backing_buffer()).append_to(reader_ct);
        TensorAccessorArgs(*buf_py->get_backing_buffer()).append_to(reader_ct);
        TensorAccessorArgs(*buf_ratio->get_backing_buffer()).append_to(reader_ct);

        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "density_map_scatter/kernels/dataflow/read_tiles.cpp",
            all_cores,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_0,
                .noc          = NOC::RISCV_0_default,
                .compile_args = reader_ct});
#endif

        std::vector<uint32_t> writer_ct;
        TensorAccessorArgs(*buf_area->get_backing_buffer()).append_to(writer_ct);

        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "density_map_scatter/kernels/dataflow/write_tile.cpp",
            all_cores,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_1,
                .noc          = NOC::RISCV_1_default,
                .compile_args = writer_ct});

#if DMS_GEOMETRY_DEVICE
        // JIT defines must match host XL..XH / grid — same as CMake would NOT suffice alone.
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "density_map_scatter/kernels/compute/overlap_compute_geometry.cpp",
            all_cores,
            ComputeConfig{
                .fp32_dest_acc_en = true,
                .math_approx_mode = false,
                .defines = {
                    {"DMS_NUM_BINS_KERNEL", std::to_string(NUM_BINS_X)},
                    {"DMS_DOMAIN_WIDTH",    std::to_string(static_cast<int>(XH - XL))},
                    {"DMS_DOMAIN_HEIGHT",   std::to_string(static_cast<int>(YH - YL))},
                    // Same float tokens as host BIN_SIZE_X/Y — avoids any int/FP divide mismatch on device.
                    {"DMS_BIN_SIZE_X_F", fmt::format("{:.9g}f", BIN_SIZE_X)},
                    {"DMS_BIN_SIZE_Y_F", fmt::format("{:.9g}f", BIN_SIZE_Y)},
                }});
#else
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "density_map_scatter/kernels/compute/overlap_compute.cpp",
            all_cores,
            ComputeConfig{
                .fp32_dest_acc_en = true,
                .math_approx_mode = false});
#endif

        uint32_t tile_start = 0;
        const auto work_groups = {
            std::make_pair(core_group_1, work1),
            std::make_pair(core_group_2, work2),
        };
        for (const auto& [group, wpc] : work_groups) {
            for (const auto& range : group.ranges()) {
                for (const auto& core : range) {
#if DMS_GEOMETRY_DEVICE
                    SetRuntimeArgs(program, reader, core,
                        {buf_cxl->address(), buf_cxr->address(), buf_bxl->address(),
                         buf_cyl->address(), buf_cyr->address(), buf_byl->address(),
                         wpc, tile_start});
#else
                    SetRuntimeArgs(program, reader, core,
                        {buf_px->address(), buf_py->address(), buf_ratio->address(),
                         wpc, tile_start});
#endif
                    SetRuntimeArgs(program, writer, core,
                        {buf_area->address(), wpc, tile_start});
                    SetRuntimeArgs(program, compute, core, {wpc});
                    tile_start += wpc;
                }
            }
        }

        distributed::MeshWorkload workload;
        workload.add_program(
            distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

        // ── Warmup + measurement runs ────────────────────────────────────────
        // 2 warmup runs (kernel cache warm, DRAM TLBs live) then 1 timed run.
        static constexpr int WARMUP_RUNS    = 2;
        static constexpr int MEASURED_RUNS  = 1;
        static constexpr int TOTAL_RUNS     = WARMUP_RUNS + MEASURED_RUNS;

        double upload_ms = 0.0, kernel_ms = 0.0, readback_ms = 0.0;
        std::vector<float> tt_areas;

        for (int run = 0; run < TOTAL_RUNS; ++run) {
            const bool is_warmup  = (run < WARMUP_RUNS);
            const char* run_label = is_warmup ? "warmup" : "MEASURED";

            // Re-upload inputs each run (simulates steady-state workload).
            auto t_up0 = std::chrono::steady_clock::now();
#if DMS_GEOMETRY_DEVICE
            // 6 buffers: ratio dropped (hardcoded 1.0 in kernel) → saves 200 MB / run
            distributed::EnqueueWriteMeshBuffer(cq, buf_cxl, t_cxl, false);
            distributed::EnqueueWriteMeshBuffer(cq, buf_cxr, t_cxr, false);
            distributed::EnqueueWriteMeshBuffer(cq, buf_bxl, t_bxl, false);
            distributed::EnqueueWriteMeshBuffer(cq, buf_cyl, t_cyl, false);
            distributed::EnqueueWriteMeshBuffer(cq, buf_cyr, t_cyr, false);
            distributed::EnqueueWriteMeshBuffer(cq, buf_byl, t_byl, false);
#else
            distributed::EnqueueWriteMeshBuffer(cq, buf_px,    t_px,    false);
            distributed::EnqueueWriteMeshBuffer(cq, buf_py,    t_py,    false);
            distributed::EnqueueWriteMeshBuffer(cq, buf_ratio, t_ratio, false);
#endif
            distributed::Finish(cq);
            auto t_up1 = std::chrono::steady_clock::now();

            auto t_dev0 = std::chrono::steady_clock::now();
            distributed::EnqueueMeshWorkload(cq, workload, false);
            distributed::Finish(cq);
            auto t_dev1 = std::chrono::steady_clock::now();

            auto t_dn0 = std::chrono::steady_clock::now();
            tt_areas.clear();
            distributed::EnqueueReadMeshBuffer(cq, tt_areas, buf_area, true);
            auto t_dn1 = std::chrono::steady_clock::now();

            double run_upload_ms   = std::chrono::duration<double, std::milli>(t_up1  - t_up0).count();
            double run_kernel_ms   = std::chrono::duration<double, std::milli>(t_dev1 - t_dev0).count();
            double run_readback_ms = std::chrono::duration<double, std::milli>(t_dn1  - t_dn0).count();

            fmt::print(
                "Run {:2d} ({:8s}) | upload {:.3f} ms | kernel {:.3f} ms | readback {:.3f} ms\n",
                run, run_label, run_upload_ms, run_kernel_ms, run_readback_ms);

            if (!is_warmup) {
                upload_ms   = run_upload_ms;
                kernel_ms   = run_kernel_ms;
                readback_ms = run_readback_ms;
            }
        }

        const double readback_gb = static_cast<double>(dram_size) / 1e9;
#if DMS_GEOMETRY_DEVICE
        const double upload_gb = (6.0 * static_cast<double>(dram_size)) / 1e9;
        fmt::print("\n=== Geometry kernel timing (post-{}-warmup measured run) ===\n", WARMUP_RUNS);
        fmt::print(
            "  DRAM upload    : {:.3f} ms  ({:.3f} GB, 6 buffers)  [{:.2f} GB/s]\n"
            "  Device kernel  : {:.3f} ms  ← host wall-clock (EnqueueMeshWorkload + Finish)\n"
            "  DRAM readback  : {:.3f} ms  ({:.3f} GB)  [{:.2f} GB/s]\n"
            "  Total          : {:.3f} ms\n",
            upload_ms,   upload_gb,   upload_gb   / (upload_ms   / 1000.0),
            kernel_ms,
            readback_ms, readback_gb, readback_gb / (readback_ms / 1000.0),
            upload_ms + kernel_ms + readback_ms);
#else
        const double upload_gb = (3.0 * static_cast<double>(dram_size)) / 1e9;
        fmt::print("\n=== Pre-multiply kernel timing (post-{}-warmup measured run) ===\n", WARMUP_RUNS);
        fmt::print(
            "  DRAM upload    : {:.3f} ms  ({:.3f} GB, 3 buffers)  [{:.2f} GB/s]\n"
            "  Device kernel  : {:.3f} ms  ← host wall-clock (EnqueueMeshWorkload + Finish)\n"
            "  DRAM readback  : {:.3f} ms  ({:.3f} GB)  [{:.2f} GB/s]\n"
            "  Total          : {:.3f} ms\n",
            upload_ms,   upload_gb,   upload_gb   / (upload_ms   / 1000.0),
            kernel_ms,
            readback_ms, readback_gb, readback_gb / (readback_ms / 1000.0),
            upload_ms + kernel_ms + readback_ms);
#endif

        // ── 7. Runtime print ──────────────────────────────────────────────────
        double dev_ms = kernel_ms;
        fmt::print(
            "Device : kernel ({} Tensix cores) {:.3f} ms\n",
            num_cores_used, dev_ms);

        // ── 8. Per-task accuracy vs exact DREAMPlace serial reference ─────────
        TT_FATAL(
            static_cast<int>(tt_areas.size()) == num_tasks,
            "Area buffer size mismatch: got {} expected {}", tt_areas.size(), num_tasks);

        constexpr float STRICT_EPS    = 5e-4f;
        size_t          strict_mismatch = 0;
        float           max_area_err    = 0.0f;
        const int sample_bad =
            (DMS_EXE_BENCHMARK || DMS_EXE_BENCHMARK_LARGE) ? 3 : 8;

        for (int i = 0; i < NUM_CELLS * MAX_BINS_CELL; i++) {
            if (!t_valid[i]) continue;
            const float expected = ref_area[i];
            const float actual   = tt_areas[i];
            const float err      = std::abs(expected - actual);
            max_area_err = std::max(max_area_err, err);
            if (err > STRICT_EPS + STRICT_EPS * std::abs(expected)) {
                if (strict_mismatch < static_cast<size_t>(sample_bad)) {
                    fmt::print(
                        stderr,
                        "  Task {:7d}: expected={:.6f}  tt={:.6f}  err={:.2e}\n",
                        i, expected, actual, err);
                }
                ++strict_mismatch;
            }
        }
        fmt::print(
            "Per-task (vs exact DREAMPlace triangle formula): "
            "{} mismatches / {} valid  max|err|={:.4e}\n",
            strict_mismatch, total_valid_tasks, max_area_err);

        // ── 9. Scatter TT areas → density map; compare to serial CPU ref ──────
        std::vector<float> tt_density(NUM_BINS_X * NUM_BINS_Y, 0.0f);
        for (int i = 0; i < NUM_CELLS * MAX_BINS_CELL; i++) {
            if (!t_valid[i]) continue;
            tt_density[t_bin_k[i] * NUM_BINS_Y + t_bin_h[i]] += tt_areas[i];
        }

        float  total_tt = 0.0f, total_ref = 0.0f;
        float  max_abs_bin = 0.0f;
        double sum_sq_err = 0.0, sum_sq_ref = 0.0;
        int    nonzero = 0;
        const int num_bins = NUM_BINS_X * NUM_BINS_Y;
        for (int i = 0; i < num_bins; i++) {
            total_tt  += tt_density[i];
            total_ref += ref_density[i];
            double d   = static_cast<double>(tt_density[i]) - static_cast<double>(ref_density[i]);
            max_abs_bin = std::max(max_abs_bin, static_cast<float>(std::abs(d)));
            sum_sq_err += d * d;
            sum_sq_ref += static_cast<double>(ref_density[i]) * static_cast<double>(ref_density[i]);
            if (tt_density[i] > 0.0f) nonzero++;
        }
        const double rms_err   = std::sqrt(sum_sq_err / static_cast<double>(num_bins));
        const double rel_l2    = (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_err) / std::sqrt(sum_sq_ref) : 0.0;
        const float  rel_total = std::abs(total_tt - total_ref) / std::max(total_ref, 1.0f);

        fmt::print("=== Accuracy: TT density map vs exact DREAMPlace CPU reference ===\n");
        fmt::print("  Non-zero bins   : {}/{}\n", nonzero, num_bins);
        fmt::print("  Total area      : TT={:.4f}  CPU={:.4f}  rel_err={:.4e}\n",
                   total_tt, total_ref, rel_total);
        fmt::print("  Max |bin delta| : {:.4e}\n", max_abs_bin);
        fmt::print("  RMS bin delta   : {:.4e}  (absolute)\n", rms_err);
        fmt::print("  Relative L2     : {:.4e}  (‖TT−ref‖₂ / ‖ref‖₂)\n", rel_l2);

        // Pass gates
        constexpr float  MAX_REL_TOTAL = 2.0e-3;  // 0.2%  integrated area
        constexpr double MAX_REL_L2    = 5.0e-2;  // 5%    SFPU FP32 vs scalar CPU
        if (rel_total > MAX_REL_TOTAL || rel_l2 > MAX_REL_L2) {
            fmt::print(
                stderr,
                "FAIL: rel_total={:.4e} (max {:.1e})  rel_L2={:.4e} (max {:.1e})\n",
                rel_total, static_cast<double>(MAX_REL_TOTAL), rel_l2, MAX_REL_L2);
            pass = false;
        }

        if (!mesh_device->close()) pass = false;

    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception: {}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }
    return 0;
}
