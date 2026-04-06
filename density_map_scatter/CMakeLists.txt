cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_density_map_scatter)

add_executable(metal_example_density_map_scatter)
target_sources(metal_example_density_map_scatter PRIVATE density_map_scatter.cpp)
target_compile_definitions(
    metal_example_density_map_scatter
    PRIVATE
        DMS_EXE_BENCHMARK=0)

add_executable(metal_example_density_map_scatter_benchmark)
target_sources(metal_example_density_map_scatter_benchmark PRIVATE density_map_scatter.cpp)
target_compile_definitions(
    metal_example_density_map_scatter_benchmark
    PRIVATE
        DMS_EXE_BENCHMARK=1)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_density_map_scatter PUBLIC TT::Metalium)
target_link_libraries(metal_example_density_map_scatter_benchmark PUBLIC TT::Metalium)

add_executable(metal_example_density_map_scatter_benchmark_large)
target_sources(metal_example_density_map_scatter_benchmark_large PRIVATE density_map_scatter.cpp)
target_compile_definitions(
    metal_example_density_map_scatter_benchmark_large
    PRIVATE
        DMS_EXE_BENCHMARK_LARGE=1)
target_link_libraries(metal_example_density_map_scatter_benchmark_large PUBLIC TT::Metalium)

add_executable(metal_example_density_map_scatter_benchmark_large_geometry)
target_sources(
    metal_example_density_map_scatter_benchmark_large_geometry
    PRIVATE
        density_map_scatter.cpp)
target_compile_definitions(
    metal_example_density_map_scatter_benchmark_large_geometry
    PRIVATE
        DMS_EXE_BENCHMARK_LARGE=1
        DMS_GEOMETRY_DEVICE=1)
target_link_libraries(
    metal_example_density_map_scatter_benchmark_large_geometry
    PUBLIC
        TT::Metalium)

if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_density_map_scatter TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_density_map_scatter_benchmark TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_density_map_scatter_benchmark_large TT::CommonPCH)
    tt_reuse_precompile_headers(
        metal_example_density_map_scatter_benchmark_large_geometry
        TT::CommonPCH)
endif()

