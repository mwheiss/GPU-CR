cmake_minimum_required(VERSION 3.18)
project(GPU-CR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# GPU vendor selection
# Usage:
#   cmake -DGPU_VENDOR=NVIDIA ..        (default)
#   cmake -DGPU_VENDOR=AMD ..
#   GPU_VENDOR=AMD cmake ..             (env override)
# ---------------------------------------------------------------------------
if(NOT DEFINED GPU_VENDOR)
    if(DEFINED ENV{GPU_VENDOR})
        set(GPU_VENDOR $ENV{GPU_VENDOR})
    else()
        set(GPU_VENDOR "NVIDIA")
    endif()
endif()

message(STATUS "Building for GPU vendor: ${GPU_VENDOR}")

if(GPU_VENDOR STREQUAL "NVIDIA")
    set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA installation directory")
    set(GPU_INCLUDE_DIR "${CUDA_ROOT}/include")
    set(GPU_LIB_DIR "${CUDA_ROOT}/lib64")
    set(GPU_LIBS cudart cuda)
elseif(GPU_VENDOR STREQUAL "AMD")
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm installation directory")
    set(GPU_INCLUDE_DIR "${ROCM_ROOT}/include")
    set(GPU_LIB_DIR "${ROCM_ROOT}/lib")
    set(GPU_LIBS amdhip64)
    add_compile_definitions(__HIP_PLATFORM_AMD__=1)
else()
    message(FATAL_ERROR "Unknown GPU_VENDOR: ${GPU_VENDOR}. Must be NVIDIA or AMD")
endif()

# ---------------------------------------------------------------------------
# Per-worker staging buffer size on hugepages (compile-time constant; consumed
# by vGPU.so and multi_cr_client via src/common.h). Increase when running
# vLLM/larger workloads at gpu_memory_utilization > ~0.5.
# Override with `cmake -DSHM_SIZE_GB=40 ..`
# ---------------------------------------------------------------------------
if(DEFINED SHM_SIZE_GB)
    add_compile_definitions(SHM_SIZE_GB=${SHM_SIZE_GB})
    message(STATUS "Per-worker SHM_SIZE_GB = ${SHM_SIZE_GB}")
endif()

set(STAGING_BUF_MB 256 CACHE STRING
    "Size in MiB of each of the two pinned GPU transfer bounce buffers")
add_compile_definitions(STAGING_BUF_MB=${STAGING_BUF_MB})
message(STATUS "Pinned staging buffer size = 2 x ${STAGING_BUF_MB} MiB")

# ---------------------------------------------------------------------------
# Optional adapter build flags
# ---------------------------------------------------------------------------
option(GPU_CR_BUILD_NCCL_ADAPTER    "Build the NCCL checkpoint adapter (plugin + runtime)" OFF)
option(GPU_CR_BUILD_NVSHMEM_ADAPTER "Build the NVSHMEM checkpoint examples"                OFF)

# ---------------------------------------------------------------------------
# Core LD_PRELOAD library: vGPU-${GPU_VENDOR}.so
# ---------------------------------------------------------------------------
file(GLOB_RECURSE VGPU_SOURCE_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Drop scratch sources.
list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "-pre\\.cpp$")

# Per-vendor backend exclusions.
if(GPU_VENDOR STREQUAL "NVIDIA")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/AMD/")
    message(STATUS "Excluding AMD backend sources")
elseif(GPU_VENDOR STREQUAL "AMD")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/NVIDIA/")
    # cuMem IPC + NCCL hooks are CUDA-specific; not built for AMD.
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/ipc_hooks\\.cpp$")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/ipc_fd_exchange\\.cpp$")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/nccl_hooks\\.cpp$")
    message(STATUS "Excluding NVIDIA backend sources and CUDA-only IPC/NCCL hooks")
endif()

file(GLOB_RECURSE VGPU_HEADER_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

add_library(vGPU SHARED ${VGPU_SOURCE_FILES} ${VGPU_HEADER_FILES})

set_target_properties(vGPU PROPERTIES
    OUTPUT_NAME "vGPU-${GPU_VENDOR}"
    PREFIX ""
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

target_include_directories(vGPU PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_directories(vGPU PRIVATE ${GPU_LIB_DIR})
target_link_libraries(vGPU PRIVATE ${GPU_LIBS} dl stdc++ pthread)
target_compile_options(vGPU PRIVATE -fPIC -O3 -g)
target_link_options(vGPU PRIVATE -shared -fPIC)

# ---------------------------------------------------------------------------
# Coordinator CLIs
#   cr_client       — single-GPU checkpoint trigger
#   multi_cr_client — multi-GPU phased orchestrator
# Both reuse src/comm/share_mem.cpp for their SHM control channel.
# ---------------------------------------------------------------------------
set(_COORD_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/comm/share_mem.cpp
)

add_executable(cr_client
    ${CMAKE_CURRENT_SOURCE_DIR}/coordinator/cr_client.cpp
    ${_COORD_COMMON_SOURCES}
)
target_include_directories(cr_client PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_directories(cr_client PRIVATE ${GPU_LIB_DIR})
target_link_libraries(cr_client PRIVATE ${GPU_LIBS} dl stdc++ pthread)
target_compile_options(cr_client PRIVATE -O3 -g)

add_executable(multi_cr_client
    ${CMAKE_CURRENT_SOURCE_DIR}/coordinator/multi_cr_client.cpp
    ${_COORD_COMMON_SOURCES}
)
target_include_directories(multi_cr_client PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_directories(multi_cr_client PRIVATE ${GPU_LIB_DIR})
target_link_libraries(multi_cr_client PRIVATE ${GPU_LIBS} dl stdc++ pthread)
target_compile_options(multi_cr_client PRIVATE -O3 -g)

# ---------------------------------------------------------------------------
# Optional adapters (only meaningful on NVIDIA)
# ---------------------------------------------------------------------------
if(GPU_CR_BUILD_NCCL_ADAPTER)
    if(NOT GPU_VENDOR STREQUAL "NVIDIA")
        message(FATAL_ERROR "GPU_CR_BUILD_NCCL_ADAPTER requires GPU_VENDOR=NVIDIA")
    endif()
    add_subdirectory(adapters/nccl)
endif()

if(GPU_CR_BUILD_NVSHMEM_ADAPTER)
    if(NOT GPU_VENDOR STREQUAL "NVIDIA")
        message(FATAL_ERROR "GPU_CR_BUILD_NVSHMEM_ADAPTER requires GPU_VENDOR=NVIDIA")
    endif()
    add_subdirectory(adapters/nvshmem)
endif()
