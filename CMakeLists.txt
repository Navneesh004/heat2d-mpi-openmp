cmake_minimum_required(VERSION 3.16)
project(heat2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HEAT_NATIVE     "Compile with -march=native"                        ON)
option(HEAT_VEC_REPORT "Print compiler vectorisation report (-fopt-info-vec)" OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(heat
  src/main.cpp
  src/kernel.cpp
  src/halo.cpp
  src/io.cpp)

target_link_libraries(heat PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(heat PRIVATE
  -Wall -Wextra
  $<$<CONFIG:Release>:-O3>
  # No FMA contraction: keeps every decomposition bit-identical to the serial
  # reference (an FMA rounds differently than a separate multiply + add).
  -ffp-contract=off
  # Debug info even in Release so `perf report` can show source lines.
  -g)
if(HEAT_NATIVE)
  target_compile_options(heat PRIVATE -march=native)
endif()
if(HEAT_VEC_REPORT)
  target_compile_options(heat PRIVATE -fopt-info-vec-optimized -fopt-info-vec-missed=vec-missed.txt)
endif()

# ---------------------------------------------------------------- tests ---
# Every configuration must reproduce the serial reference bit-for-bit.
enable_testing()
set(MPI_TEST_FLAGS ${MPIEXEC_PREFLAGS} --oversubscribe)

function(heat_test name np threads)
  add_test(NAME ${name}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPI_TEST_FLAGS}
                   $<TARGET_FILE:heat> ${ARGN} --verify)
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${threads}")
endfunction()

# odd sizes on purpose: uneven row split, rank-0 owning only boundary rows, ...
heat_test(serial          1 1 61 200 --rows 67)
heat_test(openmp_t4       1 4 61 200 --rows 67)
heat_test(mpi_np2         2 1 61 200 --rows 67)
heat_test(mpi_np3         3 1 61 200 --rows 67)
heat_test(mpi_np4         4 1 61 200 --rows 67)
heat_test(mpi_np5_tiny    5 1 20 100 --rows 7)
heat_test(hybrid_2x2      2 2 61 200 --rows 67)
heat_test(blocked_np3     3 2 61 200 --rows 67 --block 16)
heat_test(overlap_np4     4 1 61 200 --rows 67 --overlap)
heat_test(overlap_tiny    4 2 20 100 --rows 6 --overlap --block 8)
