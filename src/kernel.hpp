// kernel.hpp - the 5-point Jacobi stencil for 2D heat diffusion.
#pragma once

#include <vector>

// Update rows [r0, r1) of a row-major array with row stride nx, reading u and
// writing unew. Columns 1..nx-2 are updated; columns 0 and nx-1 are left alone
// (fixed boundary). Rows r0-1 and r1 must be valid (real or ghost rows).
//
// block <= 0 : plain row-by-row sweep (OpenMP over rows, SIMD over columns)
// block  > 0 : cache-blocked sweep - each thread walks its rows in column
//              tiles of `block` doubles so the 3-row working set stays in L1.
//
// Both variants do exactly the same floating-point operations in the same
// order per cell, so they produce bit-identical results.
void stencil_rows(const double* u, double* unew, int nx, int r0, int r1, double alpha, int block);

// Independent, deliberately simple single-threaded solver on the full global
// grid (ny x nx, no ghost rows). Used as the reference for --verify.
void reference_solve(std::vector<double>& grid, int ny, int nx, int steps, double alpha);

// Fill a full global grid with the initial condition (top row hot).
void init_global(std::vector<double>& grid, int ny, int nx);
