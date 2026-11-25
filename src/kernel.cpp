// kernel.cpp - stencil kernels.
//
// Update rule (explicit Euler, 5-point Laplacian):
//   unew[i][j] = u[i][j] + alpha * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4 u[i][j])
// Stable for alpha <= 0.25.
//
// NOTE: the expression is written in the same order everywhere and the code is
// compiled with -ffp-contract=off (no FMA fusion), so every implementation in
// this file gives bit-identical results regardless of how the grid is split.
#include "kernel.hpp"
#include "grid.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace {

inline void row_update(const double* __restrict up, const double* __restrict c,
                       const double* __restrict dn, double* __restrict out,
                       int j0, int j1, double alpha) {
#pragma omp simd
    for (int j = j0; j < j1; ++j)
        out[j] = c[j] + alpha * (up[j] + dn[j] + c[j - 1] + c[j + 1] - 4.0 * c[j]);
}

// Parallelising tiny row ranges (the 1-2 edge rows in overlap mode) costs more
// than it saves; let the calling thread do them.
constexpr int MIN_ROWS_FOR_THREADS = 4;

} // namespace

void stencil_rows(const double* u, double* unew, int nx, int r0, int r1, double alpha, int block) {
    const int nrows = r1 - r0;
    if (nrows <= 0) return;
    const int j0 = 1, j1 = nx - 1;

    if (block <= 0 || block >= j1 - j0) {
#pragma omp parallel for schedule(static) if (nrows >= MIN_ROWS_FOR_THREADS)
        for (int i = r0; i < r1; ++i) {
            const std::size_t o = static_cast<std::size_t>(i) * nx;
            row_update(u + o - nx, u + o, u + o + nx, unew + o, j0, j1, alpha);
        }
        return;
    }

    // Cache-blocked: split rows between threads exactly like schedule(static)
    // (keeps first-touch locality), then sweep each thread's rows tile by tile.
#pragma omp parallel if (nrows >= MIN_ROWS_FOR_THREADS)
    {
        const int t = omp_get_thread_num(), nt = omp_get_num_threads();
        const int chunk = nrows / nt, rem = nrows % nt;
        const int a = r0 + t * chunk + std::min(t, rem);
        const int b = a + chunk + (t < rem ? 1 : 0);
        for (int jj = j0; jj < j1; jj += block) {
            const int je = std::min(jj + block, j1);
            for (int i = a; i < b; ++i) {
                const std::size_t o = static_cast<std::size_t>(i) * nx;
                row_update(u + o - nx, u + o, u + o + nx, unew + o, jj, je, alpha);
            }
        }
    }
}

void init_global(std::vector<double>& g, int ny, int nx) {
    g.assign(static_cast<std::size_t>(ny) * nx, COLD_EDGE);
    for (int j = 0; j < nx; ++j) g[j] = HOT_EDGE;
}

void reference_solve(std::vector<double>& g, int ny, int nx, int steps, double alpha) {
    std::vector<double> gn = g; // boundaries copied once, never written again
    for (int s = 0; s < steps; ++s) {
        for (int i = 1; i < ny - 1; ++i)
            for (int j = 1; j < nx - 1; ++j) {
                const std::size_t k = static_cast<std::size_t>(i) * nx + j;
                gn[k] = g[k] + alpha * (g[k - nx] + g[k + nx] + g[k - 1] + g[k + 1] - 4.0 * g[k]);
            }
        std::swap(g, gn);
    }
}
