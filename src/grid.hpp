// grid.hpp - 1D row-wise domain decomposition and the local strip with ghost rows.
//
// Global grid: ny rows x nx columns, row-major, index i*nx + j.
// Boundary: top row (global row 0) fixed at 100, all other edges fixed at 0.
//
// Each MPI rank owns a contiguous block of `local` global rows starting at
// global row `g0`. Its local array has local+2 rows:
//   row 0           top ghost    (copy of the last owned row of rank-1)
//   rows 1..local   owned rows   (global rows g0 .. g0+local-1)
//   row local+1     bottom ghost (copy of the first owned row of rank+1)
#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

constexpr double HOT_EDGE = 100.0;
constexpr double COLD_EDGE = 0.0;

struct Decomp {
    int ny = 0, nx = 0;     // global size
    int rank = 0, size = 1; // MPI rank / communicator size
    int local = 0;          // number of owned rows
    int g0 = 0;             // global index of first owned row

    Decomp() = default;
    Decomp(int ny_, int nx_, int rank_, int size_) : ny(ny_), nx(nx_), rank(rank_), size(size_) {
        local = rows_of(rank);
        g0 = offset_of(rank);
    }

    // Block distribution: the first (ny % size) ranks get one extra row.
    int rows_of(int r) const { return ny / size + (r < ny % size ? 1 : 0); }
    int offset_of(int r) const { return r * (ny / size) + std::min(r, ny % size); }

    // Owned local rows that are NOT on the fixed global boundary, as an
    // inclusive range [first, last]. Empty if first > last.
    int first_update_row() const { return std::max(1, 2 - g0); }
    int last_update_row() const { return std::min(local, ny - 1 - g0); }
};

// Two buffers (current / next) of (local+2) x nx doubles.
// Memory is allocated uninitialised and then first-touched inside an OpenMP
// parallel loop with the same static row split as the compute kernel, so on a
// NUMA machine each thread's rows land in memory close to its core.
struct LocalGrid {
    Decomp d;
    std::size_t n = 0;
    std::unique_ptr<double[]> a, b;
    double* u = nullptr;    // current
    double* unew = nullptr; // next

    explicit LocalGrid(const Decomp& dec) : d(dec) {
        n = static_cast<std::size_t>(d.local + 2) * d.nx;
        a.reset(new double[n]); // `new double[n]` does not zero -> no serial first touch
        b.reset(new double[n]);
        u = a.get();
        unew = b.get();
        first_touch_init();
    }

    void swap() { std::swap(u, unew); }

    double* row(int i) { return u + static_cast<std::size_t>(i) * d.nx; }

private:
    void first_touch_init() {
        const int nx = d.nx, rows = d.local + 2;
        double* pa = a.get();
        double* pb = b.get();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const int g = d.g0 + i - 1; // global row (ghosts map to g0-1 / g0+local)
            const double v = (i >= 1 && i <= d.local && g == 0) ? HOT_EDGE : COLD_EDGE;
            const std::size_t o = static_cast<std::size_t>(i) * nx;
            for (int j = 0; j < nx; ++j) {
                pa[o + j] = v;
                pb[o + j] = v;
            }
        }
    }
};
