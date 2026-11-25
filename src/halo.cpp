// halo.cpp - ghost-row exchange.
//
// Local layout: row 0 = top ghost, rows 1..local = owned, row local+1 = bottom ghost.
// MPI_PROC_NULL as a peer turns a send/recv into a no-op, so the top and bottom
// ranks need no special-casing.
#include "halo.hpp"

#include <cstddef>

namespace {
constexpr int TAG_UP = 0;   // data travelling to the rank above
constexpr int TAG_DOWN = 1; // data travelling to the rank below

inline double* row(double* u, int i, int nx) { return u + static_cast<std::size_t>(i) * nx; }
} // namespace

Neighbours make_neighbours(int rank, int size) {
    Neighbours nb;
    nb.up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    nb.down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    return nb;
}

void halo_exchange(double* u, int nx, int local, const Neighbours& nb, MPI_Comm comm) {
    // send my first real row up; receive my bottom ghost from below
    MPI_Sendrecv(row(u, 1, nx), nx, MPI_DOUBLE, nb.up, TAG_UP,
                 row(u, local + 1, nx), nx, MPI_DOUBLE, nb.down, TAG_UP,
                 comm, MPI_STATUS_IGNORE);
    // send my last real row down; receive my top ghost from above
    MPI_Sendrecv(row(u, local, nx), nx, MPI_DOUBLE, nb.down, TAG_DOWN,
                 row(u, 0, nx), nx, MPI_DOUBLE, nb.up, TAG_DOWN,
                 comm, MPI_STATUS_IGNORE);
}

void halo_start(double* u, int nx, int local, const Neighbours& nb, MPI_Comm comm, MPI_Request req[4]) {
    MPI_Irecv(row(u, 0, nx), nx, MPI_DOUBLE, nb.up, TAG_DOWN, comm, &req[0]);
    MPI_Irecv(row(u, local + 1, nx), nx, MPI_DOUBLE, nb.down, TAG_UP, comm, &req[1]);
    MPI_Isend(row(u, 1, nx), nx, MPI_DOUBLE, nb.up, TAG_UP, comm, &req[2]);
    MPI_Isend(row(u, local, nx), nx, MPI_DOUBLE, nb.down, TAG_DOWN, comm, &req[3]);
}

void halo_finish(MPI_Request req[4]) {
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}
