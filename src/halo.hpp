// halo.hpp - ghost-row ("halo") exchange between neighbouring ranks.
#pragma once

#include <mpi.h>

struct Neighbours {
    int up = MPI_PROC_NULL;   // rank owning the rows above (MPI_PROC_NULL on rank 0)
    int down = MPI_PROC_NULL; // rank owning the rows below (MPI_PROC_NULL on last rank)
};

Neighbours make_neighbours(int rank, int size);

// Blocking exchange with two MPI_Sendrecv calls (deadlock-free).
void halo_exchange(double* u, int nx, int local, const Neighbours& nb, MPI_Comm comm);

// Non-blocking exchange: post 2 Irecv + 2 Isend, compute the interior, then
// wait. `req` must have room for 4 requests.
void halo_start(double* u, int nx, int local, const Neighbours& nb, MPI_Comm comm, MPI_Request req[4]);
void halo_finish(MPI_Request req[4]);
