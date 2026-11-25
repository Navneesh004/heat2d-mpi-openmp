// main.cpp - hybrid MPI + OpenMP 2D heat-diffusion solver.
//
// One binary covers every mode:
//   OMP_NUM_THREADS=8 ./heat 4096 1000                                   # OpenMP only
//   OMP_NUM_THREADS=1 mpirun -np 8 ./heat 4096 1000                      # MPI only
//   OMP_NUM_THREADS=4 mpirun -np 2 --map-by slot:PE=4 ./heat 4096 1000   # hybrid
//
// Run `./heat --help` for all options.
#include "grid.hpp"
#include "halo.hpp"
#include "io.hpp"
#include "kernel.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
    int nx = 0;           // columns
    int ny = 0;           // rows (defaults to nx)
    int steps = 0;
    double alpha = 0.25;
    int block = 0;        // 0 = no cache blocking, else column tile width
    bool overlap = false; // overlap halo exchange with interior compute
    bool verify = false;  // compare against serial reference on rank 0
    int snap_every = 0;   // write a snapshot every k steps (0 = never)
    int snap_max = 256;   // snapshot resolution cap
    std::string snap_dir = "snapshots";
    std::string out;      // write final full grid here
    bool csv = false;     // machine-readable one-line output
    bool csv_header = false;
};

void usage(const char* prog) {
    std::printf(
        "usage: %s NX STEPS [options]\n"
        "  NX                 grid columns (and rows, unless --rows is given)\n"
        "  STEPS              number of time steps\n"
        "options:\n"
        "  --rows NY          global rows (for weak scaling)\n"
        "  --alpha A          diffusion coefficient, must be <= 0.25 (default 0.25)\n"
        "  --block B          cache-blocked kernel with column tiles of B doubles\n"
        "  --overlap          overlap halo exchange with interior computation\n"
        "  --verify           gather result and compare with a serial reference\n"
        "  --snap K           write a snapshot every K steps (not timed)\n"
        "  --snap-dir DIR     snapshot directory (default: snapshots)\n"
        "  --snap-max M       max snapshot resolution (default: 256)\n"
        "  --out FILE         write final grid (int32 ny, int32 nx, float64 data)\n"
        "  --csv              print one CSV result line\n"
        "  --csv-header       print the CSV header and exit\n",
        prog);
}

const char* CSV_HEADER =
    "mode,ranks,threads,cores,nx,ny,steps,kernel,overlap,total_s,compute_s,halo_s,mlups,checksum";

bool parse(int argc, char** argv, Options& o, int rank) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                if (rank == 0) std::fprintf(stderr, "error: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { if (rank == 0) usage(argv[0]); return false; }
        else if (a == "--rows") o.ny = std::atoi(need("--rows"));
        else if (a == "--alpha") o.alpha = std::atof(need("--alpha"));
        else if (a == "--block") o.block = std::atoi(need("--block"));
        else if (a == "--overlap") o.overlap = true;
        else if (a == "--verify") o.verify = true;
        else if (a == "--snap") o.snap_every = std::atoi(need("--snap"));
        else if (a == "--snap-dir") o.snap_dir = need("--snap-dir");
        else if (a == "--snap-max") o.snap_max = std::atoi(need("--snap-max"));
        else if (a == "--out") o.out = need("--out");
        else if (a == "--csv") o.csv = true;
        else if (a == "--csv-header") o.csv_header = true;
        else if (!a.empty() && a[0] == '-') {
            if (rank == 0) std::fprintf(stderr, "error: unknown option %s\n", a.c_str());
            std::exit(2);
        } else pos.push_back(a);
    }
    if (o.csv_header) return true;
    if (pos.size() != 2) { if (rank == 0) usage(argv[0]); std::exit(2); }
    o.nx = std::atoi(pos[0].c_str());
    o.steps = std::atoi(pos[1].c_str());
    if (o.ny == 0) o.ny = o.nx;
    return true;
}

// Collect every rank's owned rows into a full ny x nx grid on rank 0.
std::vector<double> gather(const LocalGrid& g, MPI_Comm comm) {
    const Decomp& d = g.d;
    std::vector<int> counts, displs;
    std::vector<double> full;
    if (d.rank == 0) {
        counts.resize(d.size);
        displs.resize(d.size);
        for (int r = 0; r < d.size; ++r) {
            counts[r] = d.rows_of(r) * d.nx;
            displs[r] = d.offset_of(r) * d.nx;
        }
        full.resize(static_cast<std::size_t>(d.ny) * d.nx);
    }
    MPI_Gatherv(g.u + d.nx, d.local * d.nx, MPI_DOUBLE, full.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, comm);
    return full;
}

std::string mode_name(int ranks, int threads) {
    if (ranks == 1 && threads == 1) return "serial";
    if (ranks == 1) return "openmp";
    if (threads == 1) return "mpi";
    return "hybrid";
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    Options opt;
    if (!parse(argc, argv, opt, rank)) { MPI_Finalize(); return 0; }
    if (opt.csv_header) {
        if (rank == 0) std::printf("%s\n", CSV_HEADER);
        MPI_Finalize();
        return 0;
    }

    auto fail = [&](const char* msg) {
        if (rank == 0) std::fprintf(stderr, "error: %s\n", msg);
        MPI_Finalize();
        std::exit(2);
    };
    if (provided < MPI_THREAD_FUNNELED) fail("MPI library does not support MPI_THREAD_FUNNELED");
    if (opt.nx < 3 || opt.ny < 3) fail("grid must be at least 3x3");
    if (opt.steps < 0) fail("STEPS must be >= 0");
    if (opt.ny < size) fail("need at least one row per MPI rank (--rows >= number of ranks)");
    if (!(opt.alpha > 0.0 && opt.alpha <= 0.25)) fail("alpha must be in (0, 0.25] for stability");
    if (opt.block < 0) fail("--block must be >= 0");

    const int threads = omp_get_max_threads();
    const Decomp dec(opt.ny, opt.nx, rank, size);
    const Neighbours nb = make_neighbours(rank, size);
    LocalGrid g(dec);

    const int first = dec.first_update_row(); // inclusive range of rows this rank updates
    const int last = dec.last_update_row();
    // In overlap mode, rows 1 and `local` depend on ghost rows; everything
    // strictly between them can be computed while the halo is in flight.
    const int in_lo = std::max(first, 2);
    const int in_hi = std::min(last, dec.local - 1);

    auto snapshot = [&](int step) {
        std::vector<double> full = gather(g, comm);
        if (rank == 0) {
            char name[64];
            std::snprintf(name, sizeof name, "/snap_%07d.bin", step);
            if (!write_snapshot(opt.snap_dir + name, full, opt.ny, opt.nx, step, opt.snap_max))
                std::fprintf(stderr, "warning: could not write snapshot %d\n", step);
        }
    };
    if (opt.snap_every > 0) snapshot(0);

    // Untimed warm-up exchange: MPI libraries set up shared-memory/network
    // connections lazily on the first message, which would otherwise be
    // charged to the halo timer of step 1. (Ghost rows are rewritten every
    // step, so this has no effect on the result.)
    halo_exchange(g.u, opt.nx, dec.local, nb, comm);

    double t_compute = 0.0, t_halo = 0.0, t_snap = 0.0;
    MPI_Barrier(comm);
    const double t_start = MPI_Wtime();

    for (int step = 1; step <= opt.steps; ++step) {
        if (!opt.overlap) {
            double t0 = MPI_Wtime();
            halo_exchange(g.u, opt.nx, dec.local, nb, comm);
            double t1 = MPI_Wtime();
            stencil_rows(g.u, g.unew, opt.nx, first, last + 1, opt.alpha, opt.block);
            double t2 = MPI_Wtime();
            t_halo += t1 - t0;
            t_compute += t2 - t1;
        } else {
            MPI_Request req[4];
            double t0 = MPI_Wtime();
            halo_start(g.u, opt.nx, dec.local, nb, comm, req);
            double t1 = MPI_Wtime();
            stencil_rows(g.u, g.unew, opt.nx, in_lo, in_hi + 1, opt.alpha, opt.block);
            double t2 = MPI_Wtime();
            halo_finish(req);
            double t3 = MPI_Wtime();
            if (first == 1 && last >= 1)
                stencil_rows(g.u, g.unew, opt.nx, 1, 2, opt.alpha, opt.block);
            if (last == dec.local && dec.local > 1 && first <= dec.local)
                stencil_rows(g.u, g.unew, opt.nx, dec.local, dec.local + 1, opt.alpha, opt.block);
            double t4 = MPI_Wtime();
            t_halo += (t1 - t0) + (t3 - t2);
            t_compute += (t2 - t1) + (t4 - t3);
        }
        g.swap();

        if (opt.snap_every > 0 && step % opt.snap_every == 0) {
            double ts = MPI_Wtime();
            snapshot(step);
            t_snap += MPI_Wtime() - ts;
        }
    }

    const double t_total_local = MPI_Wtime() - t_start - t_snap;

    // Total: the slowest rank, since that is what limits the run.
    // Compute / halo: averaged over ranks, so compute + halo <= total. (Taking
    // the max of each separately can pick different ranks and add up to more
    // than the wall time.) Halo time includes waiting for a slower neighbour,
    // so it measures synchronisation cost, not just message transfer.
    double t_rep_total = 0.0;
    MPI_Reduce(&t_total_local, &t_rep_total, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    double local_t[2] = {t_compute, t_halo}, sum_t[2] = {0, 0};
    MPI_Reduce(local_t, sum_t, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    const double rep_t[3] = {t_rep_total, sum_t[0] / size, sum_t[1] / size};

    // Checksum: sum over all owned cells, reduced in a fixed rank order on
    // rank 0 so it does not depend on MPI's reduction tree.
    double local_sum = 0.0;
    for (int i = 1; i <= dec.local; ++i)
        for (int j = 0; j < opt.nx; ++j) local_sum += g.row(i)[j];
    std::vector<double> sums(rank == 0 ? size : 0);
    MPI_Gather(&local_sum, 1, MPI_DOUBLE, sums.data(), 1, MPI_DOUBLE, 0, comm);
    double checksum = 0.0;
    for (double s : sums) checksum += s;

    int exit_code = 0;
    std::vector<double> full;
    if (opt.verify || !opt.out.empty()) full = gather(g, comm);

    if (rank == 0) {
        const std::string mode = mode_name(size, threads);
        const char* kernel = opt.block > 0 ? "blocked" : "naive";
        const double cells = double(opt.ny - 2) * double(opt.nx - 2) * double(opt.steps);
        const double mlups = rep_t[0] > 0 ? cells / rep_t[0] / 1e6 : 0.0;

        if (opt.csv) {
            std::printf("%s,%d,%d,%d,%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.2f,%.10e\n", mode.c_str(), size,
                        threads, size * threads, opt.nx, opt.ny, opt.steps, kernel, opt.overlap ? 1 : 0,
                        rep_t[0], rep_t[1], rep_t[2], mlups, checksum);
        } else {
            std::printf("mode=%s ranks=%d threads=%d grid=%dx%d steps=%d kernel=%s%s overlap=%d\n",
                        mode.c_str(), size, threads, opt.ny, opt.nx, opt.steps, kernel,
                        opt.block > 0 ? ("(" + std::to_string(opt.block) + ")").c_str() : "",
                        opt.overlap ? 1 : 0);
            std::printf("time total=%.4fs (slowest rank) compute=%.4fs halo=%.4fs (mean over ranks)\n", rep_t[0],
                        rep_t[1], rep_t[2]);
            std::printf("throughput=%.1f MLUP/s checksum=%.10e\n", mlups, checksum);
        }

        if (!opt.out.empty() && !write_grid(opt.out, full, opt.ny, opt.nx)) {
            std::fprintf(stderr, "error: could not write %s\n", opt.out.c_str());
            exit_code = 1;
        }

        if (opt.verify) {
            std::vector<double> ref;
            init_global(ref, opt.ny, opt.nx);
            reference_solve(ref, opt.ny, opt.nx, opt.steps, opt.alpha);
            double maxdiff = 0.0;
            for (std::size_t k = 0; k < ref.size(); ++k)
                maxdiff = std::max(maxdiff, std::fabs(ref[k] - full[k]));
            const bool ok = maxdiff == 0.0;
            std::fprintf(opt.csv ? stderr : stdout, "verify: max|parallel - serial| = %.3e  %s\n",
                         maxdiff, ok ? "PASS (bit-identical)" : "FAIL");
            if (!ok) exit_code = 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, comm);
    MPI_Finalize();
    return exit_code;
}
