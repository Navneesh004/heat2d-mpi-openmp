"""Pure-Python model of heat2d's parallel logic (mirrors src/*.cpp line by line).

Simulates P MPI ranks (separate arrays, no shared memory), the ghost rows,
Sendrecv / Isend-Irecv halo exchange, the overlap ordering, the blocked
kernel's per-thread row split, and checks bit-identity vs the serial reference.
Unused ghost cells are poisoned with NaN so any illegal read is detected.
"""
import math
import itertools

NAN = float("nan")
HOT, COLD = 100.0, 0.0


class Decomp:
    def __init__(s, ny, nx, rank, size):
        s.ny, s.nx, s.rank, s.size = ny, nx, rank, size
        s.local = s.rows_of(rank)
        s.g0 = s.offset_of(rank)

    def rows_of(s, r): return s.ny // s.size + (1 if r < s.ny % s.size else 0)
    def offset_of(s, r): return r * (s.ny // s.size) + min(r, s.ny % s.size)
    def first(s): return max(1, 2 - s.g0)
    def last(s): return min(s.local, s.ny - 1 - s.g0)


def init_local(d):
    rows = d.local + 2
    a = [0.0] * (rows * d.nx)
    for i in range(rows):
        g = d.g0 + i - 1
        v = HOT if (1 <= i <= d.local and g == 0) else COLD
        if i == 0 or i == d.local + 1:
            v = NAN  # poison ghosts: must be filled by exchange before any read
        for j in range(d.nx):
            a[i * d.nx + j] = v
    return a, list(a)


def row_update(u, un, nx, i, j0, j1, alpha, touched):
    o = i * nx
    for j in range(j0, j1):
        c = u[o + j]
        un[o + j] = c + alpha * (u[o - nx + j] + u[o + nx + j] + u[o + j - 1] + u[o + j + 1] - 4.0 * c)
        touched.append((i, j))


def stencil_rows(u, un, nx, r0, r1, alpha, block, nthreads, touched):
    nrows = r1 - r0
    if nrows <= 0:
        return
    j0, j1 = 1, nx - 1
    if block <= 0 or block >= j1 - j0:
        for i in range(r0, r1):
            row_update(u, un, nx, i, j0, j1, alpha, touched)
        return
    nt = nthreads if nrows >= 4 else 1  # `if(nrows >= MIN_ROWS_FOR_THREADS)`
    for t in range(nt):
        chunk, rem = nrows // nt, nrows % nt
        a = r0 + t * chunk + min(t, rem)
        b = a + chunk + (1 if t < rem else 0)
        jj = j0
        while jj < j1:
            je = min(jj + block, j1)
            for i in range(a, b):
                row_update(u, un, nx, i, jj, je, alpha, touched)
            jj += block


def halo(ranks, U):
    """Both Sendrecv calls, done 'simultaneously' from snapshots of the send rows."""
    size = len(ranks)
    sends = []
    for r, d in enumerate(ranks):
        u, nx = U[r], d.nx
        sends.append((u[1 * nx:2 * nx], u[d.local * nx:(d.local + 1) * nx]))
    for r, d in enumerate(ranks):
        u, nx = U[r], d.nx
        up = r - 1 if r > 0 else None
        down = r + 1 if r < size - 1 else None
        if down is not None:  # bottom ghost <- first real row of rank below
            u[(d.local + 1) * nx:(d.local + 2) * nx] = sends[down][0]
        if up is not None:    # top ghost <- last real row of rank above
            u[0:nx] = sends[up][1]


def run_parallel(ny, nx, steps, size, alpha=0.25, block=0, overlap=False, nthreads=1):
    ranks = [Decomp(ny, nx, r, size) for r in range(size)]
    U, UN = [], []
    for d in ranks:
        a, b = init_local(d)
        U.append(a); UN.append(b)
    for _ in range(steps):
        for r, d in enumerate(ranks):
            d._touched = []
        if not overlap:
            halo(ranks, U)
            for r, d in enumerate(ranks):
                stencil_rows(U[r], UN[r], nx, d.first(), d.last() + 1, alpha, block, nthreads, d._touched)
        else:
            # interior while ghosts are "in flight": re-poison ghosts to catch reads
            for r, d in enumerate(ranks):
                u = U[r]
                for j in range(nx):
                    u[j] = NAN; u[(d.local + 1) * nx + j] = NAN
            for r, d in enumerate(ranks):
                in_lo, in_hi = max(d.first(), 2), min(d.last(), d.local - 1)
                stencil_rows(U[r], UN[r], nx, in_lo, in_hi + 1, alpha, block, nthreads, d._touched)
            halo(ranks, U)
            for r, d in enumerate(ranks):
                first, last = d.first(), d.last()
                if first == 1 and last >= 1:
                    stencil_rows(U[r], UN[r], nx, 1, 2, alpha, block, nthreads, d._touched)
                if last == d.local and d.local > 1 and first <= d.local:
                    stencil_rows(U[r], UN[r], nx, d.local, d.local + 1, alpha, block, nthreads, d._touched)
        # every interior cell updated exactly once per step
        for r, d in enumerate(ranks):
            t = d._touched
            assert len(t) == len(set(t)), f"rank {r}: cell updated twice"
            exp = {(i, j) for i in range(d.first(), d.last() + 1) for j in range(1, nx - 1)}
            assert set(t) == exp, f"rank {r}: wrong cell set (missing {len(exp - set(t))}, extra {len(set(t) - exp)})"
        for r in range(size):
            U[r], UN[r] = UN[r], U[r]
    # MPI_Gatherv of owned rows
    full = [None] * (ny * nx)
    counts = [d.rows_of(r) * nx for r, d in enumerate(ranks)]
    displs = [ranks[0].offset_of(r) * nx for r in range(size)]
    assert sum(counts) == ny * nx and displs[-1] + counts[-1] == ny * nx
    for r, d in enumerate(ranks):
        full[displs[r]:displs[r] + counts[r]] = U[r][nx:(d.local + 1) * nx]
    return full


def reference(ny, nx, steps, alpha=0.25):
    g = [COLD] * (ny * nx)
    for j in range(nx):
        g[j] = HOT
    gn = list(g)
    for _ in range(steps):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                k = i * nx + j
                gn[k] = g[k] + alpha * (g[k - nx] + g[k + nx] + g[k - 1] + g[k + 1] - 4.0 * g[k])
        g, gn = gn, g
    return g


def check(name, ny, nx, steps, size, **kw):
    ref = reference(ny, nx, steps)
    got = run_parallel(ny, nx, steps, size, **kw)
    nan = any(math.isnan(x) for x in got)
    maxdiff = max(abs(a - b) for a, b in zip(ref, got)) if not nan else NAN
    ok = (not nan) and maxdiff == 0.0
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {ny}x{nx} np={size} {kw}  maxdiff={maxdiff}")
    return ok


if __name__ == "__main__":
    ok = True
    # the same configurations as the CTest suite (fewer steps: pure Python)
    ok &= check("serial", 67, 61, 40, 1)
    ok &= check("mpi_np2", 67, 61, 40, 2)
    ok &= check("mpi_np3", 67, 61, 40, 3)
    ok &= check("mpi_np4", 67, 61, 40, 4)
    ok &= check("mpi_np5_tiny", 7, 20, 30, 5)
    ok &= check("blocked_np3_t2", 67, 61, 40, 3, block=16, nthreads=2)
    ok &= check("overlap_np4", 67, 61, 40, 4, overlap=True)
    ok &= check("overlap_tiny", 6, 20, 30, 4, overlap=True, block=8, nthreads=2)
    # exhaustive edge cases: every rank count up to ny, 1-row ranks, ranks owning only boundary rows
    for ny, size, ov, blk, nt in itertools.product([3, 4, 5, 9], range(1, 10), [False, True], [0, 3], [1, 3]):
        if size > ny:
            continue
        ref = reference(ny, 9, 12)
        got = run_parallel(ny, 9, 12, size, overlap=ov, block=blk, nthreads=nt)
        if any(math.isnan(x) for x in got) or got != ref:
            print(f"FAIL  edge ny={ny} np={size} overlap={ov} block={blk} nt={nt}")
            ok = False
    print("edge-case sweep done")
    # sanity: physics - heat flows in, stays within [0, 100], symmetric left/right
    g = reference(33, 33, 400)
    assert all(0.0 <= x <= 100.0 for x in g)
    assert all(abs(g[i * 33 + j] - g[i * 33 + 32 - j]) < 1e-12 for i in range(33) for j in range(33))
    assert g[1 * 33 + 16] > g[10 * 33 + 16] > g[20 * 33 + 16] > 0
    print("physics sanity: bounded, symmetric, monotone away from hot edge")
    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)
