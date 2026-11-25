# Benchmark results

## Strong scaling (4096x4096, 500 steps, median of 5 runs)

| mode | ranks x threads | cores | time (s) | halo (s, mean/rank) | speedup | efficiency |
|---|---|---|---|---|---|---|
| serial | 1x1 | 1 | 20.885 | 0.001 | 1.00x | 100% |
| mpi | 2x1 | 2 | 12.801 | 1.969 | 1.63x | 82% |
| openmp | 1x2 | 2 | 13.380 | 0.001 | 1.56x | 78% |
| hybrid | 2x2 | 4 | 11.109 | 0.865 | 1.88x | 47% |
| mpi | 4x1 | 4 | 11.075 | 2.186 | 1.89x | 47% |
| openmp | 1x4 | 4 | 11.086 | 0.001 | 1.88x | 47% |

## Weak scaling (4096 cols x 1024 rows/core)

| mode | cores | grid | time (s) | halo (s, mean/rank) | efficiency |
|---|---|---|---|---|---|
| serial | 1 | 1024x4096 | 5.683 | 0.001 | 100% |
| mpi | 2 | 2048x4096 | 6.699 | 1.169 | 85% |
| openmp | 2 | 2048x4096 | 7.313 | 0.001 | 78% |
| mpi | 4 | 4096x4096 | 11.297 | 1.990 | 50% |
| openmp | 4 | 4096x4096 | 12.558 | 0.001 | 45% |

## Mode comparison at 4 cores

| mode | ranks x threads | time (s) | compute (s, mean/rank) | halo (s, mean/rank) | MLUP/s |
|---|---|---|---|---|---|
| openmp | 1x4 | 11.587 | 11.585 | 0.001 | 723 |
| hybrid | 2x2 | 11.676 | 10.234 | 1.441 | 718 |
| mpi | 4x1 | 12.436 | 9.125 | 3.292 | 674 |

## Optimisations (before / after)

| config | variant | time (s) | halo (s, mean/rank) | change vs baseline |
|---|---|---|---|---|
| openmp 1x4 | baseline | 12.697 | 0.001 | +0.0% |
| openmp 1x4 | blocked | 14.848 | 0.001 | +16.9% |
| mpi 4x1 | baseline | 14.269 | 2.877 | +0.0% |
| mpi 4x1 | overlap | 12.555 | 1.944 | -12.0% |
| mpi 4x1 | blocked | 14.598 | 4.064 | +2.3% |
| mpi 4x1 | blocked+overlap | 14.166 | 3.073 | -0.7% |

