// io.hpp - binary output of grids and snapshots.
#pragma once

#include <string>
#include <vector>

// Full-precision grid: header int32 ny, int32 nx, then ny*nx float64 (row-major).
bool write_grid(const std::string& path, const std::vector<double>& g, int ny, int nx);

// Snapshot for animation: downsampled so neither side exceeds max_dim.
// Header int32 ny, int32 nx, int32 step, then ny*nx float32 (row-major).
bool write_snapshot(const std::string& path, const std::vector<double>& g, int ny, int nx,
                    int step, int max_dim);
