// io.cpp - binary output.
#include "io.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>

bool write_grid(const std::string& path, const std::vector<double>& g, int ny, int nx) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const std::int32_t hdr[2] = {ny, nx};
    bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
              std::fwrite(g.data(), sizeof(double), g.size(), f) == g.size();
    return std::fclose(f) == 0 && ok;
}

bool write_snapshot(const std::string& path, const std::vector<double>& g, int ny, int nx,
                    int step, int max_dim) {
    const int sy = std::max(1, (ny + max_dim - 1) / max_dim);
    const int sx = std::max(1, (nx + max_dim - 1) / max_dim);
    const int oy = (ny + sy - 1) / sy, ox = (nx + sx - 1) / sx;
    std::vector<float> out(static_cast<std::size_t>(oy) * ox);
    for (int i = 0; i < oy; ++i)
        for (int j = 0; j < ox; ++j)
            out[static_cast<std::size_t>(i) * ox + j] =
                static_cast<float>(g[static_cast<std::size_t>(i) * sy * nx + static_cast<std::size_t>(j) * sx]);

    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const std::int32_t hdr[3] = {oy, ox, step};
    bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
              std::fwrite(out.data(), sizeof(float), out.size(), f) == out.size();
    return std::fclose(f) == 0 && ok;
}
