// Copyright (c) 2026 The FireANTs authors and contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


/*
CPU implementations of the 3D fused grid sampler, the 3D warp composer and the 3D warp generator.
They follow the semantics of FusedGridSampler.cu, FusedGridComposer.cu and FusedGenerateGrid.cu:
the output (and gradients) are accumulated into zero-initialized buffers, the sampling coordinate is
A * x + s * u(x) (displacement) or s * u(x) (full warp), and batch broadcasting follows the CUDA rules.

Scatter-adds into grad_input use one private buffer per thread (no atomics), reduced at the end.
*/
#include <ATen/OpMathType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/GridSampler.h>
#include <cmath>
#include <vector>
#include "cpu_utils.h"
#include "CPUOps.h"

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
using at::native::grid_sampler_compute_source_index;
using at::native::grid_sampler_compute_source_index_set_grad;

namespace {

struct Geometry {
    int64_t N, C, Di, Hi, Wi, D, H, W;
    // strides of the sampled tensor: batch, channel, spatial element
    int64_t inp_sN, inp_sC, inp_sS;
    // strides of the output tensor: batch, channel, spatial element
    int64_t out_sN, out_sC, out_sS;
    // batch strides of grid / affine / grid affine (0 = broadcast)
    int64_t grid_sN, aff_sN, pre_sN;
    float xmin, ymin, zmin, xmax, ymax, zmax;
    double xstep, ystep, zstep;
    bool is_displacement;
    GridSamplerInterpolation interp;
    GridSamplerPadding padding;
    bool align_corners;
};

template <typename gm>
struct Coord {
    gm px, py, pz;   // pre-affine (normalized grid) coordinate
    gm x, y, z;      // sampling coordinate in [-1, 1]
};

// computes the sampling coordinate of output voxel (n, d, h, w)
template <typename grid_t, typename gm>
inline Coord<gm> sampling_coordinate(const Geometry& g, const grid_t* grid, const grid_t* aff, const grid_t* pre,
                                     int64_t n, int64_t d, int64_t h, int64_t w) {
    Coord<gm> c;
    c.px = static_cast<gm>(w) * static_cast<gm>(g.xstep) + static_cast<gm>(g.xmin);
    c.py = static_cast<gm>(h) * static_cast<gm>(g.ystep) + static_cast<gm>(g.ymin);
    c.pz = static_cast<gm>(d) * static_cast<gm>(g.zstep) + static_cast<gm>(g.zmin);
    const bool use_affine = aff != nullptr && (grid == nullptr || g.is_displacement);
    if (use_affine) {
        const grid_t* A = aff + n * g.aff_sN;
        c.x = A[0] * c.px + A[1] * c.py + A[2] * c.pz + A[3];
        c.y = A[4] * c.px + A[5] * c.py + A[6] * c.pz + A[7];
        c.z = A[8] * c.px + A[9] * c.py + A[10] * c.pz + A[11];
    } else {
        c.x = c.px; c.y = c.py; c.z = c.pz;
    }
    if (grid != nullptr) {
        const grid_t* u = grid + n * g.grid_sN + 3 * (w + g.W * (h + g.H * d));
        gm dx = u[0], dy = u[1], dz = u[2];
        if (pre != nullptr) {
            const grid_t* P = pre + n * g.pre_sN;
            const gm tx = P[0] * dx + P[1] * dy + P[2] * dz;
            const gm ty = P[3] * dx + P[4] * dy + P[5] * dz;
            const gm tz = P[6] * dx + P[7] * dy + P[8] * dz;
            dx = tx; dy = ty; dz = tz;
        }
        if (g.is_displacement) {
            c.x += dx; c.y += dy; c.z += dz;
        } else {
            c.x = dx; c.y = dy; c.z = dz;
        }
    }
    return c;
}

template <typename scalar_t, typename grid_t>
void sampler_forward(const Geometry& g, const scalar_t* inp, const grid_t* grid, const grid_t* aff, const grid_t* pre, scalar_t* out) {
    using om = at::opmath_type<scalar_t>;
    using gm = at::opmath_type<grid_t>;
    const int64_t Di = g.Di, Hi = g.Hi, Wi = g.Wi;
    const int64_t sS = g.inp_sS, sY = Wi * sS, sZ = Hi * Wi * sS;
    ffo_parallel_for(g.N * g.D * g.H, 4, [&](int64_t begin, int64_t end, int) {
        for (int64_t row = begin; row < end; ++row) {
            const int64_t h = row % g.H;
            const int64_t d = (row / g.H) % g.D;
            const int64_t n = row / (g.H * g.D);
            const scalar_t* inp_n = inp + n * g.inp_sN;
            scalar_t* out_row = out + n * g.out_sN + (g.W * (h + g.H * d)) * g.out_sS;
            for (int64_t w = 0; w < g.W; ++w) {
                const Coord<gm> c = sampling_coordinate<grid_t, gm>(g, grid, aff, pre, n, d, h, w);
                const gm sx = grid_sampler_compute_source_index(c.x, Wi, g.padding, g.align_corners);
                const gm sy = grid_sampler_compute_source_index(c.y, Hi, g.padding, g.align_corners);
                const gm sz = grid_sampler_compute_source_index(c.z, Di, g.padding, g.align_corners);
                scalar_t* o = out_row + w * g.out_sS;
                if (g.interp == GridSamplerInterpolation::Bilinear) {
                    const gm fx0 = std::floor(sx), fy0 = std::floor(sy), fz0 = std::floor(sz);
                    const int64_t x0 = static_cast<int64_t>(fx0), y0 = static_cast<int64_t>(fy0), z0 = static_cast<int64_t>(fz0);
                    const om fx = static_cast<om>(sx - fx0), fy = static_cast<om>(sy - fy0), fz = static_cast<om>(sz - fz0);
                    const om gx = 1 - fx, gy = 1 - fy, gz = 1 - fz;
                    if (x0 >= 0 && y0 >= 0 && z0 >= 0 && x0 + 1 < Wi && y0 + 1 < Hi && z0 + 1 < Di) {
                        const scalar_t* q = inp_n + x0 * sS + y0 * sY + z0 * sZ;
                        for (int64_t ch = 0; ch < g.C; ++ch, q += g.inp_sC) {
                            const om v = gz * (gy * (gx * static_cast<om>(q[0]) + fx * static_cast<om>(q[sS]))
                                             + fy * (gx * static_cast<om>(q[sY]) + fx * static_cast<om>(q[sY + sS])))
                                       + fz * (gy * (gx * static_cast<om>(q[sZ]) + fx * static_cast<om>(q[sZ + sS]))
                                             + fy * (gx * static_cast<om>(q[sZ + sY]) + fx * static_cast<om>(q[sZ + sY + sS])));
                            o[ch * g.out_sC] = static_cast<scalar_t>(static_cast<om>(o[ch * g.out_sC]) + v);
                        }
                    } else {
                        om wt[8];
                        int64_t off[8];
                        bool ok[8];
                        for (int k = 0; k < 8; ++k) {
                            const int64_t xi = x0 + (k & 1), yi = y0 + ((k >> 1) & 1), zi = z0 + (k >> 2);
                            ok[k] = xi >= 0 && yi >= 0 && zi >= 0 && xi < Wi && yi < Hi && zi < Di;
                            off[k] = xi * sS + yi * sY + zi * sZ;
                            wt[k] = ((k & 1) ? fx : gx) * (((k >> 1) & 1) ? fy : gy) * ((k >> 2) ? fz : gz);
                        }
                        for (int64_t ch = 0; ch < g.C; ++ch) {
                            const scalar_t* q = inp_n + ch * g.inp_sC;
                            om v = 0;
                            for (int k = 0; k < 8; ++k) {
                                if (ok[k]) v += wt[k] * static_cast<om>(q[off[k]]);
                            }
                            o[ch * g.out_sC] = static_cast<scalar_t>(static_cast<om>(o[ch * g.out_sC]) + v);
                        }
                    }
                } else {
                    const int64_t xn = static_cast<int64_t>(std::nearbyint(sx));
                    const int64_t yn = static_cast<int64_t>(std::nearbyint(sy));
                    const int64_t zn = static_cast<int64_t>(std::nearbyint(sz));
                    if (xn >= 0 && yn >= 0 && zn >= 0 && xn < Wi && yn < Hi && zn < Di) {
                        const scalar_t* q = inp_n + xn * sS + yn * sY + zn * sZ;
                        for (int64_t ch = 0; ch < g.C; ++ch) {
                            o[ch * g.out_sC] = static_cast<scalar_t>(static_cast<om>(o[ch * g.out_sC]) + static_cast<om>(q[ch * g.inp_sC]));
                        }
                    }
                }
            }
        }
    });
}

template <typename scalar_t, typename grid_t>
void sampler_backward(const Geometry& g, const scalar_t* inp, const grid_t* grid, const grid_t* aff, const grid_t* pre,
                      const scalar_t* gout, scalar_t* ginp, int64_t ginp_numel, grid_t* gaff, int64_t n_aff, grid_t* ggrid) {
    using om = at::opmath_type<scalar_t>;
    using gm = at::opmath_type<grid_t>;
    const int64_t Di = g.Di, Hi = g.Hi, Wi = g.Wi;
    const int64_t sS = g.inp_sS, sY = Wi * sS, sZ = Hi * Wi * sS;
    const int64_t work = g.D * g.H;
    const int nthreads = ffo_num_threads(work, 4);
    const bool bilinear = g.interp == GridSamplerInterpolation::Bilinear;
    const bool need_spatial = bilinear && (gaff != nullptr || ggrid != nullptr);

    // private scatter buffers for grad_input
    std::vector<om> ginp_priv;
    if (ginp != nullptr && nthreads > 1) {
        ginp_priv.assign(static_cast<size_t>(nthreads) * ginp_numel, om(0));
    }
    std::vector<double> gaff_priv(static_cast<size_t>(nthreads) * std::max<int64_t>(n_aff, 1) * 12, 0.0);

    ffo_parallel_for_static(work, 4, [&](int64_t begin, int64_t end, int tid) {
        om* gi_base = nullptr;
        if (ginp != nullptr) {
            gi_base = nthreads > 1 ? ginp_priv.data() + static_cast<size_t>(tid) * ginp_numel : nullptr;
        }
        auto scatter = [&](int64_t idx, om val) {
            if (gi_base != nullptr) gi_base[idx] += val;
            else ginp[idx] = static_cast<scalar_t>(static_cast<om>(ginp[idx]) + val);
        };
        double* ga = gaff_priv.data() + static_cast<size_t>(tid) * std::max<int64_t>(n_aff, 1) * 12;
        for (int64_t row = begin; row < end; ++row) {
            const int64_t h = row % g.H;
            const int64_t d = row / g.H;
            for (int64_t n = 0; n < g.N; ++n) {
                const scalar_t* inp_n = inp + n * g.inp_sN;
                const int64_t ginp_n = n * g.inp_sN;
                const scalar_t* gout_row = gout + n * g.out_sN + (g.W * (h + g.H * d)) * g.out_sS;
                for (int64_t w = 0; w < g.W; ++w) {
                    const Coord<gm> c = sampling_coordinate<grid_t, gm>(g, grid, aff, pre, n, d, h, w);
                    gm mx, my, mz;
                    const gm sx = grid_sampler_compute_source_index_set_grad(c.x, Wi, g.padding, g.align_corners, &mx);
                    const gm sy = grid_sampler_compute_source_index_set_grad(c.y, Hi, g.padding, g.align_corners, &my);
                    const gm sz = grid_sampler_compute_source_index_set_grad(c.z, Di, g.padding, g.align_corners, &mz);
                    const scalar_t* go = gout_row + w * g.out_sS;
                    if (!bilinear) {
                        if (ginp == nullptr) continue;
                        const int64_t xn = static_cast<int64_t>(std::nearbyint(sx));
                        const int64_t yn = static_cast<int64_t>(std::nearbyint(sy));
                        const int64_t zn = static_cast<int64_t>(std::nearbyint(sz));
                        if (xn >= 0 && yn >= 0 && zn >= 0 && xn < Wi && yn < Hi && zn < Di) {
                            for (int64_t ch = 0; ch < g.C; ++ch) {
                                scatter(ginp_n + ch * g.inp_sC + xn * sS + yn * sY + zn * sZ, static_cast<om>(go[ch * g.out_sC]));
                            }
                        }
                        continue;
                    }
                    const gm fx0 = std::floor(sx), fy0 = std::floor(sy), fz0 = std::floor(sz);
                    const int64_t x0 = static_cast<int64_t>(fx0), y0 = static_cast<int64_t>(fy0), z0 = static_cast<int64_t>(fz0);
                    const om fx = static_cast<om>(sx - fx0), fy = static_cast<om>(sy - fy0), fz = static_cast<om>(sz - fz0);
                    const om gx = 1 - fx, gy = 1 - fy, gz = 1 - fz;
                    const bool interior = x0 >= 0 && y0 >= 0 && z0 >= 0 && x0 + 1 < Wi && y0 + 1 < Hi && z0 + 1 < Di;
                    const int64_t base0 = x0 * sS + y0 * sY + z0 * sZ;
                    int64_t off[8];
                    bool ok[8];
                    if (!interior || ginp != nullptr) {
                        for (int k = 0; k < 8; ++k) {
                            const int64_t xi = x0 + (k & 1), yi = y0 + ((k >> 1) & 1), zi = z0 + (k >> 2);
                            ok[k] = interior || (xi >= 0 && yi >= 0 && zi >= 0 && xi < Wi && yi < Hi && zi < Di);
                            off[k] = xi * sS + yi * sY + zi * sZ;
                        }
                    }
                    om gix = 0, giy = 0, giz = 0;
                    for (int64_t ch = 0; ch < g.C; ++ch) {
                        const om gO = static_cast<om>(go[ch * g.out_sC]);
                        if (ginp != nullptr) {
                            const int64_t base = ginp_n + ch * g.inp_sC;
                            const om wt[8] = {gx * gy * gz, fx * gy * gz, gx * fy * gz, fx * fy * gz,
                                              gx * gy * fz, fx * gy * fz, gx * fy * fz, fx * fy * fz};
                            for (int k = 0; k < 8; ++k) {
                                if (ok[k]) scatter(base + off[k], wt[k] * gO);
                            }
                        }
                        if (need_spatial) {
                            const scalar_t* q = inp_n + ch * g.inp_sC;
                            om v[8];
                            if (interior) {
                                const scalar_t* p0 = q + base0;
                                v[0] = p0[0]; v[1] = p0[sS]; v[2] = p0[sY]; v[3] = p0[sY + sS];
                                v[4] = p0[sZ]; v[5] = p0[sZ + sS]; v[6] = p0[sZ + sY]; v[7] = p0[sZ + sY + sS];
                            } else {
                                for (int k = 0; k < 8; ++k) v[k] = ok[k] ? static_cast<om>(q[off[k]]) : om(0);
                            }
                            gix += gO * (gz * (gy * (v[1] - v[0]) + fy * (v[3] - v[2])) + fz * (gy * (v[5] - v[4]) + fy * (v[7] - v[6])));
                            giy += gO * (gz * (gx * (v[2] - v[0]) + fx * (v[3] - v[1])) + fz * (gx * (v[6] - v[4]) + fx * (v[7] - v[5])));
                            giz += gO * (gy * (gx * (v[4] - v[0]) + fx * (v[5] - v[1])) + fy * (gx * (v[6] - v[2]) + fx * (v[7] - v[3])));
                        }
                    }
                    if (!need_spatial) continue;
                    gm tx = static_cast<gm>(gix) * mx, ty = static_cast<gm>(giy) * my, tz = static_cast<gm>(giz) * mz;
                    if (gaff != nullptr) {
                        double* a = ga + (n_aff > 1 ? n : 0) * 12;
                        a[0] += tx * c.px; a[1] += tx * c.py; a[2] += tx * c.pz; a[3] += tx;
                        a[4] += ty * c.px; a[5] += ty * c.py; a[6] += ty * c.pz; a[7] += ty;
                        a[8] += tz * c.px; a[9] += tz * c.py; a[10] += tz * c.pz; a[11] += tz;
                    }
                    if (ggrid != nullptr) {
                        if (pre != nullptr) {
                            const grid_t* P = pre + n * g.pre_sN;
                            const gm ux = tx * P[0] + ty * P[3] + tz * P[6];
                            const gm uy = tx * P[1] + ty * P[4] + tz * P[7];
                            const gm uz = tx * P[2] + ty * P[5] + tz * P[8];
                            tx = ux; ty = uy; tz = uz;
                        }
                        grid_t* gg = ggrid + n * g.grid_sN + 3 * (w + g.W * (h + g.H * d));
                        gg[0] = static_cast<grid_t>(static_cast<gm>(gg[0]) + tx);
                        gg[1] = static_cast<grid_t>(static_cast<gm>(gg[1]) + ty);
                        gg[2] = static_cast<grid_t>(static_cast<gm>(gg[2]) + tz);
                    }
                }
            }
        }
    });

    if (ginp != nullptr && nthreads > 1) {
        ffo_parallel_for(ginp_numel, 1 << 16, [&](int64_t begin, int64_t end, int) {
            for (int64_t i = begin; i < end; ++i) {
                om acc = static_cast<om>(ginp[i]);
                for (int t = 0; t < nthreads; ++t) acc += ginp_priv[static_cast<size_t>(t) * ginp_numel + i];
                ginp[i] = static_cast<scalar_t>(acc);
            }
        });
    }
    if (gaff != nullptr) {
        const int64_t na = std::max<int64_t>(n_aff, 1);
        for (int64_t i = 0; i < na * 12; ++i) {
            double acc = 0;
            for (int t = 0; t < nthreads; ++t) acc += gaff_priv[static_cast<size_t>(t) * na * 12 + i];
            gaff[i] = static_cast<grid_t>(acc);
        }
    }
}

struct Broadcast {
    int64_t N;
    bool input, affine, grid, pre;
};

Broadcast batch_broadcast(int64_t n_input, const std::optional<torch::Tensor>& affine, const std::optional<torch::Tensor>& grid,
                          const std::optional<torch::Tensor>& pre) {
    Broadcast b{n_input, false, false, false, false};
    if (affine.has_value()) b.N = std::max(b.N, affine.value().size(0));
    if (grid.has_value()) b.N = std::max(b.N, grid.value().size(0));
    if (pre.has_value()) b.N = std::max(b.N, pre.value().size(0));
    if (b.N > 1) {
        auto check = [&](int64_t n, const char* name) {
            TORCH_CHECK(n == 1 || n == b.N, name, " batch size must be 1 or match the batch size of the other inputs");
            return n == 1;
        };
        b.input = check(n_input, "input");
        if (affine.has_value()) b.affine = check(affine.value().size(0), "affine");
        if (grid.has_value()) b.grid = check(grid.value().size(0), "grid");
        if (pre.has_value()) b.pre = check(pre.value().size(0), "grid affine");
    }
    return b;
}

void check_optional(const std::optional<torch::Tensor>& t, const char* name, int64_t d1, int64_t d2) {
    if (!t.has_value()) return;
    TORCH_CHECK(t.value().device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(t.value().is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.value().dim() == 3 && t.value().size(1) == d1 && t.value().size(2) == d2, name, " must be (B, ", d1, ", ", d2, ")");
}

template <typename T>
const T* optional_ptr(const std::optional<torch::Tensor>& t) {
    return t.has_value() ? t.value().data_ptr<T>() : nullptr;
}

Geometry make_geometry(const torch::Tensor& input, const Broadcast& b, int64_t D, int64_t H, int64_t W,
                       float xmin, float ymin, float zmin, float xmax, float ymax, float zmax,
                       bool is_displacement, int64_t interp, int64_t padding, bool align_corners) {
    Geometry g;
    g.N = b.N; g.C = input.size(1);
    g.Di = input.size(2); g.Hi = input.size(3); g.Wi = input.size(4);
    g.D = D; g.H = H; g.W = W;
    g.inp_sN = b.input ? 0 : g.C * g.Di * g.Hi * g.Wi;
    g.inp_sC = g.Di * g.Hi * g.Wi;
    g.inp_sS = 1;
    g.out_sN = g.C * D * H * W;
    g.out_sC = D * H * W;
    g.out_sS = 1;
    g.grid_sN = b.grid ? 0 : 3 * D * H * W;
    g.aff_sN = b.affine ? 0 : 12;
    g.pre_sN = b.pre ? 0 : 9;
    g.xmin = xmin; g.ymin = ymin; g.zmin = zmin; g.xmax = xmax; g.ymax = ymax; g.zmax = zmax;
    g.xstep = W > 1 ? (static_cast<double>(xmax) - xmin) / (W - 1) : 0.0;
    g.ystep = H > 1 ? (static_cast<double>(ymax) - ymin) / (H - 1) : 0.0;
    g.zstep = D > 1 ? (static_cast<double>(zmax) - zmin) / (D - 1) : 0.0;
    g.is_displacement = is_displacement;
    g.interp = static_cast<GridSamplerInterpolation>(interp);
    g.padding = static_cast<GridSamplerPadding>(padding);
    g.align_corners = align_corners;
    TORCH_CHECK(g.interp != GridSamplerInterpolation::Bicubic, "bicubic interpolation is not supported by the fused grid sampler");
    return g;
}

Geometry composer_geometry(const torch::Tensor& input, const Broadcast& b, int64_t D, int64_t H, int64_t W,
                           float xmin, float ymin, float zmin, float xmax, float ymax, float zmax, bool align_corners) {
    Geometry g;
    g.N = b.N; g.C = 3;
    g.Di = input.size(1); g.Hi = input.size(2); g.Wi = input.size(3);
    g.D = D; g.H = H; g.W = W;
    g.inp_sN = b.input ? 0 : 3 * g.Di * g.Hi * g.Wi;
    g.inp_sC = 1;
    g.inp_sS = 3;
    g.out_sN = 3 * D * H * W;
    g.out_sC = 1;
    g.out_sS = 3;
    g.grid_sN = b.grid ? 0 : 3 * D * H * W;
    g.aff_sN = b.affine ? 0 : 12;
    g.pre_sN = 0;
    g.xmin = xmin; g.ymin = ymin; g.zmin = zmin; g.xmax = xmax; g.ymax = ymax; g.zmax = zmax;
    g.xstep = W > 1 ? (static_cast<double>(xmax) - xmin) / (W - 1) : 0.0;
    g.ystep = H > 1 ? (static_cast<double>(ymax) - ymin) / (H - 1) : 0.0;
    g.zstep = D > 1 ? (static_cast<double>(zmax) - zmin) / (D - 1) : 0.0;
    g.is_displacement = true;
    g.interp = GridSamplerInterpolation::Bilinear;
    g.padding = GridSamplerPadding::Zeros;
    g.align_corners = align_corners;
    return g;
}

torch::Tensor scalar_type_of_grid(const torch::Tensor& input, const std::optional<torch::Tensor>& grid, const std::optional<torch::Tensor>& affine) {
    if (grid.has_value()) return grid.value();
    if (affine.has_value()) return affine.value();
    return input;
}

} // namespace

torch::Tensor fused_grid_sampler_3d_forward_cpu(
    const torch::Tensor &input,
    const std::optional<torch::Tensor> affine_3d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine,
    std::optional<torch::Tensor> output,
    const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    CHECK_CPU_INPUT(input);
    TORCH_CHECK(grid.has_value() || affine_3d.has_value(), "one of grid or affine_3d must exist");
    check_optional(affine_3d, "affine_3d", 3, 4);
    check_optional(grid_affine, "grid_affine", 3, 3);
    int64_t D = out_D, H = out_H, W = out_W;
    if (grid.has_value()) {
        CHECK_CPU_INPUT(grid.value());
        TORCH_CHECK(grid.value().dim() == 5 && grid.value().size(4) == 3, "grid must be (B, D, H, W, 3)");
        D = grid.value().size(1); H = grid.value().size(2); W = grid.value().size(3);
    }
    const Broadcast b = batch_broadcast(input.size(0), affine_3d, grid, grid_affine);
    const int64_t C = input.size(1);
    if (output.has_value()) {
        CHECK_CPU_INPUT(output.value());
        TORCH_CHECK(output.value().sizes() == torch::IntArrayRef({b.N, C, D, H, W}), "output must be (B, C, D, H, W)");
    } else {
        output.emplace(torch::zeros({b.N, C, D, H, W}, input.options()));
    }
    const Geometry g = make_geometry(input, b, D, H, W, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                                     is_displacement, interpolation_mode, padding_mode, align_corners);
    const auto grid_type = scalar_type_of_grid(input, grid, affine_3d).scalar_type();
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_sampler_3d_forward_cpu", [&] {
        using input_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grid_type, "fused_grid_sampler_3d_forward_cpu_grid", [&] {
            using grid_t = scalar_t;
            sampler_forward<input_t, grid_t>(g, input.data_ptr<input_t>(), optional_ptr<grid_t>(grid), optional_ptr<grid_t>(affine_3d),
                                             optional_ptr<grid_t>(grid_affine), output.value().data_ptr<input_t>());
        });
    });
    return output.value();
}

void fused_grid_sampler_3d_backward_cpu(
    const torch::Tensor &input,
    const std::optional<torch::Tensor> affine_3d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine,
    const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_input,
    const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    CHECK_CPU_INPUT(input);
    TORCH_CHECK(grid.has_value() || affine_3d.has_value(), "one of grid or affine_3d must exist");
    check_optional(affine_3d, "affine_3d", 3, 4);
    check_optional(grid_affine, "grid_affine", 3, 3);
    bool input_requires_grad = grad_input.has_value();
    bool affine_requires_grad = grad_affine.has_value() && affine_3d.has_value();
    bool grid_requires_grad = grad_grid.has_value() && grid.has_value();
    if (static_cast<GridSamplerInterpolation>(interpolation_mode) == GridSamplerInterpolation::Nearest) {
        affine_requires_grad = false;
        grid_requires_grad = false;
    }
    if (grid.has_value() && !is_displacement) {
        affine_requires_grad = false;
    }
    if (!input_requires_grad && !affine_requires_grad && !grid_requires_grad) return;
    int64_t D = out_D, H = out_H, W = out_W;
    if (grid.has_value()) {
        CHECK_CPU_INPUT(grid.value());
        D = grid.value().size(1); H = grid.value().size(2); W = grid.value().size(3);
    }
    const Broadcast b = batch_broadcast(input.size(0), affine_3d, grid, grid_affine);
    const torch::Tensor gout = grad_output.contiguous();
    TORCH_CHECK(gout.sizes() == torch::IntArrayRef({b.N, input.size(1), D, H, W}), "grad_output must be (B, C, D, H, W)");
    if (input_requires_grad) CHECK_CPU_INPUT(grad_input.value());
    if (grid_requires_grad) CHECK_CPU_INPUT(grad_grid.value());
    if (affine_requires_grad) CHECK_CPU_INPUT(grad_affine.value());
    const Geometry g = make_geometry(input, b, D, H, W, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                                     is_displacement, interpolation_mode, padding_mode, align_corners);
    const auto grid_type = scalar_type_of_grid(input, grid, affine_3d).scalar_type();
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_sampler_3d_backward_cpu", [&] {
        using input_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grid_type, "fused_grid_sampler_3d_backward_cpu_grid", [&] {
            using grid_t = scalar_t;
            sampler_backward<input_t, grid_t>(g, input.data_ptr<input_t>(), optional_ptr<grid_t>(grid), optional_ptr<grid_t>(affine_3d),
                optional_ptr<grid_t>(grid_affine), gout.data_ptr<input_t>(),
                input_requires_grad ? grad_input.value().data_ptr<input_t>() : nullptr,
                input_requires_grad ? grad_input.value().numel() : 0,
                affine_requires_grad ? grad_affine.value().data_ptr<grid_t>() : nullptr,
                affine_requires_grad ? affine_3d.value().size(0) : 0,
                grid_requires_grad ? grad_grid.value().data_ptr<grid_t>() : nullptr);
        });
    });
}

torch::Tensor fused_grid_composer_3d_forward_cpu(
    const torch::Tensor &input,
    const std::optional<torch::Tensor> affine_3d,
    const torch::Tensor grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    bool align_corners,
    std::optional<torch::Tensor> output) {
    TORCH_CHECK(input.dim() == 5 && input.size(4) == 3, "input must be (B, D, H, W, 3)");
    CHECK_CPU_INPUT(input);
    CHECK_CPU_INPUT(grid);
    TORCH_CHECK(grid.dim() == 5 && grid.size(4) == 3, "grid must be (B, D, H, W, 3)");
    TORCH_CHECK(grid.scalar_type() == input.scalar_type(), "input and grid must have the same dtype");
    check_optional(affine_3d, "affine_3d", 3, 4);
    const std::optional<torch::Tensor> no_pre;
    const Broadcast b = batch_broadcast(input.size(0), affine_3d, grid, no_pre);
    const int64_t D = grid.size(1), H = grid.size(2), W = grid.size(3);
    if (output.has_value()) {
        CHECK_CPU_INPUT(output.value());
        TORCH_CHECK(output.value().sizes() == torch::IntArrayRef({b.N, D, H, W, 3}), "output must be (B, D, H, W, 3)");
    } else {
        output.emplace(torch::zeros({b.N, D, H, W, 3}, input.options()));
    }
    const Geometry g = composer_geometry(input, b, D, H, W, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, align_corners);
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_composer_3d_forward_cpu", [&] {
        sampler_forward<scalar_t, scalar_t>(g, input.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(), optional_ptr<scalar_t>(affine_3d),
                                            nullptr, output.value().data_ptr<scalar_t>());
    });
    return output.value();
}

void fused_grid_composer_3d_backward_cpu(
    const torch::Tensor &input,
    const std::optional<torch::Tensor> affine_3d,
    const torch::Tensor grid,
    const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_input,
    const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    bool align_corners) {
    CHECK_CPU_INPUT(input);
    CHECK_CPU_INPUT(grid);
    check_optional(affine_3d, "affine_3d", 3, 4);
    const bool input_requires_grad = grad_input.has_value();
    const bool affine_requires_grad = grad_affine.has_value() && affine_3d.has_value();
    const bool grid_requires_grad = grad_grid.has_value();
    if (!input_requires_grad && !affine_requires_grad && !grid_requires_grad) return;
    const std::optional<torch::Tensor> no_pre;
    const Broadcast b = batch_broadcast(input.size(0), affine_3d, grid, no_pre);
    const int64_t D = grid.size(1), H = grid.size(2), W = grid.size(3);
    const torch::Tensor gout = grad_output.contiguous();
    TORCH_CHECK(gout.sizes() == torch::IntArrayRef({b.N, D, H, W, 3}), "grad_output must be (B, D, H, W, 3)");
    if (input_requires_grad) CHECK_CPU_INPUT(grad_input.value());
    if (grid_requires_grad) CHECK_CPU_INPUT(grad_grid.value());
    if (affine_requires_grad) CHECK_CPU_INPUT(grad_affine.value());
    const Geometry g = composer_geometry(input, b, D, H, W, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, align_corners);
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_composer_3d_backward_cpu", [&] {
        sampler_backward<scalar_t, scalar_t>(g, input.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(), optional_ptr<scalar_t>(affine_3d),
            nullptr, gout.data_ptr<scalar_t>(),
            input_requires_grad ? grad_input.value().data_ptr<scalar_t>() : nullptr,
            input_requires_grad ? grad_input.value().numel() : 0,
            affine_requires_grad ? grad_affine.value().data_ptr<scalar_t>() : nullptr,
            affine_requires_grad ? affine_3d.value().size(0) : 0,
            grid_requires_grad ? grad_grid.value().data_ptr<scalar_t>() : nullptr);
    });
}

torch::Tensor fused_warp_create_3d_forward_cpu(
    const std::optional<torch::Tensor> affine,
    const torch::Tensor grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax) {
    CHECK_CPU_INPUT(grid);
    TORCH_CHECK(grid.dim() == 5 && grid.size(4) == 3, "grid must be (B, D, H, W, 3)");
    check_optional(affine, "affine", 3, 4);
    const std::optional<torch::Tensor> no_pre;
    const Broadcast b = batch_broadcast(grid.size(0), affine, grid, no_pre);
    const int64_t D = grid.size(1), H = grid.size(2), W = grid.size(3);
    torch::Tensor output = torch::empty({b.N, D, H, W, 3}, grid.options());
    const int64_t grid_sN = b.grid ? 0 : 3 * D * H * W;
    const int64_t aff_sN = b.affine ? 0 : 12;
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grid.scalar_type(), "fused_warp_create_3d_forward_cpu", [&] {
        using gm = at::opmath_type<scalar_t>;
        const scalar_t* u = grid.data_ptr<scalar_t>();
        const scalar_t* Ab = optional_ptr<scalar_t>(affine);
        scalar_t* o = output.data_ptr<scalar_t>();
        ffo_parallel_for(b.N * D * H, 8, [&](int64_t begin, int64_t end, int) {
            for (int64_t row = begin; row < end; ++row) {
                const int64_t h = row % H, d = (row / H) % D, n = row / (H * D);
                const gm iy = H > 1 ? static_cast<gm>(h) * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin : static_cast<gm>(grid_ymin);
                const gm iz = D > 1 ? static_cast<gm>(d) * (grid_zmax - grid_zmin) / (D - 1) + grid_zmin : static_cast<gm>(grid_zmin);
                const scalar_t* A = Ab ? Ab + n * aff_sN : nullptr;
                for (int64_t w = 0; w < W; ++w) {
                    const gm ix = W > 1 ? static_cast<gm>(w) * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin : static_cast<gm>(grid_xmin);
                    const int64_t sp = 3 * (w + W * (h + H * d));
                    const scalar_t* uu = u + n * grid_sN + sp;
                    scalar_t* oo = o + n * 3 * D * H * W + sp;
                    if (A) {
                        oo[0] = static_cast<scalar_t>(A[0] * ix + A[1] * iy + A[2] * iz + A[3] + static_cast<gm>(uu[0]));
                        oo[1] = static_cast<scalar_t>(A[4] * ix + A[5] * iy + A[6] * iz + A[7] + static_cast<gm>(uu[1]));
                        oo[2] = static_cast<scalar_t>(A[8] * ix + A[9] * iy + A[10] * iz + A[11] + static_cast<gm>(uu[2]));
                    } else {
                        oo[0] = static_cast<scalar_t>(ix + static_cast<gm>(uu[0]));
                        oo[1] = static_cast<scalar_t>(iy + static_cast<gm>(uu[1]));
                        oo[2] = static_cast<scalar_t>(iz + static_cast<gm>(uu[2]));
                    }
                }
            }
        });
    });
    return output;
}

void fused_warp_create_3d_backward_cpu(
    const std::optional<torch::Tensor> affine,
    const torch::Tensor grid,
    const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax) {
    CHECK_CPU_INPUT(grid);
    check_optional(affine, "affine", 3, 4);
    const std::optional<torch::Tensor> no_pre;
    const Broadcast b = batch_broadcast(grid.size(0), affine, grid, no_pre);
    const int64_t D = grid.size(1), H = grid.size(2), W = grid.size(3);
    const torch::Tensor gout = grad_output.contiguous();
    const bool affine_requires_grad = grad_affine.has_value() && affine.has_value();
    if (grad_grid.has_value()) {
        CHECK_CPU_INPUT(grad_grid.value());
        if (b.grid) grad_grid.value().add_(gout.sum(0, true));
        else grad_grid.value().copy_(gout);
    }
    if (!affine_requires_grad) return;
    CHECK_CPU_INPUT(grad_affine.value());
    const int64_t n_aff = affine.value().size(0);
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grid.scalar_type(), "fused_warp_create_3d_backward_cpu", [&] {
        const scalar_t* go = gout.data_ptr<scalar_t>();
        const int nthreads = ffo_num_threads(D * H, 8);
        std::vector<double> acc(static_cast<size_t>(nthreads) * n_aff * 12, 0.0);
        ffo_parallel_for_static(D * H, 8, [&](int64_t begin, int64_t end, int tid) {
            double* a0 = acc.data() + static_cast<size_t>(tid) * n_aff * 12;
            for (int64_t row = begin; row < end; ++row) {
                const int64_t h = row % H, d = row / H;
                const double iy = H > 1 ? static_cast<double>(h) * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin : grid_ymin;
                const double iz = D > 1 ? static_cast<double>(d) * (grid_zmax - grid_zmin) / (D - 1) + grid_zmin : grid_zmin;
                for (int64_t n = 0; n < b.N; ++n) {
                    double* a = a0 + (n_aff > 1 ? n : 0) * 12;
                    for (int64_t w = 0; w < W; ++w) {
                        const double ix = W > 1 ? static_cast<double>(w) * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin : grid_xmin;
                        const scalar_t* g3 = go + n * 3 * D * H * W + 3 * (w + W * (h + H * d));
                        const double gx = g3[0], gy = g3[1], gz = g3[2];
                        a[0] += gx * ix; a[1] += gx * iy; a[2] += gx * iz; a[3] += gx;
                        a[4] += gy * ix; a[5] += gy * iy; a[6] += gy * iz; a[7] += gy;
                        a[8] += gz * ix; a[9] += gz * iy; a[10] += gz * iz; a[11] += gz;
                    }
                }
            }
        });
        scalar_t* ga = grad_affine.value().data_ptr<scalar_t>();
        for (int64_t i = 0; i < n_aff * 12; ++i) {
            double s = 0;
            for (int t = 0; t < nthreads; ++t) s += acc[static_cast<size_t>(t) * n_aff * 12 + i];
            ga[i] = static_cast<scalar_t>(s);
        }
    });
}
