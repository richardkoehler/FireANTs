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


#pragma once
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CPU_INPUT(x) do { CHECK_CPU(x); TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); } while (0)

// number of threads that ffo_parallel_for will use for `work` items with chunks of at least `grain` items
inline int ffo_num_threads(int64_t work, int64_t grain) {
#ifdef _OPENMP
    if (omp_in_parallel()) return 1;
    int64_t chunks = (work + grain - 1) / std::max<int64_t>(grain, 1);
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(at::get_num_threads(), chunks)));
#else
    return 1;
#endif
}

// Runs f(begin, end, thread_id) over [0, work) on the PyTorch intra-op thread count
// (at::get_num_threads()) with dynamic scheduling, which keeps hybrid (performance + efficiency core) CPUs busy.
template <typename F>
inline void ffo_parallel_for(int64_t work, int64_t grain, const F& f) {
    if (work <= 0) return;
    const int nthreads = ffo_num_threads(work, grain);
    if (nthreads <= 1) {
        f(static_cast<int64_t>(0), work, 0);
        return;
    }
#ifdef _OPENMP
    const int64_t nchunks = std::min<int64_t>((work + grain - 1) / grain, static_cast<int64_t>(nthreads) * 16);
    const int64_t chunk = (work + nchunks - 1) / nchunks;
    const int nchunks_i = static_cast<int>((work + chunk - 1) / chunk);
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (int ci = 0; ci < nchunks_i; ++ci) {
        const int64_t b = static_cast<int64_t>(ci) * chunk;
        const int64_t e = std::min(work, b + chunk);
        f(b, e, omp_get_thread_num());
    }
#endif
}

inline int64_t ffo_plane_units(int64_t planes, int64_t plane_size) {
    return planes * std::max<int64_t>(1, (plane_size + (1 << 14) - 1) / (1 << 14));
}

template <typename F>
inline void ffo_parallel_planes(int64_t planes, int64_t plane_size, const F& f) {
    const int64_t chunk = 1 << 14;
    const int64_t per_plane = std::max<int64_t>(1, (plane_size + chunk - 1) / chunk);
    ffo_parallel_for(planes * per_plane, 1, [&](int64_t begin, int64_t end, int) {
        for (int64_t u = begin; u < end; ++u) {
            const int64_t p = u / per_plane;
            const int64_t s0 = (u % per_plane) * chunk;
            const int64_t s1 = std::min(plane_size, s0 + chunk);
            f(p, s0, s1, u);
        }
    });
}

template <typename F>
inline void ffo_parallel_for_static(int64_t work, int64_t grain, const F& f) {
    if (work <= 0) return;
    const int nbuffers = ffo_num_threads(work, grain);
    if (nbuffers <= 1) {
        f(static_cast<int64_t>(0), work, 0);
        return;
    }
#ifdef _OPENMP
    const int64_t nchunks = std::min<int64_t>((work + grain - 1) / grain, static_cast<int64_t>(nbuffers) * 16);
    const int64_t chunk = (work + nchunks - 1) / nchunks;
    const int64_t nchunks_i = (work + chunk - 1) / chunk;
    #pragma omp parallel for num_threads(nbuffers) schedule(static, 1)
    for (int t = 0; t < nbuffers; ++t) {
        for (int64_t ci = t; ci < nchunks_i; ci += nbuffers) {
            const int64_t b = ci * chunk;
            f(b, std::min(work, b + chunk), t);
        }
    }
#endif
}

template <typename F>
inline void ffo_parallel_planes_static(int64_t planes, int64_t plane_size, const F& f) {
    const int64_t chunk = 1 << 14;
    const int64_t per_plane = std::max<int64_t>(1, (plane_size + chunk - 1) / chunk);
    ffo_parallel_for_static(planes * per_plane, 1, [&](int64_t begin, int64_t end, int buffer) {
        for (int64_t u = begin; u < end; ++u) {
            const int64_t p = u / per_plane;
            const int64_t s0 = (u % per_plane) * chunk;
            const int64_t s1 = std::min(plane_size, s0 + chunk);
            f(p, s0, s1, buffer);
        }
    });
}
