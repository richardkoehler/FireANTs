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


// CPU implementations of AdamUtils.cu and GaussianBlurFFT.cu
#include <ATen/OpMathType.h>
#include <ATen/Dispatch.h>
#include <c10/util/complex.h>
#include <cmath>
#include "cpu_utils.h"
#include "CPUOps.h"

void adam_update_fused_cpu(torch::Tensor &grad, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, float beta1, float beta2, float eps) {
    CHECK_CPU_INPUT(grad);
    CHECK_CPU_INPUT(exp_avg);
    CHECK_CPU_INPUT(exp_avg_sq);
    const int64_t n = grad.numel();
    TORCH_CHECK(exp_avg.numel() == n && exp_avg_sq.numel() == n, "grad, exp_avg and exp_avg_sq must have the same size");
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(), "adam_update_fused_cpu", [&] {
        using om = at::opmath_type<scalar_t>;
        scalar_t* g = grad.data_ptr<scalar_t>();
        const scalar_t* m = exp_avg.data_ptr<scalar_t>();
        const scalar_t* v = exp_avg_sq.data_ptr<scalar_t>();
        const om b1 = beta1, b2 = beta2, e = eps;
        ffo_parallel_for(n, 1 << 16, [&](int64_t begin, int64_t end, int) {
            for (int64_t i = begin; i < end; ++i) {
                g[i] = static_cast<scalar_t>(static_cast<om>(m[i]) / b1 / (std::sqrt(static_cast<om>(v[i]) / b2) + e));
            }
        });
    });
}

namespace {

template <typename scalar_t>
inline void scale_value(scalar_t& x, double s) {
    x = static_cast<scalar_t>(x * s);
}

template <typename T>
inline void scale_value(c10::complex<T>& x, double s) {
    x = c10::complex<T>(static_cast<T>(x.real() * s), static_cast<T>(x.imag() * s));
}

} // namespace

void gaussian_blur_fft2_cpu(torch::Tensor &im_fft, int64_t ys, int64_t xs, int64_t ye, int64_t xe, float multiplier) {
    CHECK_CPU_INPUT(im_fft);
    const int64_t BC = im_fft.size(0) * im_fft.size(1), H = im_fft.size(2), W = im_fft.size(3);
    const float sigma_y = static_cast<float>(H) / 4.0f, sigma_x = static_cast<float>(W) / 4.0f;
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(im_fft.scalar_type(), "gaussian_blur_fft2_cpu", [&] {
        scalar_t* p = im_fft.data_ptr<scalar_t>();
        ffo_parallel_for(BC * H, 16, [&](int64_t begin, int64_t end, int) {
            for (int64_t row = begin; row < end; ++row) {
                const int64_t h = row % H;
                const float fy = static_cast<float>(ys + h) / sigma_y;
                for (int64_t w = 0; w < W; ++w) {
                    const float fx = static_cast<float>(xs + w) / sigma_x;
                    scale_value(p[row * W + w], std::exp(-0.5 * (fy * fy + fx * fx)) * multiplier);
                }
            }
        });
    });
}

void gaussian_blur_fft3_cpu(torch::Tensor &im_fft, int64_t zs, int64_t ys, int64_t xs, int64_t ze, int64_t ye, int64_t xe, float multiplier) {
    CHECK_CPU_INPUT(im_fft);
    const int64_t BC = im_fft.size(0) * im_fft.size(1), D = im_fft.size(2), H = im_fft.size(3), W = im_fft.size(4);
    const float sigma_z = static_cast<float>(D) / 4.0f, sigma_y = static_cast<float>(H) / 4.0f, sigma_x = static_cast<float>(W) / 4.0f;
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(im_fft.scalar_type(), "gaussian_blur_fft3_cpu", [&] {
        scalar_t* p = im_fft.data_ptr<scalar_t>();
        ffo_parallel_for(BC * D * H, 16, [&](int64_t begin, int64_t end, int) {
            for (int64_t row = begin; row < end; ++row) {
                const int64_t h = row % H, d = (row / H) % D;
                const float fz = static_cast<float>(zs + d) / sigma_z;
                const float fy = static_cast<float>(ys + h) / sigma_y;
                for (int64_t w = 0; w < W; ++w) {
                    const float fx = static_cast<float>(xs + w) / sigma_x;
                    scale_value(p[row * W + w], std::exp(-0.5 * (fy * fy + fx * fx + fz * fz)) * multiplier);
                }
            }
        });
    });
}
