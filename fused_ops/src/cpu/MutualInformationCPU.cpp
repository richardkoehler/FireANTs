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
CPU implementation of the fused mutual information histogram kernels (see MutualInformation.cu).

The Parzen window of a sample is only evaluated on the bins around the sample intensity: the B-spline and
delta kernels have compact support, and the Gaussian kernel is truncated at 6 standard deviations, where
its relative weight (exp(-18)) is below single precision. This keeps the cost per sample independent of
the number of bins instead of quadratic in it.
*/
#include <ATen/OpMathType.h>
#include <ATen/Dispatch.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "cpu_utils.h"
#include "CPUOps.h"

namespace {

struct ParzenKernel {
    int num_bins;
    KernelType type;
    float bin_width, sigma;
    int half_width;

    ParzenKernel(int nb, KernelType t, float sigma_ratio) : num_bins(nb), type(t) {
        bin_width = 1.0f / nb;
        sigma = bin_width / 2 * sigma_ratio;
        if (t == KernelType::GAUSSIAN) half_width = static_cast<int>(std::ceil(6.0f * sigma / bin_width)) + 1;
        else if (t == KernelType::BSPLINE) half_width = 3;
        else half_width = 2;
    }

    inline float prob(float x, int bin) const {
        const float center = (bin + 0.5f) * bin_width;
        const float abdiff = std::fabs((x - center) / bin_width);
        switch (type) {
            case KernelType::GAUSSIAN: {
                const float t = (x - center) / sigma;
                return std::exp(-0.5f * t * t);
            }
            case KernelType::BSPLINE:
                if (abdiff < 1) return (4 + abdiff * abdiff * (-6 + 3 * abdiff)) / 6.0f;
                if (abdiff < 2) return (2 - abdiff) * (2 - abdiff) * (2 - abdiff) / 6.0f;
                return 0;
            case KernelType::DELTA:
                return std::fabs(x - center) < bin_width ? 1.0f : 0.0f;
            default:
                return 0;
        }
    }

    inline float grad(float x, int bin) const {
        const float center = (bin + 0.5f) * bin_width;
        const float abdiff = std::fabs((x - center) / bin_width);
        switch (type) {
            case KernelType::GAUSSIAN:
                return prob(x, bin) * (center - x) / sigma / sigma;
            case KernelType::BSPLINE:
                if (abdiff < 1) return (-2 + 1.5f * abdiff) * abdiff;
                if (abdiff < 2) return -(2 - abdiff) * (2 - abdiff) / 2;
                return 0;
            case KernelType::DELTA:
                return std::fabs(x - center) < bin_width ? 1.0f : 0.0f;
            default:
                return 0;
        }
    }

    // bins [lo, hi] that can have a non-negligible weight for intensity x
    inline void band(float x, int& lo, int& hi) const {
        int b = static_cast<int>(std::floor(x * num_bins));
        b = std::min(std::max(b, 0), num_bins - 1);
        lo = std::max(b - half_width, 0);
        hi = std::min(b + half_width, num_bins - 1);
    }

    // normalized weights w[bin - lo] and (optionally) their derivatives, returns the number of bins
    inline int weights(float x, int& lo, float* w, float* dw) const {
        int hi;
        band(x, lo, hi);
        const int nb = hi - lo + 1;
        float s = 0, sd = 0;
        for (int k = 0; k < nb; ++k) {
            w[k] = prob(x, lo + k);
            s += w[k];
        }
        if (dw) {
            for (int k = 0; k < nb; ++k) {
                if (type == KernelType::GAUSSIAN) dw[k] = w[k] * ((lo + k + 0.5f) * bin_width - x) / sigma / sigma;
                else dw[k] = grad(x, lo + k);
                sd += dw[k];
            }
        }
        if (dw) {
            for (int k = 0; k < nb; ++k) dw[k] = (dw[k] - w[k] * sd / s) / s;
        }
        for (int k = 0; k < nb; ++k) w[k] /= s;
        return nb;
    }
};

} // namespace

std::vector<torch::Tensor> mutual_information_histogram_fwd_cpu(torch::Tensor &input_img, torch::Tensor &target_img, int num_bins,
    KernelType kernel_type, float minval, float maxval, float sigma_ratio, bool approximate_reduction) {
    CHECK_CPU_INPUT(input_img);
    CHECK_CPU_INPUT(target_img);
    TORCH_CHECK(num_bins > 0, "num_bins must be positive");
    TORCH_CHECK(input_img.sizes() == target_img.sizes(), "input and target must have the same shape");
    const int64_t B = input_img.size(0), C = input_img.size(1);
    const int64_t S = input_img.numel() / (B * C);
    const int64_t nb = num_bins;
    const int64_t hist_size = nb * nb + 2 * nb;
    const ParzenKernel kernel(num_bins, kernel_type, sigma_ratio);
    const int nthreads_max = std::max(at::get_num_threads(), 1);
    std::vector<double> hist(static_cast<size_t>(nthreads_max) * B * C * hist_size, 0.0);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_img.scalar_type(), "mutual_information_histogram_fwd_cpu", [&] {
        const scalar_t* I = input_img.data_ptr<scalar_t>();
        const scalar_t* J = target_img.data_ptr<scalar_t>();
        const float range = maxval - minval;
        ffo_parallel_planes_static(B * C, S, [&](int64_t p, int64_t s0, int64_t s1, int tid) {
            double* h = hist.data() + (static_cast<size_t>(tid) * B * C + p) * hist_size;
            double* pab = h;
            double* pa = h + nb * nb;
            double* pb = pa + nb;
            const scalar_t* Ip = I + p * S;
            const scalar_t* Jp = J + p * S;
            if (approximate_reduction) {
                for (int64_t s = s0; s < s1; ++s) {
                    const float iv = (static_cast<float>(Ip[s]) - minval) / range;
                    const float jv = (static_cast<float>(Jp[s]) - minval) / range;
                    int64_t ib = static_cast<int64_t>(std::floor(iv * nb));
                    int64_t jb = static_cast<int64_t>(std::floor(jv * nb));
                    ib = std::min(std::max(ib, int64_t(0)), nb - 1);
                    jb = std::min(std::max(jb, int64_t(0)), nb - 1);
                    pab[jb + nb * ib] += 1;
                    pa[ib] += 1;
                    pb[jb] += 1;
                }
            } else {
                std::vector<float> wa(nb), wb(nb);
                for (int64_t s = s0; s < s1; ++s) {
                    const float iv = (static_cast<float>(Ip[s]) - minval) / range;
                    const float jv = (static_cast<float>(Jp[s]) - minval) / range;
                    int la, lb;
                    const int na = kernel.weights(iv, la, wa.data(), nullptr);
                    const int nbj = kernel.weights(jv, lb, wb.data(), nullptr);
                    for (int i = 0; i < na; ++i) {
                        pa[la + i] += wa[i];
                        double* row = pab + nb * (la + i) + lb;
                        for (int j = 0; j < nbj; ++j) row[j] += static_cast<double>(wa[i]) * wb[j];
                    }
                    for (int j = 0; j < nbj; ++j) pb[lb + j] += wb[j];
                }
            }
        });
    });
    torch::Tensor pab = torch::zeros({B, C, nb, nb}, input_img.options().dtype(torch::kDouble));
    torch::Tensor pa = torch::zeros({B, C, nb}, pab.options());
    torch::Tensor pb = torch::zeros({B, C, nb}, pab.options());
    double* pab_p = pab.data_ptr<double>();
    double* pa_p = pa.data_ptr<double>();
    double* pb_p = pb.data_ptr<double>();
    const double inv_n = 1.0 / static_cast<double>(S);
    for (int t = 0; t < nthreads_max; ++t) {
        for (int64_t p = 0; p < B * C; ++p) {
            const double* h = hist.data() + (static_cast<size_t>(t) * B * C + p) * hist_size;
            for (int64_t k = 0; k < nb * nb; ++k) pab_p[p * nb * nb + k] += h[k] * inv_n;
            for (int64_t k = 0; k < nb; ++k) {
                pa_p[p * nb + k] += h[nb * nb + k] * inv_n;
                pb_p[p * nb + k] += h[nb * nb + nb + k] * inv_n;
            }
        }
    }
    const auto dtype = input_img.scalar_type();
    return {pab.to(dtype), pa.to(dtype), pb.to(dtype)};
}

void mutual_information_histogram_bwd_cpu(torch::Tensor &input_img, torch::Tensor &target_img, torch::Tensor &grad_pab,
    torch::Tensor &grad_pa, torch::Tensor &grad_pb, int num_bins, torch::Tensor &grad_input_img,
    std::optional<torch::Tensor> &grad_target_img, KernelType kernel_type, float minval, float maxval, float sigma_ratio) {
    CHECK_CPU_INPUT(input_img);
    CHECK_CPU_INPUT(target_img);
    CHECK_CPU_INPUT(grad_input_img);
    if (grad_target_img.has_value()) CHECK_CPU_INPUT(grad_target_img.value());
    const int64_t B = input_img.size(0), C = input_img.size(1);
    const int64_t S = input_img.numel() / (B * C);
    const int64_t nb = num_bins;
    const ParzenKernel kernel(num_bins, kernel_type, sigma_ratio);
    const torch::Tensor gpab_f = grad_pab.to(torch::kFloat).contiguous();
    const torch::Tensor gpa_f = grad_pa.to(torch::kFloat).contiguous();
    const torch::Tensor gpb_f = grad_pb.to(torch::kFloat).contiguous();
    const bool target_grad = grad_target_img.has_value();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_img.scalar_type(), "mutual_information_histogram_bwd_cpu", [&] {
        const scalar_t* I = input_img.data_ptr<scalar_t>();
        const scalar_t* J = target_img.data_ptr<scalar_t>();
        scalar_t* gI = grad_input_img.data_ptr<scalar_t>();
        scalar_t* gJ = target_grad ? grad_target_img.value().data_ptr<scalar_t>() : nullptr;
        const float range = maxval - minval;
        const float inv_n = 1.0f / static_cast<float>(S);
        ffo_parallel_planes(B * C, S, [&](int64_t p, int64_t s0, int64_t s1, int) {
            const float* G = gpab_f.data_ptr<float>() + p * nb * nb;
            const float* ga = gpa_f.data_ptr<float>() + p * nb;
            const float* gb = gpb_f.data_ptr<float>() + p * nb;
            std::vector<float> wa(nb), dwa(nb), wb(nb), dwb(nb);
            const scalar_t* Ip = I + p * S;
            const scalar_t* Jp = J + p * S;
            for (int64_t s = s0; s < s1; ++s) {
                const float iv = (static_cast<float>(Ip[s]) - minval) / range;
                const float jv = (static_cast<float>(Jp[s]) - minval) / range;
                int la, lb;
                const int na = kernel.weights(iv, la, wa.data(), dwa.data());
                const int nbj = kernel.weights(jv, lb, wb.data(), target_grad ? dwb.data() : nullptr);
                float gi = 0;
                for (int i = 0; i < na; ++i) {
                    const float* row = G + nb * (la + i) + lb;
                    float v = ga[la + i];
                    for (int j = 0; j < nbj; ++j) v += row[j] * wb[j];
                    gi += dwa[i] * v;
                }
                gI[p * S + s] = static_cast<scalar_t>(gi * inv_n);
                if (gJ) {
                    float gj = 0;
                    for (int j = 0; j < nbj; ++j) {
                        float u = gb[lb + j];
                        for (int i = 0; i < na; ++i) u += G[nb * (la + i) + lb + j] * wa[i];
                        gj += dwb[j] * u;
                    }
                    gJ[p * S + s] = static_cast<scalar_t>(gj * inv_n);
                }
            }
        });
    });
}
