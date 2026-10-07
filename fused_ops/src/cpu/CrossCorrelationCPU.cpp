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
CPU implementations of the fused local normalized cross-correlation kernels (see CrossCorrelation.cu),
and an in-place box (mean) filter that replaces the depthwise convolutions of the fused loss on CPU.
*/
#include <ATen/OpMathType.h>
#include <ATen/Dispatch.h>
#include <vector>
#include "cpu_utils.h"
#include "CPUOps.h"

namespace {

template <typename scalar_t>
void box_filter_lines(scalar_t* data, int64_t outer, int64_t L, int64_t r, double scale) {
    // mean filter along contiguous lines of length L
    ffo_parallel_for(outer, 64, [&](int64_t begin, int64_t end, int) {
        // zero-padded copy of the line: buf[r + l] = x[l]
        std::vector<double> buf(static_cast<size_t>(L + 2 * r), 0.0);
        for (int64_t o = begin; o < end; ++o) {
            scalar_t* x = data + o * L;
            for (int64_t l = 0; l < L; ++l) buf[r + l] = static_cast<double>(x[l]);
            double s = 0;
            for (int64_t j = 0; j < 2 * r; ++j) s += buf[j];
            for (int64_t l = 0; l < L; ++l) {
                s += buf[l + 2 * r];
                x[l] = static_cast<scalar_t>(s * scale);
                s -= buf[l];
            }
        }
    });
}

template <typename scalar_t>
void box_filter_strided(scalar_t* data, int64_t outer, int64_t L, int64_t inner, int64_t r, double scale) {
    // mean filter along an axis of length L with stride `inner`, vectorized over the contiguous inner dimension
    const int64_t chunk = std::min<int64_t>(inner, 2048);
    const int64_t nchunks = (inner + chunk - 1) / chunk;
    ffo_parallel_for(outer * nchunks, 1, [&](int64_t begin, int64_t end, int) {
        std::vector<double> sum(static_cast<size_t>(chunk));
        std::vector<double> ring(static_cast<size_t>((r + 1) * chunk));
        for (int64_t u = begin; u < end; ++u) {
            const int64_t o = u / nchunks;
            const int64_t i0 = (u % nchunks) * chunk;
            const int64_t width = std::min(chunk, inner - i0);
            scalar_t* base = data + o * L * inner + i0;
            std::fill(sum.begin(), sum.begin() + width, 0.0);
            for (int64_t j = 0; j < std::min(r, L); ++j) {
                const scalar_t* xj = base + j * inner;
                for (int64_t i = 0; i < width; ++i) sum[i] += static_cast<double>(xj[i]);
            }
            for (int64_t l = 0; l < L; ++l) {
                scalar_t* xl = base + l * inner;
                if (l + r < L) {
                    const scalar_t* xa = base + (l + r) * inner;
                    for (int64_t i = 0; i < width; ++i) sum[i] += static_cast<double>(xa[i]);
                }
                double* slot = ring.data() + (l % (r + 1)) * chunk;
                if (l - r - 1 >= 0) {
                    for (int64_t i = 0; i < width; ++i) sum[i] -= slot[i];
                }
                for (int64_t i = 0; i < width; ++i) {
                    slot[i] = static_cast<double>(xl[i]);
                    xl[i] = static_cast<scalar_t>(sum[i] * scale);
                }
            }
        }
    });
}

} // namespace

void box_filter_3d_cpu(torch::Tensor &input, int64_t kernel_size) {
    CHECK_CPU_INPUT(input);
    TORCH_CHECK(input.dim() == 5, "input must be (B, C, D, H, W)");
    TORCH_CHECK(kernel_size >= 1 && kernel_size % 2 == 1, "kernel_size must be a positive odd integer");
    if (kernel_size == 1 || input.numel() == 0) return;
    const int64_t BC = input.size(0) * input.size(1);
    const int64_t D = input.size(2), H = input.size(3), W = input.size(4);
    const int64_t r = (kernel_size - 1) / 2;
    const double scale = 1.0 / static_cast<double>(kernel_size);
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "box_filter_3d_cpu", [&] {
        scalar_t* p = input.data_ptr<scalar_t>();
        box_filter_strided<scalar_t>(p, BC, D, H * W, r, scale);
        box_filter_strided<scalar_t>(p, BC * D, H, W, r, scale);
        box_filter_lines<scalar_t>(p, BC * D * H, W, r, scale);
    });
}

torch::Tensor create_intermediates_cpu(torch::Tensor input_img, torch::Tensor target_img, torch::Tensor intermediates) {
    CHECK_CPU_INPUT(input_img);
    CHECK_CPU_INPUT(target_img);
    CHECK_CPU_INPUT(intermediates);
    TORCH_CHECK(input_img.dim() == 5, "only 5D inputs are supported");
    const int64_t B = input_img.size(0), C = input_img.size(1);
    const int64_t S = input_img.numel() / (B * C);
    TORCH_CHECK(intermediates.numel() == 5 * input_img.numel(), "intermediates must be (B, 5C, ...)");
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input_img.scalar_type(), "create_intermediates_cpu", [&] {
        using om = at::opmath_type<scalar_t>;
        ffo_parallel_planes(B * C, S, [&](int64_t p, int64_t s0, int64_t s1, int) {
            const int64_t b = p / C, c = p % C;
            const scalar_t* I = input_img.data_ptr<scalar_t>() + p * S;
            const scalar_t* J = target_img.data_ptr<scalar_t>() + p * S;
            scalar_t* o = intermediates.data_ptr<scalar_t>() + b * 5 * C * S + c * S;
            scalar_t* o1 = o + C * S;
            scalar_t* o2 = o + 2 * C * S;
            scalar_t* o3 = o + 3 * C * S;
            scalar_t* o4 = o + 4 * C * S;
            for (int64_t s = s0; s < s1; ++s) {
                const om iv = I[s], jv = J[s];
                o[s] = static_cast<scalar_t>(iv);
                o1[s] = static_cast<scalar_t>(jv);
                o2[s] = static_cast<scalar_t>(iv * iv);
                o3[s] = static_cast<scalar_t>(jv * jv);
                o4[s] = static_cast<scalar_t>(iv * jv);
            }
        });
    });
    return intermediates;
}

torch::Tensor cc3d_fwd_interm_v1_cpu(torch::Tensor intermediates, int64_t kernel_volume, Reduction reduction, float nr, float dr) {
    CHECK_CPU_INPUT(intermediates);
    const int64_t B = intermediates.size(0), C5 = intermediates.size(1);
    TORCH_CHECK(C5 % 5 == 0, "intermediates must have 5C channels");
    const int64_t C = C5 / 5;
    const int64_t S = intermediates.numel() / (B * C5);
    const int64_t n = B * C * S;
    std::vector<int64_t> out_shape = intermediates.sizes().vec();
    out_shape[1] = C;
    torch::Tensor out;
    if (reduction == Reduction::NONE) out = torch::empty(out_shape, intermediates.options());
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, intermediates.scalar_type(), "cc3d_fwd_interm_v1_cpu", [&] {
        using om = at::opmath_type<scalar_t>;
        const om kv = static_cast<om>(kernel_volume);
        const om nr_ = static_cast<om>(nr), dr_ = static_cast<om>(dr);
        std::vector<double> partial(static_cast<size_t>(ffo_plane_units(B * C, S)), 0.0);
        ffo_parallel_planes(B * C, S, [&](int64_t p, int64_t s0, int64_t s1, int64_t unit) {
            const int64_t b = p / C, c = p % C;
            const scalar_t* x0 = intermediates.data_ptr<scalar_t>() + b * 5 * C * S + c * S;
            const scalar_t* x1 = x0 + C * S;
            const scalar_t* x2 = x0 + 2 * C * S;
            const scalar_t* x3 = x0 + 3 * C * S;
            const scalar_t* x4 = x0 + 4 * C * S;
            scalar_t* o = reduction == Reduction::NONE ? out.data_ptr<scalar_t>() + p * S : nullptr;
            double acc = 0;
            for (int64_t s = s0; s < s1; ++s) {
                const om I = x0[s], J = x1[s], I2 = x2[s], J2 = x3[s], IJ = x4[s];
                const om a = kv * (IJ - I * J);
                const om bb = std::max(kv * (I2 - I * I), dr_);
                const om cc = std::max(kv * (J2 - J * J), dr_);
                om v = (a * a + nr_) / (bb * cc + dr_);
                v = std::min(std::max(v, om(-1)), om(1));
                if (o) o[s] = static_cast<scalar_t>(v);
                else acc += static_cast<double>(v);
            }
            partial[unit] = acc;
        });
        if (reduction != Reduction::NONE) {
            double total = 0;
            for (double v : partial) total += v;
            if (reduction == Reduction::MEAN) total /= static_cast<double>(n);
            out = torch::scalar_tensor(total, intermediates.options());
        }
    });
    return out;
}

torch::Tensor cc3d_bwd_modify_interm_v1_cpu(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img,
    torch::Tensor grad_output,
    std::optional<torch::Tensor> grad_input_img, std::optional<torch::Tensor> grad_target_img,
    int64_t kernel_size, int64_t kernel_volume, float nr, float dr, Reduction reduction) {
    TORCH_CHECK(grad_input_img.has_value(), "grad_input_img must be provided");
    CHECK_CPU_INPUT(intermediates);
    const bool compute_grad_target = grad_target_img.has_value();
    const int64_t B = intermediates.size(0), C = intermediates.size(1) / 5;
    const int64_t S = intermediates.numel() / (B * 5 * C);
    const torch::Tensor go = grad_output.contiguous();
    if (reduction == Reduction::NONE) TORCH_CHECK(go.numel() == B * C * S, "grad_output must have one entry per output element");
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, intermediates.scalar_type(), "cc3d_bwd_modify_interm_v1_cpu", [&] {
        using om = at::opmath_type<scalar_t>;
        const om g_scalar = reduction == Reduction::NONE ? om(1) : static_cast<om>(go.item<scalar_t>());
        const om kv = static_cast<om>(kernel_volume);
        const om dr_ = static_cast<om>(dr);
        ffo_parallel_planes(B * C, S, [&](int64_t p, int64_t s0, int64_t s1, int) {
            const int64_t b = p / C, c = p % C;
            scalar_t* x0 = intermediates.data_ptr<scalar_t>() + b * 5 * C * S + c * S;
            scalar_t* x1 = x0 + C * S;
            scalar_t* x2 = x0 + 2 * C * S;
            scalar_t* x3 = x0 + 3 * C * S;
            scalar_t* x4 = x0 + 4 * C * S;
            const scalar_t* g = reduction == Reduction::NONE ? go.data_ptr<scalar_t>() + p * S : nullptr;
            for (int64_t s = s0; s < s1; ++s) {
                const om I = x0[s], J = x1[s], I2 = x2[s], J2 = x3[s], IJ = x4[s];
                const om a = kv * (IJ - I * J);
                om bb = kv * (I2 - I * I);
                om cc = kv * (J2 - J * J);
                om gv = g ? static_cast<om>(g[s]) : g_scalar;
                gv = 2 * gv * a / (bb * cc + dr_);
                bb += dr_;
                cc += dr_;
                x0[s] = static_cast<scalar_t>(gv);
                x1[s] = static_cast<scalar_t>(gv * a / bb);
                x2[s] = static_cast<scalar_t>(gv * (a / bb * I - J));
                if (compute_grad_target) {
                    x3[s] = static_cast<scalar_t>(gv * a / cc);
                    x4[s] = static_cast<scalar_t>(gv * (a / cc * J - I));
                }
            }
        });
    });
    return torch::Tensor();
}

void cc3d_bwd_compute_grads_cpu(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img,
                                torch::Tensor grad_input_img, std::optional<torch::Tensor> grad_target_img) {
    CHECK_CPU_INPUT(intermediates);
    CHECK_CPU_INPUT(input_img);
    CHECK_CPU_INPUT(target_img);
    CHECK_CPU_INPUT(grad_input_img);
    const int64_t B = intermediates.size(0), C = intermediates.size(1) / 5;
    const int64_t S = intermediates.numel() / (B * 5 * C);
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, intermediates.scalar_type(), "cc3d_bwd_compute_grads_cpu", [&] {
        using om = at::opmath_type<scalar_t>;
        ffo_parallel_planes(B * C, S, [&](int64_t p, int64_t s0, int64_t s1, int) {
            const int64_t b = p / C, c = p % C;
            const scalar_t* x0 = intermediates.data_ptr<scalar_t>() + b * 5 * C * S + c * S;
            const scalar_t* x1 = x0 + C * S;
            const scalar_t* x2 = x0 + 2 * C * S;
            const scalar_t* x3 = x0 + 3 * C * S;
            const scalar_t* x4 = x0 + 4 * C * S;
            const scalar_t* Ip = input_img.data_ptr<scalar_t>() + p * S;
            const scalar_t* Jp = target_img.data_ptr<scalar_t>() + p * S;
            scalar_t* gi = grad_input_img.data_ptr<scalar_t>() + p * S;
            scalar_t* gt = grad_target_img.has_value() ? grad_target_img.value().data_ptr<scalar_t>() + p * S : nullptr;
            for (int64_t s = s0; s < s1; ++s) {
                const om I = Ip[s], J = Jp[s];
                gi[s] = static_cast<scalar_t>(static_cast<om>(x0[s]) * J - static_cast<om>(x1[s]) * I + static_cast<om>(x2[s]));
            }
            if (gt) {
                for (int64_t s = s0; s < s1; ++s) {
                    const om I = Ip[s], J = Jp[s];
                    gt[s] = static_cast<scalar_t>(static_cast<om>(x0[s]) * I - static_cast<om>(x3[s]) * J + static_cast<om>(x4[s]));
                }
            }
        });
    });
}
