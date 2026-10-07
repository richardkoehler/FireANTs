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


#include <stdexcept>
#include <string>
#include "DeviceDispatch.h"
#include "CPUOps.h"
#include "FusedGridSampler.h"
#include "FusedGridSamplerGenericLabel.h"
#include "FusedGridComposer.h"
#include "FusedGenerateGrid.h"
#include "GaussianBlurFFT.h"
#include "common.h"

namespace {

[[noreturn]] void not_on_cpu(const char* op) {
    throw std::runtime_error(std::string("fireants_fused_ops: ") + op + " is not implemented for CPU tensors");
}

#ifndef FIREANTS_WITH_CUDA
[[noreturn]] void no_cuda(const char* op) {
    throw std::runtime_error(std::string("fireants_fused_ops was built without CUDA support, cannot run ") + op + " on non-CPU tensors");
}
#endif

} // namespace

#ifdef FIREANTS_WITH_CUDA
#define FFO_CUDA_CALL(op, call) return ::call
#else
#define FFO_CUDA_CALL(op, call) no_cuda(op)
#endif

namespace ffo_dispatch {

bool built_with_cuda() {
#ifdef FIREANTS_WITH_CUDA
    return true;
#else
    return false;
#endif
}

torch::Tensor create_intermediates(torch::Tensor input_img, torch::Tensor target_img, torch::Tensor intermediates) {
    if (input_img.is_cpu()) return create_intermediates_cpu(input_img, target_img, intermediates);
    FFO_CUDA_CALL("create_intermediates", create_intermediates(input_img, target_img, intermediates));
}

torch::Tensor cc3d_fwd_interm_v1(torch::Tensor intermediates, int64_t kernel_volume, Reduction reduction, float nr, float dr) {
    if (intermediates.is_cpu()) return cc3d_fwd_interm_v1_cpu(intermediates, kernel_volume, reduction, nr, dr);
    FFO_CUDA_CALL("cc3d_fwd_interm_v1", cc3d_fwd_interm_v1(intermediates, kernel_volume, reduction, nr, dr));
}

torch::Tensor cc3d_bwd_modify_interm_v1(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img,
    torch::Tensor grad_output,
    std::optional<torch::Tensor> grad_input_img, std::optional<torch::Tensor> grad_target_img,
    int64_t kernel_size, int64_t kernel_volume, float nr, float dr, Reduction reduction) {
    if (intermediates.is_cpu()) return cc3d_bwd_modify_interm_v1_cpu(intermediates, input_img, target_img, grad_output, grad_input_img, grad_target_img, kernel_size, kernel_volume, nr, dr, reduction);
    FFO_CUDA_CALL("cc3d_bwd_modify_interm_v1", cc3d_bwd_modify_interm_v1(intermediates, input_img, target_img, grad_output, grad_input_img, grad_target_img, kernel_size, kernel_volume, nr, dr, reduction));
}

void cc3d_bwd_compute_grads(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img,
    torch::Tensor grad_input_img, std::optional<torch::Tensor> grad_target_img) {
    if (intermediates.is_cpu()) return cc3d_bwd_compute_grads_cpu(intermediates, input_img, target_img, grad_input_img, grad_target_img);
    FFO_CUDA_CALL("cc3d_bwd_compute_grads", cc3d_bwd_compute_grads(intermediates, input_img, target_img, grad_input_img, grad_target_img));
}

void box_filter_3d(torch::Tensor &input, int64_t kernel_size) {
    TORCH_CHECK(input.is_cpu(), "box_filter_3d is only implemented for CPU tensors");
    box_filter_3d_cpu(input, kernel_size);
}

torch::Tensor fused_grid_sampler_3d_forward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_3d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine, std::optional<torch::Tensor> output,
    const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    if (input.is_cpu()) return fused_grid_sampler_3d_forward_cpu(input, affine_3d, grid, grid_affine, output, out_D, out_H, out_W,
        grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, is_displacement, interpolation_mode, padding_mode, align_corners);
    FFO_CUDA_CALL("fused_grid_sampler_3d_forward", fused_grid_sampler_3d_forward_impl(input, affine_3d, grid, grid_affine, output, out_D, out_H, out_W,
        grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, is_displacement, interpolation_mode, padding_mode, align_corners));
}

void fused_grid_sampler_3d_backward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_3d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine, const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_input, const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    if (input.is_cpu()) return fused_grid_sampler_3d_backward_cpu(input, affine_3d, grid, grid_affine, grad_output, grad_input, grad_affine, grad_grid,
        out_D, out_H, out_W, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, is_displacement, interpolation_mode, padding_mode, align_corners);
    FFO_CUDA_CALL("fused_grid_sampler_3d_backward", fused_grid_sampler_3d_backward_impl(input, affine_3d, grid, grid_affine, grad_output, grad_input, grad_affine, grad_grid,
        out_D, out_H, out_W, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, is_displacement, interpolation_mode, padding_mode, align_corners));
}

torch::Tensor fused_grid_sampler_2d_forward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_2d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid, std::optional<torch::Tensor> output,
    const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    if (input.is_cpu()) not_on_cpu("fused_grid_sampler_2d_forward");
    FFO_CUDA_CALL("fused_grid_sampler_2d_forward", fused_grid_sampler_2d_forward_impl(input, affine_2d, grid, affine_2d_pregrid, output, out_H, out_W,
        grid_xmin, grid_ymin, grid_xmax, grid_ymax, is_displacement, interpolation_mode, padding_mode, align_corners));
}

void fused_grid_sampler_2d_backward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_2d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid, const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_input, const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    if (input.is_cpu()) not_on_cpu("fused_grid_sampler_2d_backward");
    FFO_CUDA_CALL("fused_grid_sampler_2d_backward", fused_grid_sampler_2d_backward_impl(input, affine_2d, grid, affine_2d_pregrid, grad_output, grad_input, grad_affine, grad_grid,
        out_H, out_W, grid_xmin, grid_ymin, grid_xmax, grid_ymax, is_displacement, interpolation_mode, padding_mode, align_corners));
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> fused_grid_sampler_3d_generic_label_forward_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor> affine_3d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine, std::optional<torch::Tensor> output_labels,
    std::optional<torch::Tensor> output_weights, const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t padding_mode, bool align_corners, bool return_weight,
    const std::optional<float> background_label) {
    if (input.is_cpu()) not_on_cpu("fused_grid_sampler_3d_generic_label_forward");
    FFO_CUDA_CALL("fused_grid_sampler_3d_generic_label_forward", fused_grid_sampler_3d_generic_label_forward_impl(input, affine_3d, grid, grid_affine, output_labels, output_weights,
        out_D, out_H, out_W, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, is_displacement, padding_mode, align_corners, return_weight, background_label));
}

void fused_grid_sampler_3d_generic_label_backward_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor> affine_3d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine, const torch::Tensor& output_labels,
    const std::optional<torch::Tensor>& grad_weight, const std::optional<torch::Tensor>& grad_affine,
    const std::optional<torch::Tensor>& grad_grid, const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t padding_mode, bool align_corners, bool return_weight) {
    if (input.is_cpu()) not_on_cpu("fused_grid_sampler_3d_generic_label_backward");
    FFO_CUDA_CALL("fused_grid_sampler_3d_generic_label_backward", fused_grid_sampler_3d_generic_label_backward_impl(input, affine_3d, grid, grid_affine, output_labels,
        grad_weight, grad_affine, grad_grid, out_D, out_H, out_W, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
        is_displacement, padding_mode, align_corners, return_weight));
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> fused_grid_sampler_2d_generic_label_forward_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor> affine_2d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid, std::optional<torch::Tensor> output_labels,
    std::optional<torch::Tensor> output_weights, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    const bool is_displacement, int64_t padding_mode, bool align_corners, bool return_weight,
    const std::optional<float> background_label) {
    if (input.is_cpu()) not_on_cpu("fused_grid_sampler_2d_generic_label_forward");
    FFO_CUDA_CALL("fused_grid_sampler_2d_generic_label_forward", fused_grid_sampler_2d_generic_label_forward_impl(input, affine_2d, grid, affine_2d_pregrid, output_labels,
        output_weights, out_H, out_W, grid_xmin, grid_ymin, grid_xmax, grid_ymax, is_displacement, padding_mode, align_corners, return_weight, background_label));
}

void fused_grid_sampler_2d_generic_label_backward_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor> affine_2d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid, const torch::Tensor& output_labels,
    const std::optional<torch::Tensor>& grad_weight, const std::optional<torch::Tensor>& grad_affine,
    const std::optional<torch::Tensor>& grad_grid, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    const bool is_displacement, int64_t padding_mode, bool align_corners, bool return_weight) {
    if (input.is_cpu()) not_on_cpu("fused_grid_sampler_2d_generic_label_backward");
    FFO_CUDA_CALL("fused_grid_sampler_2d_generic_label_backward", fused_grid_sampler_2d_generic_label_backward_impl(input, affine_2d, grid, affine_2d_pregrid, output_labels,
        grad_weight, grad_affine, grad_grid, out_H, out_W, grid_xmin, grid_ymin, grid_xmax, grid_ymax, is_displacement, padding_mode, align_corners, return_weight));
}

torch::Tensor fused_grid_composer_3d_forward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_3d, const torch::Tensor grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    bool align_corners, std::optional<torch::Tensor> output) {
    if (input.is_cpu()) return fused_grid_composer_3d_forward_cpu(input, affine_3d, grid, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, align_corners, output);
    FFO_CUDA_CALL("fused_grid_composer_3d_forward", fused_grid_composer_3d_forward_impl(input, affine_3d, grid, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, align_corners, output));
}

void fused_grid_composer_3d_backward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_3d, const torch::Tensor grid,
    const torch::Tensor &grad_output, const std::optional<torch::Tensor> &grad_input,
    const std::optional<torch::Tensor> &grad_affine, const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax, bool align_corners) {
    if (input.is_cpu()) return fused_grid_composer_3d_backward_cpu(input, affine_3d, grid, grad_output, grad_input, grad_affine, grad_grid,
        grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, align_corners);
    FFO_CUDA_CALL("fused_grid_composer_3d_backward", fused_grid_composer_3d_backward_impl(input, affine_3d, grid, grad_output, grad_input, grad_affine, grad_grid,
        grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax, align_corners));
}

torch::Tensor fused_grid_composer_2d_forward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_2d, const torch::Tensor grid,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    bool align_corners, std::optional<torch::Tensor> output) {
    if (input.is_cpu()) not_on_cpu("fused_grid_composer_2d_forward");
    FFO_CUDA_CALL("fused_grid_composer_2d_forward", fused_grid_composer_2d_forward_impl(input, affine_2d, grid, grid_xmin, grid_ymin, grid_xmax, grid_ymax, align_corners, output));
}

void fused_grid_composer_2d_backward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_2d, const torch::Tensor grid,
    const torch::Tensor &grad_output, const std::optional<torch::Tensor> &grad_input,
    const std::optional<torch::Tensor> &grad_affine, const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax, bool align_corners) {
    if (input.is_cpu()) not_on_cpu("fused_grid_composer_2d_backward");
    FFO_CUDA_CALL("fused_grid_composer_2d_backward", fused_grid_composer_2d_backward_impl(input, affine_2d, grid, grad_output, grad_input, grad_affine, grad_grid,
        grid_xmin, grid_ymin, grid_xmax, grid_ymax, align_corners));
}

torch::Tensor fused_warp_create_3d_forward_impl(
    const std::optional<torch::Tensor> affine, const torch::Tensor grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax) {
    if (grid.is_cpu()) return fused_warp_create_3d_forward_cpu(affine, grid, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax);
    FFO_CUDA_CALL("fused_warp_create_3d_forward", fused_warp_create_3d_forward_impl(affine, grid, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax));
}

void fused_warp_create_3d_backward_impl(
    const std::optional<torch::Tensor> affine, const torch::Tensor grid, const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_affine, const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax) {
    if (grid.is_cpu()) return fused_warp_create_3d_backward_cpu(affine, grid, grad_output, grad_affine, grad_grid, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax);
    FFO_CUDA_CALL("fused_warp_create_3d_backward", fused_warp_create_3d_backward_impl(affine, grid, grad_output, grad_affine, grad_grid, grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax));
}

void adam_update_fused(torch::Tensor &grad, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, float beta1, float beta2, float eps) {
    if (grad.is_cpu()) return adam_update_fused_cpu(grad, exp_avg, exp_avg_sq, beta1, beta2, eps);
    FFO_CUDA_CALL("adam_update_fused", adam_update_fused(grad, exp_avg, exp_avg_sq, beta1, beta2, eps));
}

void gaussian_blur_fft2(torch::Tensor &im_fft, int64_t ys, int64_t xs, int64_t ye, int64_t xe, float multiplier) {
    if (im_fft.is_cpu()) return gaussian_blur_fft2_cpu(im_fft, ys, xs, ye, xe, multiplier);
    FFO_CUDA_CALL("gaussian_blur_fft2", gaussian_blur_fft2(im_fft, ys, xs, ye, xe, multiplier));
}

void gaussian_blur_fft3(torch::Tensor &im_fft, int64_t zs, int64_t ys, int64_t xs, int64_t ze, int64_t ye, int64_t xe, float multiplier) {
    if (im_fft.is_cpu()) return gaussian_blur_fft3_cpu(im_fft, zs, ys, xs, ze, ye, xe, multiplier);
    FFO_CUDA_CALL("gaussian_blur_fft3", gaussian_blur_fft3(im_fft, zs, ys, xs, ze, ye, xe, multiplier));
}

std::vector<torch::Tensor> mutual_information_histogram_fwd(torch::Tensor &input_img, torch::Tensor &target_img, int num_bins,
    KernelType kernel_type, float minval, float maxval, float sigma_ratio, bool approximate_reduction) {
    if (input_img.is_cpu()) return mutual_information_histogram_fwd_cpu(input_img, target_img, num_bins, kernel_type, minval, maxval, sigma_ratio, approximate_reduction);
    FFO_CUDA_CALL("mutual_information_histogram_fwd", mutual_information_histogram_fwd(input_img, target_img, num_bins, kernel_type, minval, maxval, sigma_ratio, approximate_reduction));
}

void mutual_information_histogram_bwd(torch::Tensor &input_img, torch::Tensor &target_img, torch::Tensor &grad_pab,
    torch::Tensor &grad_pa, torch::Tensor &grad_pb, int num_bins, torch::Tensor &grad_input_img,
    std::optional<torch::Tensor> &grad_target_img, KernelType kernel_type, float minval, float maxval, float sigma_ratio) {
    if (input_img.is_cpu()) return mutual_information_histogram_bwd_cpu(input_img, target_img, grad_pab, grad_pa, grad_pb, num_bins, grad_input_img, grad_target_img, kernel_type, minval, maxval, sigma_ratio);
    FFO_CUDA_CALL("mutual_information_histogram_bwd", mutual_information_histogram_bwd(input_img, target_img, grad_pab, grad_pa, grad_pb, num_bins, grad_input_img, grad_target_img, kernel_type, minval, maxval, sigma_ratio));
}

} // namespace ffo_dispatch
