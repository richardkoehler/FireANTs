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


// Device dispatch for the bound operations: CPU tensors go to the CPU kernels (CPUOps.h),
// CUDA tensors to the CUDA kernels (only when the extension is built with FIREANTS_WITH_CUDA).
#pragma once
#include <torch/torch.h>
#include <optional>
#include <tuple>
#include <vector>
#include "CrossCorrelation.h"
#include "MutualInformation.h"

namespace ffo_dispatch {

bool built_with_cuda();

torch::Tensor create_intermediates(torch::Tensor input_img, torch::Tensor target_img, torch::Tensor intermediates);
torch::Tensor cc3d_fwd_interm_v1(torch::Tensor intermediates, int64_t kernel_volume, Reduction reduction, float nr, float dr);
torch::Tensor cc3d_bwd_modify_interm_v1(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img,
    torch::Tensor grad_output,
    std::optional<torch::Tensor> grad_input_img, std::optional<torch::Tensor> grad_target_img,
    int64_t kernel_size, int64_t kernel_volume, float nr, float dr, Reduction reduction);
void cc3d_bwd_compute_grads(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img,
    torch::Tensor grad_input_img, std::optional<torch::Tensor> grad_target_img);
void box_filter_3d(torch::Tensor &input, int64_t kernel_size);

torch::Tensor fused_grid_sampler_3d_forward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_3d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine, std::optional<torch::Tensor> output,
    const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
void fused_grid_sampler_3d_backward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_3d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine, const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_input, const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
torch::Tensor fused_grid_sampler_2d_forward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_2d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid, std::optional<torch::Tensor> output,
    const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
void fused_grid_sampler_2d_backward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_2d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid, const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_input, const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    const bool is_displacement, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

std::tuple<torch::Tensor, std::optional<torch::Tensor>> fused_grid_sampler_3d_generic_label_forward_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor> affine_3d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine, std::optional<torch::Tensor> output_labels,
    std::optional<torch::Tensor> output_weights, const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t padding_mode, bool align_corners, bool return_weight,
    const std::optional<float> background_label);
void fused_grid_sampler_3d_generic_label_backward_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor> affine_3d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine, const torch::Tensor& output_labels,
    const std::optional<torch::Tensor>& grad_weight, const std::optional<torch::Tensor>& grad_affine,
    const std::optional<torch::Tensor>& grad_grid, const int64_t out_D, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    const bool is_displacement, int64_t padding_mode, bool align_corners, bool return_weight);
std::tuple<torch::Tensor, std::optional<torch::Tensor>> fused_grid_sampler_2d_generic_label_forward_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor> affine_2d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid, std::optional<torch::Tensor> output_labels,
    std::optional<torch::Tensor> output_weights, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    const bool is_displacement, int64_t padding_mode, bool align_corners, bool return_weight,
    const std::optional<float> background_label);
void fused_grid_sampler_2d_generic_label_backward_impl(
    const torch::Tensor& input, const std::optional<torch::Tensor> affine_2d, const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid, const torch::Tensor& output_labels,
    const std::optional<torch::Tensor>& grad_weight, const std::optional<torch::Tensor>& grad_affine,
    const std::optional<torch::Tensor>& grad_grid, const int64_t out_H, const int64_t out_W,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    const bool is_displacement, int64_t padding_mode, bool align_corners, bool return_weight);

torch::Tensor fused_grid_composer_3d_forward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_3d, const torch::Tensor grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax,
    bool align_corners, std::optional<torch::Tensor> output);
void fused_grid_composer_3d_backward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_3d, const torch::Tensor grid,
    const torch::Tensor &grad_output, const std::optional<torch::Tensor> &grad_input,
    const std::optional<torch::Tensor> &grad_affine, const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax, bool align_corners);
torch::Tensor fused_grid_composer_2d_forward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_2d, const torch::Tensor grid,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax,
    bool align_corners, std::optional<torch::Tensor> output);
void fused_grid_composer_2d_backward_impl(
    const torch::Tensor &input, const std::optional<torch::Tensor> affine_2d, const torch::Tensor grid,
    const torch::Tensor &grad_output, const std::optional<torch::Tensor> &grad_input,
    const std::optional<torch::Tensor> &grad_affine, const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, const float grid_ymin, const float grid_xmax, const float grid_ymax, bool align_corners);

torch::Tensor fused_warp_create_3d_forward_impl(
    const std::optional<torch::Tensor> affine, const torch::Tensor grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax);
void fused_warp_create_3d_backward_impl(
    const std::optional<torch::Tensor> affine, const torch::Tensor grid, const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_affine, const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, const float grid_ymin, const float grid_zmin,
    const float grid_xmax, const float grid_ymax, const float grid_zmax);

void adam_update_fused(torch::Tensor &grad, torch::Tensor exp_avg, torch::Tensor exp_avg_sq, float beta1, float beta2, float eps);
void gaussian_blur_fft2(torch::Tensor &im_fft, int64_t ys, int64_t xs, int64_t ye, int64_t xe, float multiplier);
void gaussian_blur_fft3(torch::Tensor &im_fft, int64_t zs, int64_t ys, int64_t xs, int64_t ze, int64_t ye, int64_t xe, float multiplier);

std::vector<torch::Tensor> mutual_information_histogram_fwd(torch::Tensor &input_img, torch::Tensor &target_img, int num_bins,
    KernelType kernel_type, float minval, float maxval, float sigma_ratio, bool approximate_reduction);
void mutual_information_histogram_bwd(torch::Tensor &input_img, torch::Tensor &target_img, torch::Tensor &grad_pab,
    torch::Tensor &grad_pa, torch::Tensor &grad_pb, int num_bins, torch::Tensor &grad_input_img,
    std::optional<torch::Tensor> &grad_target_img, KernelType kernel_type, float minval, float maxval, float sigma_ratio);

} // namespace ffo_dispatch
