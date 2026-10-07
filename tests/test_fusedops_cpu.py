# Copyright (c) 2026 The FireANTs authors and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
Parity tests for the CPU kernels of fireants_fused_ops against the PyTorch implementations
(forward, backward and gradcheck on small double tensors).
'''
import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from fireants.interpolator import FFO_CPU_AVAILABLE, ffo_supports_tensors
from fireants.interpolator.grid_sample import (
    torch_grid_sampler_3d, torch_warp_composer_3d, torch_affine_warp_3d,
)

pytestmark = pytest.mark.skipif(not FFO_CPU_AVAILABLE, reason="fireants_fused_ops with CPU kernels is not installed")

if FFO_CPU_AVAILABLE:
    import fireants_fused_ops as ffo
    from fireants.interpolator.fused_grid_sample import fused_grid_sampler_3d, fused_warp_composer_3d, fused_affine_warp_3d
    from fireants.losses.fusedcc import FusedLocalNormalizedCrossCorrelationLoss
    from fireants.losses.fusedmi import FusedGlobalMutualInformationLoss, MI_histogram_kernel
    from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
    from fireants.registration.optimizers.adam import adam_update_fused_baseline
    from fireants.utils.imageutils import downsample_fft

seed = 1234


def random_affine(B, scale=0.05, dtype=torch.float32, generator=None):
    A = torch.eye(3, 4, dtype=dtype)[None].repeat(B, 1, 1)
    A = A + scale * torch.randn(B, 3, 4, dtype=dtype, generator=generator)
    return A.contiguous()


def smooth_volume(shape, dtype=torch.float32, generator=None):
    ''' random volume that is smooth enough for trilinear interpolation to be well-conditioned '''
    x = torch.randn(shape, dtype=dtype, generator=generator)
    x = F.avg_pool3d(x, 3, stride=1, padding=1)
    return x.contiguous()


def frac_close(a, b, atol, rtol=0.0):
    return (torch.abs(a - b) <= atol + rtol * torch.abs(b)).float().mean().item()


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [True, False])
def test_grid_sampler_3d_forward(mode, padding, align_corners):
    g = torch.Generator().manual_seed(seed)
    inp = smooth_volume((2, 3, 13, 11, 9), generator=g)
    out_shape = (10, 12, 7)
    affine = random_affine(2, generator=g)
    disp = 0.05 * torch.randn(2, *out_shape, 3, generator=g)
    pre = (torch.eye(3) + 0.1 * torch.randn(3, 3, generator=g))[None].repeat(2, 1, 1).contiguous()
    base = F.affine_grid(torch.eye(3, 4)[None].repeat(2, 1, 1), (2, 1, *out_shape), align_corners=align_corners)
    cases = [
        dict(affine=affine, grid=None, is_displacement=False, out_shape=out_shape),
        dict(affine=None, grid=(base + disp).contiguous(), is_displacement=False),
        dict(affine=None, grid=disp, grid_affine=pre, is_displacement=True),
        dict(affine=affine, grid=disp, is_displacement=True),
        dict(affine=affine, grid=disp, grid_affine=pre, is_displacement=True),
    ]
    for kw in cases:
        out = fused_grid_sampler_3d(inp, mode=mode, padding_mode=padding, align_corners=align_corners, **kw)
        ref = torch_grid_sampler_3d(inp, mode=mode, padding_mode=padding, align_corners=align_corners, **kw)
        assert out.shape == ref.shape
        if mode == "bilinear":
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)
        else:
            # rounding ties at exactly .5 can pick a different neighbour
            assert frac_close(out, ref, 1e-6) > 0.995


def test_grid_sampler_3d_output_accumulates():
    g = torch.Generator().manual_seed(seed)
    inp = smooth_volume((1, 2, 8, 8, 8), generator=g)
    disp = 0.05 * torch.randn(1, 8, 8, 8, 3, generator=g)
    out0 = torch.randn(1, 2, 8, 8, 8, generator=g)
    out = fused_grid_sampler_3d(inp, grid=disp, output=out0.clone(), align_corners=True)
    ref = out0 + torch_grid_sampler_3d(inp, grid=disp, align_corners=True)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("padding", ["zeros", "border"])
def test_grid_sampler_3d_backward_vs_torch(align_corners, padding):
    g = torch.Generator().manual_seed(seed)
    inp = smooth_volume((2, 2, 12, 10, 11), generator=g)
    out_shape = (9, 8, 10)
    affine = random_affine(2, generator=g)
    disp = 0.05 * torch.randn(2, *out_shape, 3, generator=g)
    pre = (torch.eye(3) + 0.1 * torch.randn(3, 3, generator=g))[None].repeat(2, 1, 1).contiguous()
    gout = torch.randn(2, 2, *out_shape, generator=g)
    for use_pre in [False, True]:
        grads = []
        for fn in [fused_grid_sampler_3d, torch_grid_sampler_3d]:
            x = inp.clone().requires_grad_(True)
            A = affine.clone().requires_grad_(True)
            u = disp.clone().requires_grad_(True)
            out = fn(x, affine=A, grid=u, grid_affine=pre if use_pre else None, padding_mode=padding,
                     align_corners=align_corners, is_displacement=True)
            out.backward(gout)
            grads.append((x.grad, A.grad, u.grad))
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)


def test_grid_sampler_3d_affine_only_backward_vs_torch():
    g = torch.Generator().manual_seed(seed)
    inp = smooth_volume((1, 1, 14, 12, 10), generator=g)
    affine = random_affine(1, generator=g)
    gout = torch.randn(1, 1, 9, 10, 11, generator=g)
    grads = []
    for fn in [fused_grid_sampler_3d, torch_grid_sampler_3d]:
        x = inp.clone().requires_grad_(True)
        A = affine.clone().requires_grad_(True)
        fn(x, affine=A, out_shape=(9, 10, 11), align_corners=True, is_displacement=False).backward(gout)
        grads.append((x.grad, A.grad))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)


def test_grid_sampler_3d_broadcast():
    g = torch.Generator().manual_seed(seed)
    inp = smooth_volume((1, 2, 9, 9, 9), generator=g)
    affine = random_affine(3, generator=g)
    disp = 0.05 * torch.randn(1, 7, 8, 9, 3, generator=g)
    gout = torch.randn(3, 2, 7, 8, 9, generator=g)
    grads = []
    for fn, expand in [(fused_grid_sampler_3d, False), (torch_grid_sampler_3d, True)]:
        x = inp.clone().requires_grad_(True)
        A = affine.clone().requires_grad_(True)
        u = disp.clone().requires_grad_(True)
        out = fn(x.expand(3, -1, -1, -1, -1) if expand else x, affine=A, grid=u.expand(3, -1, -1, -1, -1) if expand else u,
                 align_corners=True, is_displacement=True)
        out.backward(gout)
        grads.append((out.detach(), x.grad, A.grad, u.grad))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)


def test_grid_sampler_3d_gradcheck():
    g = torch.Generator().manual_seed(seed)
    inp = smooth_volume((1, 2, 6, 5, 7), dtype=torch.float64, generator=g).requires_grad_(True)
    affine = random_affine(1, dtype=torch.float64, generator=g).requires_grad_(True)
    disp = (0.05 * torch.randn(1, 4, 5, 6, 3, dtype=torch.float64, generator=g)).requires_grad_(True)
    pre = (torch.eye(3, dtype=torch.float64) + 0.1 * torch.randn(3, 3, dtype=torch.float64, generator=g))[None].contiguous()
    for grid_affine in [None, pre]:
        fn = lambda x, A, u: fused_grid_sampler_3d(x, affine=A, grid=u, grid_affine=grid_affine, align_corners=True, is_displacement=True)
        assert gradcheck(fn, (inp, affine, disp), eps=1e-6, atol=1e-5)
    fn = lambda x, A: fused_grid_sampler_3d(x, affine=A, out_shape=(4, 5, 3), align_corners=False, is_displacement=False)
    assert gradcheck(fn, (inp, affine), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("align_corners", [True, False])
def test_warp_composer_3d(align_corners):
    g = torch.Generator().manual_seed(seed)
    u = 0.05 * torch.randn(2, 10, 9, 8, 3, generator=g)
    v = 0.05 * torch.randn(2, 7, 9, 11, 3, generator=g)
    affine = random_affine(2, generator=g)
    gout = torch.randn(2, 7, 9, 11, 3, generator=g)
    grads = []
    for fn in [fused_warp_composer_3d, torch_warp_composer_3d]:
        uu = u.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True)
        A = affine.clone().requires_grad_(True)
        out = fn(uu, affine=A, v=vv, align_corners=align_corners)
        out.backward(gout)
        grads.append((out.detach(), uu.grad, vv.grad, A.grad))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)


def test_warp_composer_3d_gradcheck():
    g = torch.Generator().manual_seed(seed)
    u = (0.05 * torch.randn(1, 5, 6, 4, 3, dtype=torch.float64, generator=g)).requires_grad_(True)
    v = (0.05 * torch.randn(1, 4, 5, 6, 3, dtype=torch.float64, generator=g)).requires_grad_(True)
    A = random_affine(1, dtype=torch.float64, generator=g).requires_grad_(True)
    fn = lambda u, v, A: fused_warp_composer_3d(u, affine=A, v=v, align_corners=True)
    assert gradcheck(fn, (u, v, A), eps=1e-6, atol=1e-5)


def test_affine_warp_3d():
    g = torch.Generator().manual_seed(seed)
    grid = 0.05 * torch.randn(2, 6, 7, 8, 3, generator=g)
    affine = random_affine(2, generator=g)
    gout = torch.randn(2, 6, 7, 8, 3, generator=g)
    grads = []
    for fn in [fused_affine_warp_3d, torch_affine_warp_3d]:
        u = grid.clone().requires_grad_(True)
        A = affine.clone().requires_grad_(True)
        out = fn(A, u, align_corners=True)
        out.backward(gout)
        grads.append((out.detach(), u.grad, A.grad))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    u = grid[:1].double().requires_grad_(True)
    A = affine[:1].double().requires_grad_(True)
    assert gradcheck(lambda A, u: fused_affine_warp_3d(A, u, align_corners=True), (A, u), eps=1e-6, atol=1e-6)


def test_dispatcher_uses_cpu_kernels():
    x = torch.rand(1, 1, 4, 4, 4)
    u = torch.zeros(1, 4, 4, 4, 3)
    assert ffo_supports_tensors([x, u])
    assert not ffo_supports_tensors([x[0], u[0]])  # 2D-shaped tensors stay on the torch path
    assert not ffo_supports_tensors([x, u], mode="genericlabel")


@pytest.mark.parametrize("shape,k", [((1, 2, 9, 10, 11), 3), ((2, 1, 5, 6, 7), 5), ((1, 1, 4, 5, 3), 9), ((1, 1, 7, 1, 6), 3)])
def test_box_filter_3d(shape, k):
    g = torch.Generator().manual_seed(seed)
    for dtype in [torch.float32, torch.float64]:
        x = torch.randn(shape, dtype=dtype, generator=g)
        ref = x
        for dim in range(3):
            kernel_shape = [1, 1, 1, 1, 1]
            kernel_shape[2 + dim] = k
            pad = [0, 0, 0]
            pad[dim] = k // 2
            w = torch.ones(kernel_shape, dtype=dtype).repeat(shape[1], 1, 1, 1, 1) / k
            ref = F.conv3d(ref, w, padding=tuple(pad), groups=shape[1])
        y = x.clone()
        ffo.box_filter_3d(y, k)
        torch.testing.assert_close(y, ref, atol=1e-5 if dtype == torch.float32 else 1e-12, rtol=1e-5)


def reference_fused_cc(I, J, k, nr, dr):
    ''' forward of the fused LNCC written with avg_pool3d (zero padded box mean) '''
    mean = lambda x: F.avg_pool3d(x, k, stride=1, padding=k // 2, count_include_pad=True)
    kv = k ** 3
    mI, mJ, mI2, mJ2, mIJ = mean(I), mean(J), mean(I * I), mean(J * J), mean(I * J)
    a = kv * (mIJ - mI * mJ)
    b = torch.clamp(kv * (mI2 - mI * mI), min=dr)
    c = torch.clamp(kv * (mJ2 - mJ * mJ), min=dr)
    return ((a * a + nr) / (b * c + dr)).clamp(-1, 1)


@pytest.mark.parametrize("use_ants_gradient", [True, False])
def test_fused_cc_cpu(use_ants_gradient):
    g = torch.Generator().manual_seed(seed)
    N, k = 20, 5
    img1 = torch.rand(1, 1, N, N, N, generator=g) + 0.1
    img2 = (torch.rand(1, 1, N, N, N, generator=g) * 0.5 + 0.5 * img1)
    loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=k, reduction='mean', use_ants_gradient=use_ants_gradient, smooth_nr=0)
    x = img2.clone().requires_grad_(True)
    out = loss(img1, x)
    ref = -reference_fused_cc(img1, img2, k, 0.0, loss.smooth_dr).mean()
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    out.backward()
    # like the CUDA kernels, the fused gradient is not divided by the number of voxels for 'mean'
    grad_ours = x.grad / N ** 3
    if use_ants_gradient:
        # gradient that ignores the dependence of neighbouring windows (ANTs-style), written out explicitly
        mean = lambda t: F.avg_pool3d(t, k, stride=1, padding=k // 2, count_include_pad=True)
        kv = k ** 3
        I, J = img2, img1
        mI, mJ, mI2, mJ2, mIJ = mean(I), mean(J), mean(I * I), mean(J * J), mean(I * J)
        a = kv * (mIJ - mI * mJ)
        b = kv * (mI2 - mI * mI)
        c = kv * (mJ2 - mJ * mJ)
        gO = -2 * a / (b * c + loss.smooth_dr)
        ref_grad = (gO * J - gO * a / (b + loss.smooth_dr) * I + gO * (a / (b + loss.smooth_dr) * mI - mJ)) / N ** 3
    else:
        # the windowed sums are box means, so the exact gradient is scaled by 1 / kernel volume
        y = img2.clone().requires_grad_(True)
        LocalNormalizedCrossCorrelationLoss(3, kernel_size=k, reduction='mean', smooth_nr=0)(img1, y).backward()
        ref_grad = y.grad / k ** 3
    torch.testing.assert_close(grad_ours, ref_grad, atol=1e-5, rtol=1e-3)


def test_fused_cc_cpu_multichannel_batch():
    g = torch.Generator().manual_seed(seed)
    img1 = torch.rand(2, 3, 12, 11, 10, generator=g) + 0.1
    img2 = torch.rand(2, 3, 12, 11, 10, generator=g) * 0.5 + 0.5 * img1
    for reduction in ['mean', 'sum', 'none']:
        loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=3, reduction=reduction, use_ants_gradient=False)
        x = img2.clone().requires_grad_(True)
        t = img1.clone().requires_grad_(True)
        out = loss(x, t)
        ref = -reference_fused_cc(img2, img1, 3, loss.smooth_nr, loss.smooth_dr)
        ref = {'mean': ref.mean(), 'sum': ref.sum(), 'none': ref}[reduction]
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        (out.sum() if reduction == 'none' else out).backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(t.grad).all()


def reference_parzen(x, num_bins, sigma_ratio):
    centers = (torch.arange(num_bins, dtype=x.dtype) + 0.5) / num_bins
    sigma = 0.5 / num_bins * sigma_ratio
    w = torch.exp(-0.5 * ((x.reshape(*x.shape[:2], -1, 1) - centers) / sigma) ** 2)
    return w / w.sum(-1, keepdim=True)


def test_mi_histogram_forward():
    g = torch.Generator().manual_seed(seed)
    nb = 16
    I = torch.rand(2, 2, 6, 7, 8, generator=g)
    J = (0.5 * I + 0.5 * torch.rand(2, 2, 6, 7, 8, generator=g))
    # approximate (hard binning) histogram
    pab, pa, pb = ffo.mutual_information_histogram_fwd(I, J, nb, ffo.KernelType.GAUSSIAN, 0.0, 1.0, 1.0, True)
    ib = (I * nb).floor().clamp(0, nb - 1).long().flatten(2)
    jb = (J * nb).floor().clamp(0, nb - 1).long().flatten(2)
    n = ib.shape[-1]
    ref_pab = F.one_hot(ib, nb).double().transpose(-1, -2) @ F.one_hot(jb, nb).double() / n
    torch.testing.assert_close(pab.double(), ref_pab, atol=1e-6, rtol=0)
    torch.testing.assert_close(pa.double(), ref_pab.sum(-1), atol=1e-6, rtol=0)
    torch.testing.assert_close(pb.double(), ref_pab.sum(-2), atol=1e-6, rtol=0)
    # exact Parzen histogram
    pab, pa, pb = ffo.mutual_information_histogram_fwd(I, J, nb, ffo.KernelType.GAUSSIAN, 0.0, 1.0, 1.0, False)
    wa, wb = reference_parzen(I.double(), nb, 1.0), reference_parzen(J.double(), nb, 1.0)
    torch.testing.assert_close(pab.double(), wa.transpose(-1, -2) @ wb / n, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(pa.double(), wa.mean(-2), atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(pb.double(), wb.mean(-2), atol=1e-6, rtol=1e-4)


def test_mi_histogram_backward_matches_parzen_autograd():
    g = torch.Generator().manual_seed(seed)
    nb = 16
    I = torch.rand(1, 2, 5, 6, 7, generator=g, dtype=torch.float64)
    J = (0.5 * I + 0.5 * torch.rand(1, 2, 5, 6, 7, generator=g, dtype=torch.float64))
    Gab = torch.randn(1, 2, nb, nb, generator=g, dtype=torch.float64)
    Ga = torch.randn(1, 2, nb, generator=g, dtype=torch.float64)
    Gb = torch.randn(1, 2, nb, generator=g, dtype=torch.float64)
    x = I.clone().requires_grad_(True)
    y = J.clone().requires_grad_(True)
    wa, wb = reference_parzen(x, nb, 1.0), reference_parzen(y, nb, 1.0)
    n = wa.shape[-2]
    obj = (Gab * (wa.transpose(-1, -2) @ wb / n)).sum() + (Ga * wa.mean(-2)).sum() + (Gb * wb.mean(-2)).sum()
    obj.backward()
    gI, gJ = torch.zeros_like(I), torch.zeros_like(J)
    ffo.mutual_information_histogram_bwd(I, J, Gab, Ga, Gb, nb, gI, gJ, ffo.KernelType.GAUSSIAN, 0.0, 1.0, 1.0)
    torch.testing.assert_close(gI, x.grad, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(gJ, y.grad, atol=1e-6, rtol=1e-4)


def test_fused_mi_loss_cpu_runs():
    g = torch.Generator().manual_seed(seed)
    I = torch.rand(1, 1, 16, 16, 16, generator=g)
    J = (0.7 * I + 0.3 * torch.rand(1, 1, 16, 16, 16, generator=g)).requires_grad_(True)
    loss = FusedGlobalMutualInformationLoss(num_bins=32)
    out = loss(J, I)
    out.backward()
    assert torch.isfinite(out) and out.item() < 0
    assert torch.isfinite(J.grad).all() and J.grad.abs().sum() > 0


def test_adam_update_fused():
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(1, 6, 7, 8, 3, generator=g)
    v = torch.rand(1, 6, 7, 8, 3, generator=g)
    grad = torch.empty_like(m)
    ref = torch.empty_like(m)
    ffo.adam_update_fused(grad, m, v, 0.3, 0.2, 1e-8)
    adam_update_fused_baseline(ref, m, v, 0.3, 0.2, 1e-8)
    torch.testing.assert_close(grad, ref, atol=1e-6, rtol=1e-5)


def test_gaussian_blur_fft3():
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 2, 6, 8, 10, dtype=torch.complex64, generator=g)
    zs, ys, xs = -3, -4, -5
    y = x.clone()
    ffo.gaussian_blur_fft3(y, zs, ys, xs, 3, 4, 5, 0.5)
    D, H, W = x.shape[2:]
    fz = (zs + torch.arange(D)) / (D / 4.0)
    fy = (ys + torch.arange(H)) / (H / 4.0)
    fx = (xs + torch.arange(W)) / (W / 4.0)
    filt = torch.exp(-0.5 * (fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)) * 0.5
    torch.testing.assert_close(y, x * filt, atol=1e-6, rtol=1e-5)
    img = torch.rand(1, 1, 32, 30, 28, generator=g)
    small = downsample_fft(img, [16, 15, 14])
    assert small.shape == (1, 1, 16, 15, 14) and torch.isfinite(small).all()


def synthetic_pair(n=40):
    import SimpleITK as sitk
    from fireants.io.image import Image, BatchedImages
    z, y, x = torch.meshgrid(*[torch.linspace(-1, 1, n)] * 3, indexing='ij')
    blob = lambda x, y, z: torch.exp(-((x / 0.5) ** 2 + (y / 0.35) ** 2 + (z / 0.45) ** 2) * 2) + 0.5 * (torch.sqrt(x ** 2 + y ** 2 + z ** 2) < 0.3).float()
    fixed = blob(x, y, z)
    moving = blob(x + 0.1 * torch.sin(3 * z), y - 0.08, z)
    to_batch = lambda t: BatchedImages([Image(sitk.GetImageFromArray(t.numpy()), device='cpu')])
    return to_batch(fixed), to_batch(moving)


@pytest.mark.parametrize("loss_type", ["cc", "fusedcc"])
def test_greedy_registration_cpu_kernels_match_torch_path(loss_type):
    ''' a greedy registration with the CPU kernels follows the same trajectory as with the torch fallback '''
    from fireants.interpolator import fireants_interpolator
    from fireants.registration.greedy import GreedyRegistration
    fixed, moving = synthetic_pair()
    results = []
    prev = fireants_interpolator.use_ffo
    try:
        for use_ffo in [True, False]:
            fireants_interpolator.use_ffo = use_ffo
            reg = GreedyRegistration([2, 1], [30, 15], fixed, moving, loss_type=loss_type, cc_kernel_size=5,
                                     optimizer='adam', optimizer_lr=0.5, progress_bar=False)
            reg.optimize()
            with torch.no_grad():
                results.append((reg.evaluate(fixed, moving), reg.get_warped_coordinates(fixed, moving)))
    finally:
        fireants_interpolator.use_ffo = prev
    (moved_ffo, coords_ffo), (moved_torch, coords_torch) = results
    initial = F.mse_loss(moving(), fixed()).item()
    assert F.mse_loss(moved_ffo, fixed()).item() < 0.5 * initial
    # displacement agreement in voxels (rounding differences are amplified a little by the normalized Adam updates)
    diff_vox = (coords_ffo - coords_torch).abs() * (40 - 1) / 2
    assert diff_vox.mean().item() < 0.02 and diff_vox.max().item() < 0.5
    assert (moved_ffo - moved_torch).abs().mean().item() < 1e-3


def test_affine_and_inverse_export_cpu():
    ''' affine (fused MI) + greedy (fused CC) on CPU, and the inverse warp undoes the forward warp '''
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.greedy import GreedyRegistration
    fixed, moving = synthetic_pair()
    aff = AffineRegistration([4, 2, 1], [60, 30, 15], fixed, moving, loss_type='fusedmi', optimizer_lr=3e-3, progress_bar=False)
    aff.optimize()
    reg = GreedyRegistration([2, 1], [30, 15], fixed, moving, loss_type='fusedcc', cc_kernel_size=5, optimizer='adam',
                             optimizer_lr=0.5, init_affine=aff.get_affine_matrix().detach(), progress_bar=False)
    reg.optimize()
    with torch.no_grad():
        fwd = reg.get_warped_coordinates(fixed, moving)
        inv = reg.get_inverse_warped_coordinates(fixed, moving)
    # inv(fwd(x)) = x for fixed grid points that land inside the moving image
    roundtrip = F.grid_sample(inv.permute(0, 4, 1, 2, 3), fwd, mode='bilinear', align_corners=True).permute(0, 2, 3, 4, 1)
    grid = F.affine_grid(torch.eye(3, 4)[None], [1, 1, 40, 40, 40], align_corners=True)
    inside = (fwd.abs() < 0.9).all(-1)
    err_vox = ((roundtrip - grid).abs() * (40 - 1) / 2)[inside]
    assert err_vox.mean().item() < 0.1



@pytest.fixture(params=[2, 3])
def num_threads(request):
    prev = torch.get_num_threads()
    torch.set_num_threads(request.param)
    yield request.param
    torch.set_num_threads(prev)


def repeat_equal(fn, repeats=4):
    ''' results of fn (a tuple of tensors) must be bitwise identical in every repetition '''
    first = fn()
    for _ in range(repeats - 1):
        for a, b in zip(first, fn()):
            assert torch.equal(a, b)


def test_grid_sampler_3d_backward_is_reproducible(num_threads):
    g = torch.Generator().manual_seed(seed)
    inp = torch.rand(2, 2, 24, 26, 28, generator=g)
    out_shape = (26, 30, 34)
    affine = random_affine(2, generator=g)
    disp = 0.1 * torch.randn(2, *out_shape, 3, generator=g)
    gout = torch.randn(2, 2, *out_shape, generator=g)
    for mode in ['bilinear', 'nearest']:
        def run():
            x, A, u = inp.clone().requires_grad_(True), affine.clone().requires_grad_(True), disp.clone().requires_grad_(True)
            fused_grid_sampler_3d(x, affine=A, grid=u, mode=mode, padding_mode='border', align_corners=True,
                                  is_displacement=True).backward(gout)
            return x.grad, A.grad, u.grad
        repeat_equal(run)


def test_warp_composer_and_affine_warp_backward_are_reproducible(num_threads):
    g = torch.Generator().manual_seed(seed)
    shape = (1, 24, 28, 30, 3)
    u0, v0 = 0.05 * torch.randn(*shape, generator=g), 0.05 * torch.randn(*shape, generator=g)
    affine = random_affine(1, generator=g)
    gout = torch.randn(*shape, generator=g)

    def run_composer():
        u, v, A = u0.clone().requires_grad_(True), v0.clone().requires_grad_(True), affine.clone().requires_grad_(True)
        fused_warp_composer_3d(u, A, v).backward(gout)
        return u.grad, v.grad, A.grad

    def run_affine():
        v, A = v0.clone().requires_grad_(True), affine.clone().requires_grad_(True)
        fused_affine_warp_3d(A, v).backward(gout)
        return v.grad, A.grad
    repeat_equal(run_composer)
    repeat_equal(run_affine)


@pytest.mark.parametrize("reduction", ['mean', 'sum'])
def test_fused_cc_is_reproducible(num_threads, reduction):
    g = torch.Generator().manual_seed(seed)
    img1 = torch.rand(2, 2, 30, 36, 34, generator=g)
    img2 = 0.5 * img1 + 0.5 * torch.rand(2, 2, 30, 36, 34, generator=g)
    loss = FusedLocalNormalizedCrossCorrelationLoss(3, kernel_size=5, reduction=reduction)

    def run():
        x = img2.clone().requires_grad_(True)
        out = loss(img1, x)
        out.backward()
        return out.detach(), x.grad
    repeat_equal(run)


@pytest.mark.parametrize("approximate", [True, False])
def test_mi_histogram_is_reproducible(num_threads, approximate):
    g = torch.Generator().manual_seed(seed)
    I = torch.rand(2, 2, 30, 36, 34, generator=g)
    J = 0.5 * I + 0.5 * torch.rand(2, 2, 30, 36, 34, generator=g)
    nb = 24
    gab, ga, gb = torch.randn(2, 2, nb, nb, generator=g), torch.randn(2, 2, nb, generator=g), torch.randn(2, 2, nb, generator=g)

    def run():
        pab, pa, pb = ffo.mutual_information_histogram_fwd(I, J, nb, ffo.KernelType.BSPLINE, 0.0, 1.0, 1.0, approximate)
        gI, gJ = torch.zeros_like(I), torch.zeros_like(J)
        ffo.mutual_information_histogram_bwd(I, J, gab, ga, gb, nb, gI, gJ, ffo.KernelType.BSPLINE, 0.0, 1.0, 1.0)
        return pab, pa, pb, gI, gJ
    repeat_equal(run)
