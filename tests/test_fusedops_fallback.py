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
Tests for the choice between the fused kernels and the PyTorch fallbacks, for builds of fireants_fused_ops with and
without CPU / CUDA kernels. The decisions are checked with simulated builds and CUDA-like tensors, so no GPU is needed.
'''
import logging
from types import SimpleNamespace

import pytest
import torch

import fireants.interpolator as interpolator
from fireants.interpolator import ffo_device_supported, ffo_tensors_supported, ffo_supports_tensors


def fake_tensor(device_type, ndim=5):
    return SimpleNamespace(device=SimpleNamespace(type=device_type), ndim=ndim)


@pytest.mark.parametrize("device_type,cpu,cuda,expected", [
    ("cpu", True, False, True),
    ("cpu", False, True, False),
    ("cuda", False, True, True),
    ("cuda", True, False, False),
    ("cuda", True, True, True),
    ("cpu", True, True, True),
    ("mps", True, True, False),
    ("meta", True, True, False),
    ("cpu", False, False, False),
])
def test_ffo_device_supported(device_type, cpu, cuda, expected):
    assert ffo_device_supported(device_type, cpu, cuda) == expected


@pytest.mark.parametrize("device_types,ndims,mode,cpu,cuda,expected", [
    (["cuda", "cuda"], [5, 5], None, False, True, True),
    (["cuda"], [4], None, False, True, True),
    (["cuda"], [4], "genericlabel", False, True, True),
    (["cuda", "cuda"], [5, 5], None, True, False, False),
    (["cpu", "cpu"], [5, 5], None, True, False, True),
    (["cpu", "cpu"], [5, 5], None, False, True, False),
    (["cpu", "cpu"], [4, 4], None, True, False, False),
    (["cpu", "cpu"], [5, 5], "genericlabel", True, False, False),
    (["cpu", "cuda"], [5, 5], None, True, True, False),
    ([], [], None, True, True, False),
])
def test_ffo_tensors_supported(device_types, ndims, mode, cpu, cuda, expected):
    assert ffo_tensors_supported(device_types, ndims, mode, cpu, cuda) == expected


def test_cpu_only_build_sends_cuda_tensors_to_torch(monkeypatch, caplog):
    monkeypatch.setattr(interpolator, "FFO_AVAILABLE", True)
    monkeypatch.setattr(interpolator, "FFO_CPU_AVAILABLE", True)
    monkeypatch.setattr(interpolator, "FFO_CUDA_AVAILABLE", False)
    monkeypatch.setattr(interpolator, "_warned_no_cuda_kernels", False)
    tensors = [fake_tensor("cuda"), fake_tensor("cuda")]
    with caplog.at_level(logging.WARNING, logger=interpolator.logger.name):
        assert not ffo_supports_tensors(tensors)
        assert not ffo_supports_tensors(tensors)
    warnings = [r for r in caplog.records if "without CUDA support" in r.getMessage()]
    assert len(warnings) == 1
    assert ffo_supports_tensors([fake_tensor("cpu"), fake_tensor("cpu")])


def test_cuda_only_build_sends_cpu_tensors_to_torch(monkeypatch):
    monkeypatch.setattr(interpolator, "FFO_AVAILABLE", True)
    monkeypatch.setattr(interpolator, "FFO_CPU_AVAILABLE", False)
    monkeypatch.setattr(interpolator, "FFO_CUDA_AVAILABLE", True)
    assert ffo_supports_tensors([fake_tensor("cuda")])
    assert not ffo_supports_tensors([fake_tensor("cpu"), fake_tensor("cpu")])


def test_no_fused_ops_sends_everything_to_torch(monkeypatch):
    monkeypatch.setattr(interpolator, "FFO_AVAILABLE", False)
    monkeypatch.setattr(interpolator, "FFO_CPU_AVAILABLE", False)
    monkeypatch.setattr(interpolator, "FFO_CUDA_AVAILABLE", False)
    assert not ffo_supports_tensors([fake_tensor("cuda")])
    assert not ffo_supports_tensors([fake_tensor("cpu")])


def test_fused_loss_selection_falls_back_like_import_error(monkeypatch):
    import fireants.registration.abstract as abstract
    monkeypatch.setattr(abstract, "FFO_CPU_AVAILABLE", True)
    monkeypatch.setattr(abstract, "FFO_CUDA_AVAILABLE", False)
    abstract._check_fused_ops_device(torch.device("cpu"))
    with pytest.raises(ImportError):
        abstract._check_fused_ops_device(torch.device("cuda"))
    monkeypatch.setattr(abstract, "FFO_CPU_AVAILABLE", False)
    monkeypatch.setattr(abstract, "FFO_CUDA_AVAILABLE", True)
    abstract._check_fused_ops_device(torch.device("cuda"))
    with pytest.raises(ImportError):
        abstract._check_fused_ops_device(torch.device("cpu"))


def test_warp_adam_kernel_follows_build(monkeypatch):
    import fireants.registration.optimizers.adam as adam
    warp = torch.zeros(1, 4, 4, 4, 3)
    monkeypatch.setattr(adam, "FFO_CPU_AVAILABLE", False)
    assert adam.WarpAdam(warp, 0.1).adam_update_kernel is adam.adam_update_fused_baseline
    monkeypatch.setattr(adam, "FFO_CPU_AVAILABLE", True)
    monkeypatch.setattr(adam, "adam_update_fused", object())
    assert adam.WarpAdam(warp, 0.1).adam_update_kernel is adam.adam_update_fused


def test_smoothing_keeps_contiguous_copy_for_cuda_only():
    from fireants.registration.optimizers.adam import _permute_for_filtering
    warp = torch.randn(1, 4, 5, 6, 3)
    permute = (0, 4, 1, 2, 3)
    cpu = _permute_for_filtering(warp, permute)
    assert not cpu.is_contiguous() and torch.equal(cpu, warp.permute(*permute))
    cuda_like = SimpleNamespace(device=SimpleNamespace(type="cuda"), permute=lambda *p: SimpleNamespace(
        device=SimpleNamespace(type="cuda"), contiguous=lambda: "contiguous copy"))
    assert _permute_for_filtering(cuda_like, permute) == "contiguous copy"


def test_downsample_ignores_fft_kernels_missing_for_device(monkeypatch):
    import fireants.utils.imageutils as imageutils
    monkeypatch.setattr(imageutils, "FFO_CPU_AVAILABLE", False)
    monkeypatch.setattr(imageutils, "FFO_CUDA_AVAILABLE", False)
    monkeypatch.setattr(imageutils, "ffo", object())
    calls = []
    monkeypatch.setattr(imageutils, "downsample_fft", lambda *a, **k: calls.append(1))
    image = torch.rand(1, 1, 8, 8, 8)
    out = imageutils.downsample(image, [4, 4, 4], "trilinear", use_fft=True)
    assert out.shape == (1, 1, 4, 4, 4) and not calls
