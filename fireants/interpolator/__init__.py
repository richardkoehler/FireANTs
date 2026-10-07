# Copyright (c) 2026 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


'''
Dispatcher for fused operations
'''
import os
from typing import Dict, Any, Callable
from fireants.interpolator.grid_sample import torch_grid_sampler_2d, torch_grid_sampler_3d, torch_warp_composer_2d, torch_warp_composer_3d, torch_affine_warp_2d, torch_affine_warp_3d
import logging
import torch
logger = logging.getLogger(__name__)

# Try to import fused operations
try:
    import fireants_fused_ops as ffo
    FFO_AVAILABLE = True
    # safe to import fused grid sampler
    from fireants.interpolator.fused_grid_sample import fused_grid_sampler_3d, fused_warp_composer_3d, fused_affine_warp_3d
    from fireants.interpolator.fused_grid_sample import fused_grid_sampler_2d, fused_warp_composer_2d
    from fireants.interpolator.fused_grid_sample_genericlabel import (
        fused_grid_sampler_2d_generic_label,
        fused_grid_sampler_3d_generic_label,
    )
except ImportError:
    logger.warning("Fused operations not available, compile the fused ops to use them")
    FFO_AVAILABLE = False
    ffo = None
    fused_grid_sampler_2d = None
    fused_grid_sampler_3d = None
    fused_warp_composer_3d = None
    fused_affine_warp_3d = None
    fused_warp_composer_2d = None
    fused_grid_sampler_2d_generic_label = None
    fused_grid_sampler_3d_generic_label = None

# builds that expose `built_with_cuda` also contain CPU kernels for the 3D ops, builds without it are CUDA-only
FFO_CPU_AVAILABLE = FFO_AVAILABLE and hasattr(ffo, 'built_with_cuda')
FFO_CUDA_AVAILABLE = FFO_AVAILABLE and (not hasattr(ffo, 'built_with_cuda') or ffo.built_with_cuda())

# Get environment variable with default True
USE_FFO = os.getenv('USE_FFO', 'True').lower() == 'true'

_warned_no_cuda_kernels = False

def ffo_device_supported(device_type: str, cpu_available: bool, cuda_available: bool) -> bool:
    ''' check if a build with the given kernels has fused ops for tensors on this device type '''
    return (device_type == 'cuda' and cuda_available) or (device_type == 'cpu' and cpu_available)

def ffo_tensors_supported(device_types, ndims, mode, cpu_available: bool, cuda_available: bool) -> bool:
    ''' check if the fused ops can run on tensors with these device types and numbers of dimensions '''
    device_types = set(device_types)
    if len(device_types) != 1:
        return False
    device_type = device_types.pop()
    if not ffo_device_supported(device_type, cpu_available, cuda_available):
        return False
    # on CPU only the 3D ops (5D tensors) have kernels, generic-label sampling is CUDA-only
    return device_type == 'cuda' or (mode != 'genericlabel' and any(n == 5 for n in ndims))

def ffo_supports_tensors(tensors, mode=None) -> bool:
    ''' check if the installed fused ops can run on these tensors, otherwise callers fall back to PyTorch '''
    global _warned_no_cuda_kernels
    device_types = set(x.device.type for x in tensors)
    if device_types == {'cuda'} and FFO_AVAILABLE and not FFO_CUDA_AVAILABLE and not _warned_no_cuda_kernels:
        _warned_no_cuda_kernels = True
        logger.warning("fireants_fused_ops was built without CUDA support, falling back to PyTorch for CUDA tensors")
    return ffo_tensors_supported(device_types, [x.ndim for x in tensors], mode, FFO_CPU_AVAILABLE, FFO_CUDA_AVAILABLE)

class GridSampleDispatcher:
    """Dispatcher for grid sample operations that handles both fused and non-fused implementations."""
    
    def __init__(self):
        self._use_ffo = (USE_FFO and FFO_AVAILABLE)
        print(f"USE_FFO: {USE_FFO}")
        print(f"FFO_AVAILABLE: {FFO_AVAILABLE}")
        logger.info(f"Using FFO: {self._use_ffo}")
        print(f"Using FFO: {self._use_ffo}")
        self._registry: Dict[bool, Dict[str, Callable]] = {
            True: {},  # FFO backend
            False: {}  # PyTorch backend
        }
        self._setup_registry()
    
    def _setup_registry(self) -> None:
        """Set up the function registry with appropriate implementations."""
        # Set up FFO backend
        self._registry[True]['grid_sample_2d'] = fused_grid_sampler_2d
        self._registry[True]['grid_sample_3d'] = fused_grid_sampler_3d
        self._registry[True]['warp_composer_2d'] = fused_warp_composer_2d
        self._registry[True]['warp_composer_3d'] = fused_warp_composer_3d
        self._registry[True]['affine_warp_3d'] = fused_affine_warp_3d
        self._registry[True]['affine_warp_2d'] = torch_affine_warp_2d    # TODO: add fused affine warp 2d

        # Set up PyTorch backend
        self._registry[False]['grid_sample_2d'] = torch_grid_sampler_2d
        self._registry[False]['grid_sample_3d'] = torch_grid_sampler_3d
        self._registry[False]['warp_composer_2d'] = torch_warp_composer_2d
        self._registry[False]['warp_composer_3d'] = torch_warp_composer_3d
        self._registry[False]['affine_warp_3d'] = torch_affine_warp_3d
        self._registry[False]['affine_warp_2d'] = torch_affine_warp_2d
    
    @property
    def use_ffo(self) -> bool:
        """Get the current backend state."""
        return self._use_ffo
    
    @use_ffo.setter
    def use_ffo(self, value: bool) -> None:
        """Set the backend state. Raises ValueError if trying to use FFO when not available."""
        print(f"Setting use_ffo to {value}")
        if value and not FFO_AVAILABLE:
            raise ValueError("Cannot set use_ffo to True when fused operations are not available")
        self._use_ffo = value
    
    def _get_image_dim(self, *args, image_idx: int = 0, arg_name: str = "input", **kwargs) -> int:
        ''' return spatial dimensions of image '''
        image = kwargs.get(arg_name, None)
        if image is None:
            image = args[image_idx]
        if isinstance(image, torch.Tensor):
            return image.ndim - 2
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    @staticmethod
    def augment_registry_backend(func: Callable):
        ''' For each function, we want to fallback to torch if the fused ops cannot run on the tensors' device '''
        def wrapper(self, *args, **kwargs):
            prev_ffo = self._use_ffo
            # get any tensor arg
            tensor_arg = [arg for arg in args if isinstance(arg, torch.Tensor)] + [arg for arg in kwargs.values() if isinstance(arg, torch.Tensor)]
            if len(tensor_arg) > 0 and not ffo_supports_tensors(tensor_arg, kwargs.get("mode")):
                self._use_ffo = False
            ret = func(self, *args, **kwargs)
            self._use_ffo = prev_ffo
            return ret
        return wrapper
    
    @augment_registry_backend.__func__
    def __call__(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate grid sample implementation."""
        mode = kwargs.get("mode", "bilinear")
        if mode == "genericlabel":
            # Reroute to generic-label fused sampler (single tensor, return_probs=False)
            if not self._use_ffo or fused_grid_sampler_2d_generic_label is None:
                raise RuntimeError(
                    "genericlabel interpolation requires fused ops (FFO). "
                    "Either compile fused ops or use is_onehot=True for segmentation images."
                )
            dim = self._get_image_dim(*args, **kwargs)
            input_tensor = kwargs.get("input", args[0] if args else None)
            if dim == 2:
                return fused_grid_sampler_2d_generic_label(*args, **kwargs)
            return fused_grid_sampler_3d_generic_label(*args, **kwargs)
        # otherwise, dispatch to the appropriate grid sample implementation
        dim = self._get_image_dim(*args, **kwargs)
        return self._registry[self._use_ffo][f'grid_sample_{dim}d'](*args, **kwargs)
    
    @augment_registry_backend.__func__
    def warp_composer(self, *args, **kwargs) -> Any:
        ''' Dispatch to appropriate warp composer implementation '''
        dim = self._get_image_dim(*args, **kwargs)
        return self._registry[self._use_ffo][f'warp_composer_{dim}d'](*args, **kwargs)
    
    @augment_registry_backend.__func__
    def affine_warp(self, *args, **kwargs) -> Any:
        ''' Dispatch to appropriate affine warp implementation '''
        dim = self._get_image_dim(*args, image_idx=1, **kwargs, arg_name="grid")
        return self._registry[self._use_ffo][f'affine_warp_{dim}d'](*args, **kwargs)

    # individual functions are called by the user at their discretion, we do not need to check
    # for the backend

    def warp_composer_2d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 2D warp composer implementation."""
        return self._registry[self._use_ffo]['warp_composer_2d'](*args, **kwargs)
    
    def warp_composer_3d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 3D warp composer implementation."""
        return self._registry[self._use_ffo]['warp_composer_3d'](*args, **kwargs)
    
    def grid_sample_2d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 2D grid sample implementation."""
        return self._registry[self._use_ffo]['grid_sample_2d'](*args, **kwargs)
    
    def grid_sample_3d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 3D grid sample implementation."""
        return self._registry[self._use_ffo]['grid_sample_3d'](*args, **kwargs)
    
    def __str__(self) -> str:
        """String representation of the dispatcher."""
        return f"GridSampleDispatcher(use_ffo={self._use_ffo}, registry={self._registry})"
    
    def affine_warp_3d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 3D affine warp implementation."""
        return self._registry[self._use_ffo]['affine_warp_3d'](*args, **kwargs)
    
    def affine_warp_2d(self, *args, **kwargs) -> Any:
        """Dispatch to appropriate 2D affine warp implementation."""
        return self._registry[self._use_ffo]['affine_warp_2d'](*args, **kwargs)


# Create a singleton instance
fireants_interpolator = GridSampleDispatcher()
