# Changelog

All notable changes to this project are documented in this file.

### unreleased - CPU kernels for fireants_fused_ops - changes

- `fireants_fused_ops` now always builds CPU (C++/OpenMP) kernels for the 3D grid sampler, warp composer and warp generator, the fused cross-correlation (plus an O(N) box filter), the fused mutual information histograms, the fused Adam update and the FFT Gaussian blur; the CUDA kernels are added when a CUDA toolkit is found (`FIREANTS_FUSED_OPS_CPU_ONLY=1` forces a CPU-only build). Bindings dispatch on the tensor device (`fused_ops/src/DeviceDispatch.cpp`), `ffo.built_with_cuda()` reports the build type
- the CPU kernels are bitwise reproducible from run to run for a fixed `torch.get_num_threads()`: reductions use partial sums per fixed chunk or a fixed chunk-to-buffer assignment instead of per-thread accumulators under dynamic scheduling
- if PyTorch is a CUDA build but no CUDA toolkit is found, `setup.py` builds CPU kernels only and prints a warning (set `CUDA_HOME`, or `FIREANTS_FUSED_OPS_CPU_ONLY=1` to silence it); the Python side checks `built_with_cuda()`, so CUDA tensors then use the PyTorch fallbacks (with a one-time warning) and `fusedcc` / `fusedmi` fall back to `cc` / `mi`
- on macOS the CPU kernels are built with OpenMP when `libomp` is found (`brew --prefix libomp`, or `FIREANTS_OPENMP_CFLAGS` / `FIREANTS_OPENMP_LDFLAGS`), otherwise they run single-threaded; on Windows `torch` must be imported before `fireants_fused_ops` (documented in `fused_ops/README.md`)
- the interpolator dispatcher, `WarpAdam` and `downsample` use the fused ops for CPU tensors when the CPU kernels are available (2D ops and generic-label sampling stay on the torch path on CPU)
- `GreedyRegistration.get_inverse_warp_parameters` now inverts `A x + u(x)` correctly when the affine is not the identity in normalized coordinates (fixed and moving images with different fields of view)
- `WarpAdam` smoothing no longer copies the warp into channels-first memory before the separable convolution (CPU tensors only, CUDA behaviour is unchanged)
- `fireants.io` imports on Windows (`fcntl` is only used where available)
- modified files: `fireants/interpolator/__init__.py`, `fireants/io/image.py`, `fireants/losses/fusedcc.py`, `fireants/registration/abstract.py`, `fireants/registration/greedy.py`, `fireants/registration/optimizers/adam.py`, `fireants/utils/imageutils.py`, `fused_ops/setup.py`, `fused_ops/pyproject.toml`, `fused_ops/README.md`, `docs/docs/install.md`, `fused_ops/src/src.cpp`, `fused_ops/include/CrossCorrelation.h`, `fused_ops/include/MutualInformation.h`; CUDA-only tests now skip without CUDA; new tests in `tests/test_fusedops_cpu.py` and `tests/test_fusedops_fallback.py`

### 2026-04-08 - fireants 1.5.1, fireants_fused_ops 1.2.0 - changes

added `compose_n` option to adam and sgd with reasonable defaults (n=1 for adam, n=10 for sgd)


### 2026-03-19 - fireants 1.5.0, fireants_fused_ops 1.2.0 - changes

- Added Levenberg-Marquardt algorithm for FireANTs


### 2026-02-23 - fireants 1.4.0, fireants_fused_ops 1.2.0 - changes

- added genericlabel support!!! 


### 2026-02-21 - fireants 1.3.2, fireants_fused_ops 1.1.0 - changes

- add masked_fusedcc to test


### 2026-02-21 - fireants 1.3.1, fireants_fused_ops 1.1.0 - changes

- added quarterion-based rigid registration 
- kept legacy rigid for "legacy reasons"


### 2026-02-20 - fireants 1.3.0, fireants_fused_ops 1.1.0 - changes

- added new functionality (2d affine registration of binary shapes using subspace based contour matching: useful for stuff like histology to MRI pre-registration
- makefile with scripts to deploy quicker


### 2026-02-12 - fireants 1.2.0, fireants_fused_ops 1.1.0 - changes

- added grid_sample and warp_composer ops for 2d
- ran unittests in envs with and without fused_ops


### 2026-02-11 - fireants 1.1.2, fireants_fused_ops 1.0.0 - changes

- added 2d cuda kernels of fused ops grid sampler, warp composer
- minor fixes
- add update_versions_and_changelog script for easier project update


### Fixed

- **Data type mismatch**: Fixed data type mismatch in several registration classes (abstract.py, distributedgreedy.py, greedy.py, syn.py). Reference: #77

### Added

- **Masked losses**: Support for region-of-interest (ROI) driven registration using masked variants of CC and MSE.
  - Concatenate a mask as the last channel to fixed and moving images; the loss is computed only where the mask is non-zero.
  - Use `loss_type="masked_cc"` or `loss_type="masked_mse"` to enable masked mode (no extra options required).
  - Helper `generate_image_mask_allones(image)` for images that have no mask (use with `apply_mask_to_image` so both images have a mask channel).
  - Gaussian smoothing in multi-resolution registration is applied only to image channels, not the mask channel, when masked mode is active.
  - See [How To: Masked losses](docs/docs/howto/masked-losses.md) in the documentation.
