"""GPU environment setup for TensorFlow 2.15 in the RG conda env.

The `nvidia-*-cu12` pip wheels ship the CUDA 12.2 / cuDNN 8.9 shared libraries that
TF 2.15 needs, but the CPU-built TF wheel does NOT find them automatically — the dynamic
loader only searches `LD_LIBRARY_PATH`, which it reads once at process start. So we must
prepend the wheel lib dirs and then re-exec the interpreter so the change takes effect.

`configure_gpu_env()` MUST be called before `import tensorflow`. It is a no-op (no re-exec)
when running CPU-only (`CUDA_VISIBLE_DEVICES=-1`) or when the nvidia wheels aren't installed,
so the CPU fallback path costs nothing.

PyTorch translation: none of the above applies. The torch wheels (torch==2.13.0+cu126) bundle
their CUDA/cuDNN libraries and load them themselves, and torch only touches a GPU when a tensor
is moved to it (experiment.py --device). Every function below is kept with its original name and
signature so the call sites in experiment.py / run_experiments.py stay structurally identical,
but they are no-ops. The original TF implementation is kept as `#TF:` comments.
"""

import glob
import os
import sys

_READY_FLAG = "_RG_GPU_ENV_READY"


def _nvidia_lib_dirs():
    """Return the lib/ dirs of the installed nvidia-*-cu12 pip wheels (if any).

    PyTorch: unused (torch finds its bundled CUDA libraries itself); returns []."""
    #TF: try:
    #TF:     import nvidia  # namespace package provided by the nvidia-*-cu12 wheels
    #TF: except ImportError:
    #TF:     return []
    #TF: dirs = []
    #TF: for base in list(nvidia.__path__):
    #TF:     dirs += glob.glob(os.path.join(base, "*", "lib"))
    #TF: return sorted(set(dirs))
    return []


def _nvidia_bin_dirs():
    """Return bin/ dirs (e.g. ptxas from nvidia-cuda-nvcc) to add to PATH.

    PyTorch: unused; returns []."""
    #TF: try:
    #TF:     import nvidia
    #TF: except ImportError:
    #TF:     return []
    #TF: dirs = []
    #TF: for base in list(nvidia.__path__):
    #TF:     dirs += glob.glob(os.path.join(base, "*", "bin"))
    #TF: return sorted(set(dirs))
    return []


def configure_gpu_env():
    """Prepend pip CUDA libs to LD_LIBRARY_PATH (re-exec once) so TF can use the GPU.

    Idempotent and inherited across subprocesses via the _RG_GPU_ENV_READY guard.

    PyTorch: no-op. Torch's CUDA runtime is bundled with the wheel; no LD_LIBRARY_PATH
    manipulation and no re-exec are needed. The guard flag is still set so subprocess
    launchers behave as before."""
    if os.environ.get(_READY_FLAG) == "1":
        return
    os.environ[_READY_FLAG] = "1"

    #TF: # Explicit CPU request -> nothing to configure.
    #TF: if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    #TF:     return
    #TF:
    #TF: lib_dirs = _nvidia_lib_dirs()
    #TF: if not lib_dirs:
    #TF:     # No pip CUDA libs installed; TF will fall back to CPU on its own.
    #TF:     return
    #TF:
    #TF: current = os.environ.get("LD_LIBRARY_PATH", "")
    #TF: parts = [p for p in current.split(os.pathsep) if p]
    #TF: if all(d in parts for d in lib_dirs):
    #TF:     return  # already on the path (e.g. inherited from parent)
    #TF:
    #TF: os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + parts)
    #TF:
    #TF: bin_dirs = _nvidia_bin_dirs()
    #TF: if bin_dirs:
    #TF:     path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    #TF:     os.environ["PATH"] = os.pathsep.join(bin_dirs + path_parts)
    #TF:
    #TF: # Re-exec so the dynamic loader picks up the new LD_LIBRARY_PATH.
    #TF: os.execv(sys.executable, [sys.executable] + sys.argv)
    return


def cuda_ld_library_path():
    """pathsep-joined lib dirs of the pip CUDA wheels (empty string if none).

    PyTorch: always "" (nothing to inject into child environments)."""
    return os.pathsep.join(_nvidia_lib_dirs())


def cuda_bin_path():
    """pathsep-joined bin dirs of the pip CUDA wheels (empty string if none).

    PyTorch: always ""."""
    return os.pathsep.join(_nvidia_bin_dirs())


def enable_memory_growth(framework):
    """Allow multiple job processes to share one GPU without pre-allocating all VRAM.

    PyTorch: no-op. Torch's caching allocator already grows on demand and never
    pre-allocates the whole device, so concurrent jobs share a GPU without any setting
    (the argument used to be the `tf` module; it is ignored)."""
    #TF: for gpu in tf.config.list_physical_devices("GPU"):
    #TF:     try:
    #TF:         tf.config.experimental.set_memory_growth(gpu, True)
    #TF:     except RuntimeError:
    #TF:         # Already initialized; safe to ignore.
    #TF:         pass
    return
