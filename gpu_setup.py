"""GPU environment setup for TensorFlow 2.15 in the RG conda env.

The `nvidia-*-cu12` pip wheels ship the CUDA 12.2 / cuDNN 8.9 shared libraries that
TF 2.15 needs, but the CPU-built TF wheel does NOT find them automatically — the dynamic
loader only searches `LD_LIBRARY_PATH`, which it reads once at process start. So we must
prepend the wheel lib dirs and then re-exec the interpreter so the change takes effect.

`configure_gpu_env()` MUST be called before `import tensorflow`. It is a no-op (no re-exec)
when running CPU-only (`CUDA_VISIBLE_DEVICES=-1`) or when the nvidia wheels aren't installed,
so the CPU fallback path costs nothing.
"""

import glob
import os
import sys

_READY_FLAG = "_RG_GPU_ENV_READY"


def _nvidia_lib_dirs():
    """Return the lib/ dirs of the installed nvidia-*-cu12 pip wheels (if any)."""
    try:
        import nvidia  # namespace package provided by the nvidia-*-cu12 wheels
    except ImportError:
        return []
    dirs = []
    for base in list(nvidia.__path__):
        dirs += glob.glob(os.path.join(base, "*", "lib"))
    return sorted(set(dirs))


def _nvidia_bin_dirs():
    """Return bin/ dirs (e.g. ptxas from nvidia-cuda-nvcc) to add to PATH."""
    try:
        import nvidia
    except ImportError:
        return []
    dirs = []
    for base in list(nvidia.__path__):
        dirs += glob.glob(os.path.join(base, "*", "bin"))
    return sorted(set(dirs))


def configure_gpu_env():
    """Prepend pip CUDA libs to LD_LIBRARY_PATH (re-exec once) so TF can use the GPU.

    Idempotent and inherited across subprocesses via the _RG_GPU_ENV_READY guard.
    """
    if os.environ.get(_READY_FLAG) == "1":
        return
    os.environ[_READY_FLAG] = "1"

    # Explicit CPU request -> nothing to configure.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return

    lib_dirs = _nvidia_lib_dirs()
    if not lib_dirs:
        # No pip CUDA libs installed; TF will fall back to CPU on its own.
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if all(d in parts for d in lib_dirs):
        return  # already on the path (e.g. inherited from parent)

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + parts)

    bin_dirs = _nvidia_bin_dirs()
    if bin_dirs:
        path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        os.environ["PATH"] = os.pathsep.join(bin_dirs + path_parts)

    # Re-exec so the dynamic loader picks up the new LD_LIBRARY_PATH.
    os.execv(sys.executable, [sys.executable] + sys.argv)


def cuda_ld_library_path():
    """pathsep-joined lib dirs of the pip CUDA wheels (empty string if none)."""
    return os.pathsep.join(_nvidia_lib_dirs())


def cuda_bin_path():
    """pathsep-joined bin dirs of the pip CUDA wheels (empty string if none)."""
    return os.pathsep.join(_nvidia_bin_dirs())


def enable_memory_growth(tf):
    """Allow multiple job processes to share one GPU without pre-allocating all VRAM."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Already initialized; safe to ignore.
            pass
