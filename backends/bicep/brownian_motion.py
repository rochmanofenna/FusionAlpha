import os
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

# pick the array backend once
USE_CUPY = cp is not None and os.getenv("DISABLE_CUPY") != "1"
_xp = cp if USE_CUPY else np

import os
import time
import numpy as _np
import psutil
import logging
import numpy as np, cupy as cp

from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from .stochastic_control import apply_stochastic_controls

# ——— Logging setup ———
LOG_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir, os.pardir,
    "results", "logs", "simulation.log"
)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def detect_system_resources():
    mem = psutil.virtual_memory().available / (1024**3)
    cpu = psutil.cpu_count(logical=False)
    try:
        import cupy as cp
        gpu = True
        gmem = cp.cuda.Device(0).mem_info[1] / (1024**3)
    except Exception:
        gpu, gmem = False, 0
    logging.info(f"Resources: mem={mem:.2f}GB cpu={cpu} gpu={gpu} gmem={gmem:.2f}GB")
    return mem, cpu, gpu, gmem



def calculate_optimal_parameters(n_paths, n_steps, mem, cpu, gpu, gmem):
    est_mb = n_steps * 4 / 2**20
    batch = min(max(1, int(mem*1024//est_mb)), n_paths, cpu*100)
    gpu_thr = 1000 if gpu and gmem>=8 else n_paths+1
    save_int = max(2000, batch*2)
    logging.info(f"Params: batch={batch} gpu_thr={gpu_thr} save_int={save_int}")
    return batch, save_int, gpu_thr

def setup_dask_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit="2GB")
    client = Client(cluster)
    logging.info("Dask cluster started")
    return client

def simulate_batch(T, n_steps, n_paths, xp, dir_bias=0.0, var_adj=None):
    dt = T / n_steps
    # shape (n_paths, n_steps)
    inc = xp.random.normal(dir_bias, 1.0, size=(n_paths, n_steps)) * xp.sqrt(dt)
    if var_adj is not None:
        # shape (n_steps,)
        scale = xp.asarray(var_adj(xp.linspace(dt, T, n_steps)))
        inc *= scale[None, :]
    out = xp.empty((n_paths, n_steps+1), dtype=inc.dtype)
    out[:, 0] = 0.0
    out[:, 1:] = xp.cumsum(inc, axis=1)
    return out

def simulate_single_path(T, n_steps, initial_value, dt,
                         directional_bias, variance_adjustment,
                         xp, apply_ctrl):
    state_count = 1
    feedback = 0.5
    t0 = 0.1
    total = n_steps

    times = xp.linspace(0, T, n_steps+1)
    inc = xp.random.normal(directional_bias, 1.0, size=n_steps)
    if variance_adjustment:
        var_scale = variance_adjustment(times[1:])
        inc *= xp.sqrt(var_scale * dt)
    else:
        inc *= xp.sqrt(dt)

    for i in range(n_steps):
        inc[i] = apply_ctrl(inc[i], state_count, feedback, t0, total)

    path = xp.empty(n_steps+1, dtype=inc.dtype)
    path[0]= initial_value
    path[1:]= xp.cumsum(inc) + initial_value
    return path

def brownian_motion_paths(
    T,
    n_steps,
    *,
    initial_value=0.0,
    n_paths=10,
    directional_bias=0.0,
    variance_adjustment=None,
    batch=2000,        # paths per chunk
):
    """
    Vectorized CPU/GPU implementation:
      – on GPU (CuPy) it will slice into chunks of size `batch`
        but still use a single cumsum per chunk
      – on CPU (NumPy) same pattern
    """
    if T <= 0:
        raise ValueError("Time duration T must be greater than 0.")
    if n_steps <= 0:
        raise ValueError("Number of steps n_steps must be greater than 0.")
    xp = _xp
    dt = T / n_steps

    # time‐grid
    tg = xp.linspace(0, T, n_steps + 1)

    # if zero paths, short‐circuit
    if n_paths == 0:
        return tg, xp.empty((0, n_steps + 1))

    # build chunks of up to `batch` paths
    pieces = []
    for start in range(0, n_paths, batch):
        size = min(batch, n_paths - start)
        chunk = simulate_batch(
            T, n_steps, size,
            xp,
            dir_bias=directional_bias,
            var_adj=variance_adjustment,
        )
        pieces.append(chunk)

    # stitch back together & shift by initial_value
    paths = xp.concatenate(pieces, axis=0) + initial_value
    return tg, paths

def brownian_motion_paths_gpu_stream(
    T: float,
    n_steps: int,
    n_paths: int,
    batch: int = 1024,
    initial_value: float = 0.0,
    directional_bias: float = 0.0,
    variance_adjustment=None,
):
    """
    Simulate in batch‐sized chunks on the GPU, copy each chunk back into
    a host‐side NumPy memmap, and never try to concatenate one giant CuPy array.
    """
    if cp is None:
        raise RuntimeError("Cupy not available")
    # 1) build a GPU time‐grid & then copy to host
    tg_gpu = cp.linspace(0, T, n_steps+1)
    tg = tg_gpu.get()  # now a NumPy array on host

    # 2) prepare a host‐side memmap to hold every path
    fname = "brownian_paths.dat"
    host_paths = np.memmap(
        fname, mode="w+",
        dtype="float32",
        shape=(n_paths, n_steps+1)
    )

    pool = cp.get_default_memory_pool()

    # 3) stream chunk by chunk
    for start in range(0, n_paths, batch):
        size = min(batch, n_paths - start)
        # simulate a full batch on GPU
        chunk_gpu = simulate_batch(
            T, n_steps, size,
            xp=cp,
            dir_bias=directional_bias,
            var_adj=variance_adjustment
        )
        # shift by initial_value
        chunk_gpu += initial_value

        # copy it back
        host_paths[start:start+size] = chunk_gpu.get()

        # free as much GPU memory as possible
        pool.free_all_blocks()

    # 4) return the host copies
    return tg, host_paths

def brownian_motion_paths_cpu_stream(
    T: float,
    n_steps: int,
    n_paths: int,
    batch: int,
    filename: str = "cpu_paths.dat"
):
    """
    Generate Brownian paths in NumPy but stream them to disk in `batch`-sized chunks
    so you never have the full (n_paths x (n_steps+1)) array in RAM.
    """
    import numpy as np
    dt = T / n_steps
    # time‐grid in memory (tiny)
    tg = np.linspace(0.0, T, n_steps + 1)

    # open a float32 memmap on disk for the full result
    mm = np.memmap(
        filename,
        mode="w+",
        dtype=np.float32,
        shape=(n_paths, n_steps + 1),
    )

    # generate chunk-by-chunk
    for start in range(0, n_paths, batch):
        size = min(batch, n_paths - start)
        # simulate_batch is your existing function, just pass xp=np
        paths = simulate_batch(
            T,                 # total time
            n_steps,           # steps per path
            size,              # batch size
            np,                # numpy backend
            dir_bias=0.0,      # if you have a bias
            var_adj=None       # or a variance adjuster
        )
        # write it directly into the memmap (only these rows in RAM at once)
        mm[start : start + size, :] = paths

    # flush to disk and return
    mm.flush()
    return tg, mm

