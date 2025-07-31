import triton
import triton.language as tl
import torch
import time

# 1) Triton kernel with bespoke Box–Muller RNG
@triton.jit
def fused_sde_boxmuller_kernel(path_ptr, n_steps, stride, T,
                               feedback_value, decay_rate,
                               high_threshold, low_threshold,
                               total_steps, base_variance):
    pid    = tl.program_id(0)
    path   = path_ptr + pid * stride
    dt     = T / n_steps
    acc    = tl.load(path)   # initial

    for i in range(n_steps):
        # two uniform draws
        u1 = tl.rand(seed=pid, offset=2 * i)
        u2 = tl.rand(seed=pid, offset=2 * i + 1)
        # Box–Muller → standard normal
        z0 = tl.sqrt(-2.0 * tl.log(u1)) * tl.cos(2 * 3.141592653589793 * u2)
        rnd = z0

        # your existing control math…
        norm    = 1.0 / total_steps
        factor1 = tl.where(norm < low_threshold,
                           1.5,
                           tl.where(norm > high_threshold, 0.5, 1.0))
        t       = i * dt
        vf      = base_variance * factor1 * tl.exp(-decay_rate * t)
        scale2  = tl.clamp(0.5 + feedback_value * 0.5, 0.2, 1.0)

        inc = rnd * tl.sqrt(dt) * scale2 * vf
        acc += inc
        tl.store(path + i + 1, acc)

# 2) Host‐side benchmark harness
n_paths, n_steps = 1024, 1000
stride = n_steps + 1
paths  = torch.zeros((n_paths, stride), device='cuda', dtype=torch.float32)
grid   = (n_paths,)

# control params
feedback_value  = 0.5
decay_rate      = 0.1
high_threshold  = 10.0
low_threshold   = 2.0
total_steps     = float(n_steps)
base_variance   = 1.0

# warm-up compile
fused_sde_boxmuller_kernel[grid](
    paths, n_steps, stride, 1.0,
    feedback_value, decay_rate,
    high_threshold, low_threshold,
    total_steps, base_variance
)

# benchmark
t0 = time.time()
for _ in range(100):
    fused_sde_boxmuller_kernel[grid](
        paths, n_steps, stride, 1.0,
        feedback_value, decay_rate,
        high_threshold, low_threshold,
        total_steps, base_variance
    )
print("Avg box-muller kernel time:", (time.time() - t0)/100)
