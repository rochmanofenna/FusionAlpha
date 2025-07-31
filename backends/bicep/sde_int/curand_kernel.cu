#include <curand_kernel.h>
#include <cuda_runtime.h>

extern "C" __global__ void sde_curand_kernel(
    float* paths,
    int    n_steps,
    int    stride,
    float  T,
    float  feedback_value,
    float  decay_rate,
    float  high_threshold,
    float  low_threshold,
    float  total_steps,
    float  base_variance
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    float dt = T / n_steps;
    curandStatePhilox4_32_10_t state;
    curand_init(1234ULL, pid, 0, &state);

    float acc = paths[pid * stride];
    for (int i = 0; i < n_steps; ++i) {
        float rnd    = curand_normal(&state);
        float norm0  = 1.0f / total_steps;
        float factor = norm0 < low_threshold
                         ? 1.5f
                         : (norm0 > high_threshold ? 0.5f : 1.0f);
        float t      = i * dt;
        float vf     = base_variance * factor * expf(-decay_rate * t);
        float tmp    = 0.5f + feedback_value * 0.5f;
        float scale2 = fminf(1.0f, fmaxf(0.2f, tmp));
        float inc    = rnd * sqrtf(dt) * scale2 * vf;
        acc += inc;
        paths[pid * stride + i + 1] = acc;
    }
}

extern "C" void sde_curand_launch_cu(
    float* paths,
    int    n_steps,
    int    stride,
    float  T,
    float  feedback_value,
    float  decay_rate,
    float  high_threshold,
    float  low_threshold,
    float  total_steps,
    float  base_variance,
    int    n_paths      // ← you need to pass in number of paths too
) {
    const int threads = 256;
    const int blocks  = (n_paths + threads - 1) / threads;

    // **THIS** is how you launch your kernel in host code:
    sde_curand_kernel<<<blocks, threads>>>(
        paths,
        n_steps,
        stride,
        T,
        feedback_value,
        decay_rate,
        high_threshold,
        low_threshold,
        total_steps,
        base_variance
    );
    cudaDeviceSynchronize();
}
