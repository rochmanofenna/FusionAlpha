#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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
);

void sde_curand_launch(
    torch::Tensor paths,
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
    float* paths_data = paths.data_ptr<float>();
    int64_t n_paths   = paths.size(0);

    // launch configuration
    const int threads = 256;
    const int blocks  = (n_paths + threads - 1) / threads;

    // fire the kernel
    sde_curand_kernel<<<blocks, threads>>>(
        paths_data,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sde_curand", &sde_curand_launch, "SDE with CURAND");
}
