#include <torch/extension.h>

// prototype for your NVCC‐compiled launcher
extern "C" void sde_curand_launch_cu(
    float*, int, int, float, float, float, float, float, float, float, int
);

void sde_curand_launch(
    torch::Tensor paths,
    int           n_steps,
    int           stride,
    float         T,
    float         feedback_value,
    float         decay_rate,
    float         high_threshold,
    float         low_threshold,
    float         total_steps,
    float         base_variance
) {
    sde_curand_launch_cu(
        paths.data_ptr<float>(),
        n_steps,
        stride,
        T,
        feedback_value,
        decay_rate,
        high_threshold,
        low_threshold,
        total_steps,
        base_variance,
        /*n_paths=*/paths.size(0)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sde_curand", &sde_curand_launch, "SDE with CURAND");
}
