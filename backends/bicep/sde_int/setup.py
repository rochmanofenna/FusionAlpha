from torch.utils.cpp_extension import load

sde_ext = load(
    name="sde_ext",
    sources=[
      "backends/bicep/sde_int/curand_kernel.cu",
      "backends/bicep/sde_int/binding.cpp",
    ],
    extra_cuda_cflags=["-O3","-I/usr/local/cuda/include"],
    extra_ldflags=["-lcurand"],
    verbose=True,
)
