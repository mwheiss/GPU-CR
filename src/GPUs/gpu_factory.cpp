#include "GPU.h"

#ifdef __HIP_PLATFORM_AMD__
#include "AMD/amd.h"
#else
#include "NVIDIA/nv.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

GPU* createGPU() {
    const char* vendor = getenv("GPU_VENDOR");

    if (!vendor) {
#ifdef __HIP_PLATFORM_AMD__
        vendor = "AMD";
#else
        vendor = "NVIDIA";
#endif
        fprintf(stderr, "[GPU Factory] GPU_VENDOR not set; using build target %s\n", vendor);
    }

#ifdef __HIP_PLATFORM_AMD__
    if (strcasecmp(vendor, "AMD") == 0 || strcasecmp(vendor, "ROCM") == 0 || strcasecmp(vendor, "HIP") == 0) {
        fprintf(stderr, "[GPU Factory] Creating AMD GPU instance\n");
        return new amd();
    } else {
        fprintf(stderr, "[GPU Factory] ERROR: This build only supports AMD GPUs (vendor='%s')\n", vendor);
        exit(EXIT_FAILURE);
    }
#else
    if (strcasecmp(vendor, "NVIDIA") == 0 || strcasecmp(vendor, "CUDA") == 0) {
        fprintf(stderr, "[GPU Factory] Creating NVIDIA GPU instance\n");
        return new nv();
    } else {
        fprintf(stderr, "[GPU Factory] ERROR: This build only supports NVIDIA GPUs (vendor='%s')\n", vendor);
        exit(EXIT_FAILURE);
    }
#endif
}
