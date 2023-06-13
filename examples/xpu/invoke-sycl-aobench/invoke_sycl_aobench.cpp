// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include <ispcrt.h>

#ifdef AOBENCH_TARGET_SYCL
#include <sycl/sycl.hpp>
// ze_api and sycl level zero backend must be in this order
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// sycl include must come first
#include "ao_sycl_kernel.inl"
#endif

#ifndef ISPC_SIMD_WIDTH
#error "Pass -DISPC_SIMD_WIDTH=<ispc kernel simd width>"
#endif

static unsigned char *img;

static void savePPM(const char *fn, int w, int h, uint8_t *buf) {
    FILE *fp = fopen(fn, "wb");
    if (!fp) {
        perror(fn);
        exit(1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(buf, w * h * 3, 1, fp);
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

#ifndef AOBENCH_TARGET_SYCL
struct alignas(16) vec3f {
    float x = 0;
    float y = 0;
    float z = 0;
};

struct Sphere {
    vec3f center;
    float radius = 0;
};

struct Plane {
    vec3f p, n;
};

struct Scene {
    Plane *planes = nullptr;
    Sphere *spheres = nullptr;

    unsigned int n_planes = 0;
    unsigned int n_spheres = 0;
};

struct Parameters {
    int width = 256;
    int height = 256;
    int y_offset = 0;
    int n_samples = 4;
    Scene *scene = nullptr;
    int *rng_seeds = nullptr;
    float *image = nullptr;
};
#else
// We place the SYCL objects in a struct so we can
// force them to be destroyed before the ISPCRT objects
struct SYCLObjects {
    sycl::platform platform;
    sycl::device device;
    sycl::context context;
    sycl::queue queue;

    SYCLObjects(ISPCRTDevice ispcdevice, ISPCRTTaskQueue ispcqueue) {
        platform = sycl::ext::oneapi::level_zero::make_platform(
            reinterpret_cast<pi_native_handle>(ispcrtPlatformNativeHandle(ispcdevice)));
        device = sycl::ext::oneapi::level_zero::make_device(
            platform, reinterpret_cast<pi_native_handle>(ispcrtDeviceNativeHandle(ispcdevice)));

        context = sycl::ext::oneapi::level_zero::make_context(
            std::vector<sycl::device>{device},
            reinterpret_cast<pi_native_handle>(ispcrtDeviceContextNativeHandle(ispcdevice)), true);

        queue = sycl::queue(context, device, {sycl::property::queue::enable_profiling()});
    }
};
#endif

int main(int argc, char *argv[]) {
    const int numIter = 3;
    ispcrtSetErrorFunc([](ISPCRTError e, const char *m) {
        std::cerr << "ISPCRT Error! --> " << m << std::endl;
        std::exit(1);
    });

    // Init compute device (CPU or GPU) //
    auto device = ispcrtGetDevice(ISPCRT_DEVICE_TYPE_GPU, 0);

#ifndef AOBENCH_TARGET_SYCL
    // Load ISPC module
    // libraryCompilation is needed when linking is required
    ISPCRTModuleOptions opts_vector = ispcrtNewModuleOptions(device, ISPCRT_VECTOR_MODULE, false, 0);
#ifdef AOBENCH_SYCL_LINKING
    // Load ISPC module with invoke_sycl call to SYCL
    auto runModule = ispcrtLoadModuleWithOptions(device, "ao_ispc_sycl", opts_vector);
#else
    // Load ISPC-only module
    auto runModule = ispcrtLoadModuleWithOptions(device, "ao_ispc", opts_vector);
#endif

#ifdef AOBENCH_SYCL_LINKING
    // For loading the SYCL library, specify that it's a scalar module
    ISPCRTModuleOptions opts_scalar = ispcrtNewModuleOptions(device, ISPCRT_SCALAR_MODULE, true, 0);
    auto libraryModule = ispcrtLoadModuleWithOptions(device, "ao_sycl_lib", opts_scalar);
    std::array<ISPCRTModule, 2> modules = {runModule, libraryModule};
#ifdef VISA_LINKING
    // vISA linking, the result of a linking is a new module
    auto linkedModule = ispcrtStaticLinkModules(device, modules.data(), modules.size());
    auto kernel = ispcrtNewKernel(device, linkedModule, "compute_ao_tile");
#else
    // binary linking, the result of the linking is a entry module with all deps
    // resolved
    ispcrtDynamicLinkModules(device, modules.data(), modules.size());
    auto kernel = ispcrtNewKernel(device, runModule, "compute_ao_tile");
#endif
#else
    auto kernel = ispcrtNewKernel(device, runModule, "compute_ao_tile");
#endif
#endif

    // Create task queue and execute kernel
    auto queue = ispcrtNewTaskQueue(device);

    ISPCRTNewMemoryViewFlags flags;
    flags.allocType = ISPCRT_ALLOC_TYPE_DEVICE;

    std::vector<Plane> planes = {Plane{vec3f{0.f, -0.5f, 0.f}, vec3f{0.f, 1.f, 0.f}}};
    std::vector<Sphere> spheres = {Sphere{vec3f{-2.f, 0.f, -3.5f}, 0.5f}, Sphere{vec3f{-0.5f, 0.f, -3.f}, 0.5f},
                                   Sphere{vec3f{1.f, 0.f, -2.2f}, 0.5f}};
    auto buf_planes = ispcrtNewMemoryView(device, planes.data(), planes.size() * sizeof(Plane), &flags);
    auto buf_spheres = ispcrtNewMemoryView(device, spheres.data(), spheres.size() * sizeof(Sphere), &flags);
    Scene scene;
    scene.planes = static_cast<Plane *>(ispcrtDevicePtr(buf_planes));
    scene.spheres = static_cast<Sphere *>(ispcrtDevicePtr(buf_spheres));
    scene.n_planes = planes.size();
    scene.n_spheres = spheres.size();
    auto scene_dev = ispcrtNewMemoryView(device, &scene, sizeof(scene), &flags);

    // Parameters data
    Parameters parameters;
    std::vector<float> imgBuf(parameters.width * parameters.height);
    std::fill(imgBuf.begin(), imgBuf.end(), 0.f);
    img = new unsigned char[parameters.width * parameters.height];

    // Compute the RNG seeds
    std::mt19937 rng;
    std::vector<int> rngBuf(parameters.width * parameters.height, 0);
    for (auto &s : rngBuf) {
        s = rng();
    }

    auto buf_rng_seeds = ispcrtNewMemoryView(device, rngBuf.data(), rngBuf.size() * sizeof(int), &flags);
    auto buf_image = ispcrtNewMemoryView(device, imgBuf.data(), imgBuf.size() * sizeof(float), &flags);
    parameters.rng_seeds = static_cast<int *>(ispcrtDevicePtr(buf_rng_seeds));
    parameters.image = static_cast<float *>(ispcrtDevicePtr(buf_image));
    parameters.scene = static_cast<Scene *>(ispcrtDevicePtr(scene_dev));

    auto p_dev = ispcrtNewMemoryView(device, &parameters, sizeof(parameters), &flags);

    // Copy data to the device
    ispcrtCopyToDevice(queue, buf_planes);
    ispcrtCopyToDevice(queue, buf_spheres);
    ispcrtCopyToDevice(queue, scene_dev);
    ispcrtCopyToDevice(queue, buf_rng_seeds);
    ispcrtCopyToDevice(queue, p_dev);
    ispcrtSync(queue);

#ifdef AOBENCH_TARGET_SYCL
    auto syclObjects = std::make_unique<SYCLObjects>(device, queue);
#endif

    // Now run the AO compute kernel
    for (int i = 0; i < numIter; i++) {
#ifndef AOBENCH_TARGET_SYCL
        auto res = ispcrtLaunch2D(queue, kernel, p_dev, parameters.width / ISPC_SIMD_WIDTH, parameters.height);
        ispcrtRetain(res);
#else
        Parameters *p = reinterpret_cast<Parameters *>(ispcrtDevicePtr(p_dev));
        auto syclEvent = syclObjects->queue.submit([&](sycl::handler &cgh) {
            const sycl::range<2> launchDims(parameters.width, parameters.height);
            cgh.parallel_for(launchDims, [=](sycl::item<2> taskIndex) {
                const uniform int tile_x = taskIndex.get_id(0);
                const uniform int tile_y = taskIndex.get_id(1) + p->y_offset;
                compute_tile(p, tile_x, tile_y);
            });
        });
        syclEvent.wait_and_throw();
#endif
        ispcrtCopyToHost(queue, buf_image);
        ispcrtSync(queue);

        // Get the kernel execution time
        double kernelTicks = 1e30;
#ifndef AOBENCH_TARGET_SYCL
        if (ispcrtFutureIsValid(res)) {
            kernelTicks = ispcrtFutureGetTimeNs(res) * 1e-6;
        }
        ispcrtRelease(res);
#else
        const uint64_t t0 = syclEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 = syclEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
        // SYCL times are also in nanoseconds
        kernelTicks = (t1 - t0) * 1e-6;
#endif
        printf("@time of #%i run:\t\t\t[%.3f] milliseconds\n", i, kernelTicks);
    }

    std::vector<uint8_t> rgb_image;
    for (const auto &p : imgBuf) {
        const uint8_t val = std::max(std::min(p * 255.f, 255.f), 0.f);
        rgb_image.push_back(val);
        rgb_image.push_back(val);
        rgb_image.push_back(val);
    }
    // Write resulting images
    const std::string sycl_reference_fname = "ao_sycl_reference_" + std::to_string(parameters.n_samples) + ".ppm";
    const std::string dpcpp_fname = "ao_sycl_" + std::to_string(parameters.n_samples) + ".ppm";
    const std::string validation_fname = "ao_ispc_" + std::to_string(parameters.n_samples) + ".ppm";
#ifdef AOBENCH_SYCL_LINKING
    const std::string outname = dpcpp_fname;
#elif defined(AOBENCH_TARGET_SYCL)
    const std::string outname = sycl_reference_fname;
#else
    const std::string outname = validation_fname;
#endif
    savePPM(outname.c_str(), parameters.width, parameters.height, rgb_image.data());

    // Free memory
    delete[] img;

#ifdef AOBENCH_TARGET_SYCL
    syclObjects = nullptr;
#endif

    ispcrtRelease(queue);
#ifndef AOBENCH_TARGET_SYCL
    ispcrtRelease(kernel);
    ispcrtRelease(runModule);
    ispcrtRelease(opts_vector);
#ifdef AOBENCH_SYCL_LINKING
    ispcrtRelease(libraryModule);
    ispcrtRelease(opts_scalar);
#ifdef VISA_LINKING
    ispcrtRelease(linkedModule);
#endif
#endif
#endif
    ispcrtRelease(buf_planes);
    ispcrtRelease(buf_spheres);
    ispcrtRelease(scene_dev);
    ispcrtRelease(buf_rng_seeds);
    ispcrtRelease(buf_image);
    ispcrtRelease(p_dev);
    ispcrtRelease(device);
    return 0;
}
