// cacheprobe: does VkPipelineCache data written by one process serve compute pipeline creations in another process?
// cacheprobe [options] export <spv> <out-i> <out-ii>      compile one pipeline into a fresh cache and save its data (i); compile it
//                                                         again into an empty cache and save that data (ii); a third creation with no cache
// cacheprobe [options] load <spv> <cache-file|none> [warm.spv]  optionally compile an unrelated pipeline first, then import the file into a
//                                                         new cache and create the pipeline from it
// cacheprobe [options] fill <spv> <count> <out.bin>       compile <count> specialization variants into one cache and save its data
// cacheprobe [options] probe <spv> <cache.bin> <seed>     import a cache and create variant <seed> from it
// Options: --subgroup32 (required subgroup size 32), --full (REQUIRE_FULL_SUBGROUPS), --uab (update-after-bind storage buffer layout),
// --push (a push constant range), --sets2 (a second set layout), --spec N (N specialization constants), --seed N (variant),
// --pre FILE / --post FILE (import another cache file before / after the one under test), --busy N (N threads compiling cold variants
// during the creation), --prefill N (compile N cold variants into a scratch cache first), --device N (physical device index; the default
// is the first NVIDIA device), --any-vendor (take the first device of any vendor).
// Every creation under test prints the VK_EXT_pipeline_creation_feedback hit flag and the driver's duration. Synthetic shaders only.
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    bool subgroup32 = false;
    bool full = false;
    bool uab = false;
    bool push = false;
    bool sets2 = false;
    std::uint32_t spec = 1;
    std::uint32_t seed = 0;
    const char* pre = nullptr;
    const char* post = nullptr;
    std::uint32_t busy = 0;
    std::uint32_t prefill = 0;
    int device = -1;
    bool anyVendor = false;
};

Options options;

[[noreturn]] void fail(const char* what, VkResult result = VK_SUCCESS) {
    std::fprintf(stderr, "cacheprobe: %s (%d)\n", what, static_cast<int>(result));
    std::exit(2);
}

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) fail(what, result);
}

std::vector<char> readFile(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void writeFile(const char* path, const std::vector<char>& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail(path);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

struct Gpu {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    bool feedback = false;
};

Gpu open() {
    Gpu gpu;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "cacheprobe";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &app;
    check(vkCreateInstance(&instanceInfo, nullptr, &gpu.instance), "vkCreateInstance");
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(gpu.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(gpu.instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
    for (std::uint32_t index = 0; index < count; ++index) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[index], &properties);
        std::printf("physical device %u: %s (vendor 0x%04x)\n", index, properties.deviceName, properties.vendorID);
        const bool chosen = options.device >= 0 ? static_cast<int>(index) == options.device : (options.anyVendor || properties.vendorID == 0x10de);
        if (chosen && gpu.physical == VK_NULL_HANDLE) {
            gpu.physical = devices[index];
            gpu.properties = properties;
        }
    }
    if (gpu.physical == VK_NULL_HANDLE) fail(options.device >= 0 ? "no physical device with that index" : "no NVIDIA device (use --device N or --any-vendor)");
    std::uint32_t extensionCount = 0;
    check(vkEnumerateDeviceExtensionProperties(gpu.physical, nullptr, &extensionCount, nullptr), "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(extensionCount);
    check(vkEnumerateDeviceExtensionProperties(gpu.physical, nullptr, &extensionCount, extensions.data()), "vkEnumerateDeviceExtensionProperties");
    const auto has = [&](const char* name) {
        for (const auto& extension : extensions) {
            if (std::strcmp(extension.extensionName, name) == 0) return true;
        }
        return false;
    };
    gpu.feedback = has(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME);
    std::vector<const char*> enabled;
    if (gpu.feedback) enabled.push_back(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME);
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void** tail = &features.pNext;
    if (options.subgroup32 || options.full) {
        if (!has(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) fail("VK_EXT_subgroup_size_control unavailable");
        enabled.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        subgroupFeatures.subgroupSizeControl = VK_TRUE;
        subgroupFeatures.computeFullSubgroups = options.full ? VK_TRUE : VK_FALSE;
        *tail = &subgroupFeatures;
        tail = &subgroupFeatures.pNext;
    }
    if (options.uab) {
        if (!has(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) fail("VK_EXT_descriptor_indexing unavailable");
        enabled.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        enabled.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
        indexingFeatures.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
        *tail = &indexingFeatures;
        tail = &indexingFeatures.pNext;
    }
    std::uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu.physical, &familyCount, families.data());
    std::uint32_t family = familyCount;
    for (std::uint32_t index = 0; index < familyCount; ++index) {
        if (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = index;
            break;
        }
    }
    if (family == familyCount) fail("no compute queue family");
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = family;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &features;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<std::uint32_t>(enabled.size());
    deviceInfo.ppEnabledExtensionNames = enabled.data();
    check(vkCreateDevice(gpu.physical, &deviceInfo, nullptr, &gpu.device), "vkCreateDevice");
    std::string uuid;
    for (const auto byte : gpu.properties.pipelineCacheUUID) {
        char digits[3];
        std::snprintf(digits, sizeof(digits), "%02x", static_cast<unsigned>(byte));
        uuid += digits;
    }
    std::printf("device %s driver 0x%08x api 0x%08x cacheUUID %s feedback %s; options subgroup32=%d full=%d uab=%d push=%d sets2=%d spec=%u seed=%u\n", gpu.properties.deviceName, gpu.properties.driverVersion, gpu.properties.apiVersion, uuid.c_str(), gpu.feedback ? "ext" : "unavailable",
            options.subgroup32, options.full, options.uab, options.push, options.sets2, options.spec, options.seed);
    return gpu;
}

struct Program {
    VkShaderModule module = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> setLayouts;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

VkDescriptorSetLayout setLayout(const Gpu& gpu, bool updateAfterBind) {
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    const VkDescriptorBindingFlagsEXT bindingFlags = updateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT : 0u;
    VkDescriptorSetLayoutBindingFlagsCreateInfoEXT flagsInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT};
    flagsInfo.bindingCount = 1;
    flagsInfo.pBindingFlags = &bindingFlags;
    VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setInfo.pNext = updateAfterBind ? &flagsInfo : nullptr;
    setInfo.flags = updateAfterBind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT : 0u;
    setInfo.bindingCount = 1;
    setInfo.pBindings = &binding;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(gpu.device, &setInfo, nullptr, &layout), "vkCreateDescriptorSetLayout");
    return layout;
}

Program program(const Gpu& gpu, const char* spvPath) {
    Program result;
    const auto code = readFile(spvPath);
    if (code.size() % 4 != 0 || code.size() < 20) fail("not a SPIR-V file");
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = code.size();
    moduleInfo.pCode = reinterpret_cast<const std::uint32_t*>(code.data());
    check(vkCreateShaderModule(gpu.device, &moduleInfo, nullptr, &result.module), "vkCreateShaderModule");
    result.setLayouts.push_back(setLayout(gpu, options.uab));
    if (options.sets2) result.setLayouts.push_back(setLayout(gpu, false));
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = static_cast<std::uint32_t>(result.setLayouts.size());
    layoutInfo.pSetLayouts = result.setLayouts.data();
    layoutInfo.pushConstantRangeCount = options.push ? 1u : 0u;
    layoutInfo.pPushConstantRanges = options.push ? &push : nullptr;
    check(vkCreatePipelineLayout(gpu.device, &layoutInfo, nullptr, &result.layout), "vkCreatePipelineLayout");
    std::printf("shader %s: %zu words\n", spvPath, code.size() / 4);
    return result;
}

VkPipeline create(const Gpu& gpu, const Program& program, VkPipelineCache cache, const char* label) {
    std::vector<std::uint32_t> constantValues;
    std::vector<VkSpecializationMapEntry> entries;
    for (std::uint32_t index = 0; index < options.spec; ++index) {
        constantValues.push_back(7u + index + options.seed * 131u);
        entries.push_back({index, index * 4u, 4u});
    }
    VkSpecializationInfo specialization{static_cast<std::uint32_t>(entries.size()), entries.data(), constantValues.size() * 4u, constantValues.data()};
    VkPipelineCreationFeedbackEXT pipelineFeedback{};
    VkPipelineCreationFeedbackEXT stageFeedback{};
    VkPipelineCreationFeedbackCreateInfoEXT feedbackInfo{VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT};
    feedbackInfo.pPipelineCreationFeedback = &pipelineFeedback;
    feedbackInfo.pipelineStageCreationFeedbackCount = 1;
    feedbackInfo.pPipelineStageCreationFeedbacks = &stageFeedback;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
    subgroup.requiredSubgroupSize = 32;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.pNext = gpu.feedback ? &feedbackInfo : nullptr;
    info.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.pNext = (options.subgroup32 || options.full) ? &subgroup : nullptr;
    info.stage.flags = options.full ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT : 0u;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = program.module;
    info.stage.pName = "main";
    info.stage.pSpecializationInfo = &specialization;
    info.layout = program.layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto start = std::chrono::steady_clock::now();
    check(vkCreateComputePipelines(gpu.device, cache, 1, &info, nullptr, &pipeline), "vkCreateComputePipelines");
    const auto wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const bool valid = (pipelineFeedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT_EXT) != 0;
    const bool hit = (pipelineFeedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT_EXT) != 0;
    std::printf("%s: wall %.0f ms, feedback %s %s, driver %.0f ms (stage %s, %.0f ms)\n", label, wallMs, valid ? "valid" : "invalid", hit ? "cache-hit" : "cache-miss", pipelineFeedback.duration / 1e6,
            (stageFeedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT_EXT) != 0 ? "cache-hit" : "cache-miss", stageFeedback.duration / 1e6);
    return pipeline;
}

std::vector<char> cacheData(const Gpu& gpu, VkPipelineCache cache) {
    std::size_t size = 0;
    check(vkGetPipelineCacheData(gpu.device, cache, &size, nullptr), "vkGetPipelineCacheData size");
    std::vector<char> data(size);
    const auto result = vkGetPipelineCacheData(gpu.device, cache, &size, data.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) check(result, "vkGetPipelineCacheData");
    data.resize(size);
    return data;
}

VkPipelineCache createCache(const Gpu& gpu, const std::vector<char>* initial, const char* label) {
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (initial != nullptr) {
        info.initialDataSize = initial->size();
        info.pInitialData = initial->data();
    }
    VkPipelineCache cache = VK_NULL_HANDLE;
    const auto result = vkCreatePipelineCache(gpu.device, &info, nullptr, &cache);
    const auto afterwards = result == VK_SUCCESS ? cacheData(gpu, cache).size() : 0u;
    std::printf("cache %s: vkCreatePipelineCache %d, initial %zu bytes, data right after creation %zu bytes\n", label, static_cast<int>(result), initial != nullptr ? initial->size() : 0u, afterwards);
    check(result, "vkCreatePipelineCache");
    return cache;
}

void prefill(const Gpu& gpu, const Program& program) {
    if (options.prefill == 0) return;
    const auto cache = createCache(gpu, nullptr, "prefill");
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t index = 0; index < options.prefill; ++index) {
        std::vector<std::uint32_t> constantValues;
        std::vector<VkSpecializationMapEntry> entries;
        for (std::uint32_t at = 0; at < options.spec; ++at) {
            constantValues.push_back(7u + at + (5000u + index) * 131u);
            entries.push_back({at, at * 4u, 4u});
        }
        VkSpecializationInfo specialization{static_cast<std::uint32_t>(entries.size()), entries.data(), constantValues.size() * 4u, constantValues.data()};
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = program.module;
        info.stage.pName = "main";
        info.stage.pSpecializationInfo = &specialization;
        info.layout = program.layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        check(vkCreateComputePipelines(gpu.device, cache, 1, &info, nullptr, &pipeline), "vkCreateComputePipelines prefill");
        vkDestroyPipeline(gpu.device, pipeline, nullptr);
    }
    const auto data = cacheData(gpu, cache);
    std::printf("prefill: %u cold variants compiled in %.0f ms; their cache holds %zu bytes\n", options.prefill, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), data.size());
}

int exportMode(int argc, char** argv) {
    if (argc != 4) fail("export <spv> <out-i> <out-ii>");
    const auto gpu = open();
    const auto slow = program(gpu, argv[1]);
    const auto fresh = createCache(gpu, nullptr, "fresh");
    const auto first = create(gpu, slow, fresh, "first creation into the fresh cache");
    const auto dataI = cacheData(gpu, fresh);
    writeFile(argv[2], dataI);
    std::printf("(i) fresh cache data after the compile: %zu bytes -> %s\n", dataI.size(), argv[2]);
    const auto empty = createCache(gpu, nullptr, "empty");
    const auto repeat = create(gpu, slow, empty, "repeat creation into the empty cache");
    const auto dataII = cacheData(gpu, empty);
    writeFile(argv[3], dataII);
    std::printf("(ii) empty cache data after the repeat: %zu bytes -> %s\n", dataII.size(), argv[3]);
    const auto again = create(gpu, slow, VK_NULL_HANDLE, "third creation with no cache (in-process state)");
    vkDestroyPipeline(gpu.device, again, nullptr);
    vkDestroyPipeline(gpu.device, repeat, nullptr);
    vkDestroyPipeline(gpu.device, first, nullptr);
    return 0;
}

int loadMode(int argc, char** argv) {
    if (argc != 3 && argc != 4) fail("load <spv> <cache-file|none> [warm.spv]");
    const auto gpu = open();
    if (argc == 4) {
        const auto warm = program(gpu, argv[3]);
        const auto warmCache = createCache(gpu, nullptr, "warm-up");
        const auto pipeline = create(gpu, warm, warmCache, "warm-up creation of the unrelated pipeline");
        vkDestroyPipeline(gpu.device, pipeline, nullptr);
    }
    const auto slow = program(gpu, argv[1]);
    if (options.prefill != 0) {
        if (argc == 4) prefill(gpu, program(gpu, argv[3]));
        else prefill(gpu, slow);
    }
    std::vector<char> preData;
    if (options.pre != nullptr) {
        preData = readFile(options.pre);
        createCache(gpu, &preData, "pre-import");
    }
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::vector<char> initial;
    if (std::strcmp(argv[2], "none") != 0) {
        initial = readFile(argv[2]);
        cache = createCache(gpu, &initial, argv[2]);
    }
    std::vector<char> postData;
    if (options.post != nullptr) {
        postData = readFile(options.post);
        createCache(gpu, &postData, "post-import");
    }
    std::vector<std::thread> workers;
    const auto busySeed = options.seed;
    for (std::uint32_t worker = 0; worker < options.busy; ++worker) {
        workers.emplace_back([&gpu, &slow, worker, busySeed] {
            for (std::uint32_t round = 0; round < 3; ++round) {
                std::vector<std::uint32_t> constantValues;
                std::vector<VkSpecializationMapEntry> entries;
                for (std::uint32_t at = 0; at < options.spec; ++at) {
                    constantValues.push_back(7u + at + (busySeed + 3000u + worker * 10u + round) * 131u);
                    entries.push_back({at, at * 4u, 4u});
                }
                VkSpecializationInfo specialization{static_cast<std::uint32_t>(entries.size()), entries.data(), constantValues.size() * 4u, constantValues.data()};
                VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                info.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
                info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                info.stage.module = slow.module;
                info.stage.pName = "main";
                info.stage.pSpecializationInfo = &specialization;
                info.layout = slow.layout;
                VkPipeline pipeline = VK_NULL_HANDLE;
                check(vkCreateComputePipelines(gpu.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline), "vkCreateComputePipelines busy");
                vkDestroyPipeline(gpu.device, pipeline, nullptr);
            }
        });
    }
    if (options.busy != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        std::printf("busy: %u threads compiling cold variants during the creation\n", options.busy);
    }
    const auto pipeline = create(gpu, slow, cache, cache != VK_NULL_HANDLE ? "creation with the loaded cache" : "creation with no cache");
    for (auto& worker : workers) worker.join();
    if (cache != VK_NULL_HANDLE) std::printf("loaded cache data after the creation: %zu bytes\n", cacheData(gpu, cache).size());
    vkDestroyPipeline(gpu.device, pipeline, nullptr);
    return 0;
}

int fillMode(int argc, char** argv) {
    if (argc != 4) fail("fill <spv> <count> <out.bin>");
    const auto gpu = open();
    const auto slow = program(gpu, argv[1]);
    const auto count = static_cast<std::uint32_t>(std::atoi(argv[2]));
    const auto cache = createCache(gpu, nullptr, "fill");
    const auto start = std::chrono::steady_clock::now();
    std::vector<VkPipeline> pipelines;
    for (std::uint32_t index = 0; index < count; ++index) {
        options.seed = index;
        char label[64];
        std::snprintf(label, sizeof(label), "fill variant %u", index);
        if (index < 2 || index + 1 == count) pipelines.push_back(create(gpu, slow, cache, label));
        else {
            std::vector<std::uint32_t> constantValues;
            std::vector<VkSpecializationMapEntry> entries;
            for (std::uint32_t at = 0; at < options.spec; ++at) {
                constantValues.push_back(7u + at + options.seed * 131u);
                entries.push_back({at, at * 4u, 4u});
            }
            VkSpecializationInfo specialization{static_cast<std::uint32_t>(entries.size()), entries.data(), constantValues.size() * 4u, constantValues.data()};
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
            info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            info.stage.module = slow.module;
            info.stage.pName = "main";
            info.stage.pSpecializationInfo = &specialization;
            info.layout = slow.layout;
            VkPipeline pipeline = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(gpu.device, cache, 1, &info, nullptr, &pipeline), "vkCreateComputePipelines fill");
            pipelines.push_back(pipeline);
        }
    }
    const auto data = cacheData(gpu, cache);
    writeFile(argv[3], data);
    std::printf("fill: %u variants in %.0f ms, cache data %zu bytes -> %s\n", count, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), data.size(), argv[3]);
    for (const auto pipeline : pipelines) vkDestroyPipeline(gpu.device, pipeline, nullptr);
    return 0;
}

int probeMode(int argc, char** argv) {
    if (argc != 4) fail("probe <spv> <cache.bin> <seed>");
    const auto gpu = open();
    const auto slow = program(gpu, argv[1]);
    prefill(gpu, slow);
    const auto initial = readFile(argv[2]);
    const auto cache = createCache(gpu, &initial, argv[2]);
    options.seed = static_cast<std::uint32_t>(std::atoi(argv[3]));
    char label[64];
    std::snprintf(label, sizeof(label), "creation of variant %u from the imported cache", options.seed);
    const auto pipeline = create(gpu, slow, cache, label);
    vkDestroyPipeline(gpu.device, pipeline, nullptr);
    return 0;
}

}

int main(int argc, char** argv) {
    int at = 1;
    while (at < argc && std::strncmp(argv[at], "--", 2) == 0) {
        const std::string option = argv[at++];
        if (option == "--subgroup32") options.subgroup32 = true;
        else if (option == "--full") options.full = options.subgroup32 = true;
        else if (option == "--uab") options.uab = true;
        else if (option == "--push") options.push = true;
        else if (option == "--sets2") options.sets2 = true;
        else if (option == "--spec" && at < argc) options.spec = static_cast<std::uint32_t>(std::atoi(argv[at++]));
        else if (option == "--seed" && at < argc) options.seed = static_cast<std::uint32_t>(std::atoi(argv[at++]));
        else if (option == "--pre" && at < argc) options.pre = argv[at++];
        else if (option == "--post" && at < argc) options.post = argv[at++];
        else if (option == "--busy" && at < argc) options.busy = static_cast<std::uint32_t>(std::atoi(argv[at++]));
        else if (option == "--prefill" && at < argc) options.prefill = static_cast<std::uint32_t>(std::atoi(argv[at++]));
        else if (option == "--device" && at < argc) options.device = std::atoi(argv[at++]);
        else if (option == "--any-vendor") options.anyVendor = true;
        else fail("unknown option");
    }
    if (at >= argc) fail("export|load|fill|probe ...");
    if (std::strcmp(argv[at], "export") == 0) return exportMode(argc - at, argv + at);
    if (std::strcmp(argv[at], "load") == 0) return loadMode(argc - at, argv + at);
    if (std::strcmp(argv[at], "fill") == 0) return fillMode(argc - at, argv + at);
    if (std::strcmp(argv[at], "probe") == 0) return probeMode(argc - at, argv + at);
    fail("export|load|fill|probe ...");
}
