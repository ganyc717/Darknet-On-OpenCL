#ifdef GPU
#include"cl_warpper.h"
#include"cl_kernel_source.h"
static int a = 0;


CLWarpper::CLWarpper(int gpu, bool verbose) {
	init(gpu, verbose);
}
CLWarpper::CLWarpper(int gpu) {
	init(gpu, true);
}
CLWarpper::CLWarpper(bool verbose) {
	init(0, verbose);
}
CLWarpper::CLWarpper() {
	init(0, true);
}
CLWarpper::CLWarpper(cl_platform_id platform_id, cl_device_id device, bool verbose) {
	commonConstructor(platform_id, device, verbose);
}
CLWarpper::CLWarpper(cl_platform_id platform_id, cl_device_id device) {
	commonConstructor(platform_id, device, true);
}
CLWarpper::~CLWarpper() {
	if (queue != 0) {
		clReleaseCommandQueue(*queue);
		delete queue;
	}
	if (context != 0) {
		clReleaseContext(*context);
		delete context;
	}
}
void CLWarpper::init(int gpuIndex, bool verbose) {

	error = 0;
	queue = 0;
	context = 0;

	cl_uint num_platforms;
	error = clGetPlatformIDs(1, &platform_id, &num_platforms);

	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL platforms ids: " + errorMessage(error));
	}
	if (num_platforms == 0) {
		throw std::runtime_error("Error: no OpenCL platforms available");
	}

	cl_uint num_devices;
	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, 0, &num_devices);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL device ids: " + errorMessage(error));
	}

	cl_device_id *device_ids = new cl_device_id[num_devices];
	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, num_devices, device_ids, &num_devices);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL device ids: " + errorMessage(error));
	}

	if (gpuIndex >= static_cast<int>(num_devices)) {
		throw std::runtime_error("requested gpuindex " + toString(gpuIndex) + " goes beyond number of available device " + toString(num_devices));
	}
	device = device_ids[gpuIndex];
	delete[] device_ids;

	// Context
	context = new cl_context();
	*context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error creating OpenCL context, OpenCL errorcode: " + errorMessage(error));
	}
	// Command-queue
	queue = new cl_command_queue;
	*queue = clCreateCommandQueue(*context, device, 0, &error);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error creating OpenCL command queue, OpenCL errorcode: " + errorMessage(error));
	}
}
void CLWarpper::commonConstructor(cl_platform_id platform_id, cl_device_id device, bool verbose)
{
	queue = 0;
	context = 0;

	this->platform_id = platform_id;
	this->device = device;

	// Context
	context = new cl_context();
	*context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error creating OpenCL context, OpenCL errocode: " + errorMessage(error));
	}
	// Command-queue
	queue = new cl_command_queue;
	*queue = clCreateCommandQueue(*context, device, 0, &error);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error creating OpenCL command queue, OpenCL errorcode: " + errorMessage(error));
	}
}
int CLWarpper::roundUp(int quantization, int minimum) {
	return ((minimum + quantization - 1) / quantization * quantization);
}
int CLWarpper::getPower2Upperbound(int value) {
	int upperbound = 1;
	while (upperbound < value) {
		upperbound <<= 1;
	}
	return upperbound;
}

std::shared_ptr<CLWarpper> CLWarpper::createForIndexedGpu(int gpu) {
	cl_int error;
	int currentGpuIndex = 0;
	cl_platform_id platform_ids[10];
	cl_uint num_platforms;
	error = clGetPlatformIDs(10, platform_ids, &num_platforms);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL platforms ids, OpenCL errorcode: " + errorMessage(error));
	}
	if (num_platforms == 0) {
		throw std::runtime_error("Error: no OpenCL platforms available");
	}
	for (int platform = 0; platform < (int)num_platforms; platform++) {
		cl_platform_id platform_id = platform_ids[platform];

		cl_device_id device_ids[100];
		cl_uint num_devices;
		error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 100, device_ids, &num_devices);
		if (error != CL_SUCCESS) {
			continue;
		}

		if ((gpu - currentGpuIndex) < (int)num_devices) {
			return std::shared_ptr<CLWarpper>(new CLWarpper(platform_id, device_ids[(gpu - currentGpuIndex)], true));
		}
		else {
			currentGpuIndex += num_devices;
		}
	}
	if (gpu == 0) {
		throw std::runtime_error("No OpenCL-enabled GPUs found");
	}
	else {
		throw std::runtime_error("Not enough OpenCL-enabled GPUs found to satisfy gpu index: " + toString(gpu));
	}
	if (a == 0)
	{
		clblasSetup();
	}
}
std::shared_ptr<CLWarpper> CLWarpper::createForFirstGpu() {
	return createForIndexedGpu(0);
}
std::shared_ptr<CLWarpper> CLWarpper::createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId) {
	return std::shared_ptr<CLWarpper>(new CLWarpper(platformId, deviceId));
}
std::shared_ptr<CLWarpper> CLWarpper::createForPlatformDeviceIndexes(int platformIndex, int deviceIndex) {
	cl_int error;
	cl_platform_id platform_ids[10];
	cl_uint num_platforms;
	error = clGetPlatformIDs(10, platform_ids, &num_platforms);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL platforms ids, OpenCL errorcode: " + errorMessage(error));
	}
	if (num_platforms == 0) {
		throw std::runtime_error("Error: no OpenCL platforms available");
	}
	if (platformIndex >= (int)num_platforms) {
		throw std::runtime_error("Error: OpenCL platform index " + toString(platformIndex) + " not available. There are only: " + toString(num_platforms) + " platforms available");
	}
	cl_platform_id platform_id = platform_ids[platformIndex];
	cl_device_id device_ids[100];
	cl_uint num_devices;
	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 100, device_ids, &num_devices);
	if (error != CL_SUCCESS) {
		throw std::runtime_error("Error getting OpenCL device ids for platform index " + toString(platformIndex) + ": OpenCL errorcode: " + errorMessage(error));
	}
	if (num_devices == 0) {
		throw std::runtime_error("Error: no OpenCL devices available for platform index " + toString(platformIndex));
	}
	if (deviceIndex >= (int)num_devices) {
		throw std::runtime_error("Error: OpenCL device index " + toString(deviceIndex) + " goes beyond the available devices on platform index " + toString(platformIndex) + ", which has " + toString(num_devices) + " devices");
	}
	return std::shared_ptr<CLWarpper>(new CLWarpper(platform_id, device_ids[deviceIndex]));
}

std::string CLWarpper::errorMessage(cl_int error) {
	return toString(error);
}
void CLWarpper::checkError(cl_int error) {
	if (error != CL_SUCCESS) {
		std::string message = toString(error);
		switch (error) {
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			message = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			break;
		case CL_INVALID_ARG_SIZE:
			message = "CL_INVALID_ARG_SIZE";
			break;
		case CL_INVALID_BUFFER_SIZE:
			message = "CL_INVALID_BUFFER_SIZE";
			break;
		}
		std::cout << "opencl execution error, code " << error << " " << message << std::endl;
		throw std::runtime_error(std::string("OpenCL error, code: ") + message);
	}
}

std::string CLWarpper::getFileContents(std::string filename) {
	std::ifstream t(filename.c_str());
	std::stringstream buffer;
	buffer << t.rdbuf();
	return buffer.str();
}
void CLWarpper::gpu(int gpuIndex) {
	finish();
	if (queue != 0) {
		clReleaseCommandQueue(*queue);
		delete queue;
	}
	if (context != 0) {
		clReleaseContext(*context);
		delete context;
	}

	init(gpuIndex, this->verbose);
}
void CLWarpper::finish() {
	error = clFinish(*queue);
	switch (error) {
	case CL_SUCCESS:
		break;
	case -36:
		throw std::runtime_error("Invalid command queue: often indicates out of bounds memory access within kernel");
	default:
		checkError(error);
	}
}

int CLWarpper::getComputeUnits() {
	return (int)this->getDeviceInfoInt64(CL_DEVICE_MAX_COMPUTE_UNITS);
}
int CLWarpper::getLocalMemorySize() {
	return (int)this->getDeviceInfoInt64(CL_DEVICE_LOCAL_MEM_SIZE);
}
int CLWarpper::getLocalMemorySizeKB() {
	return (int)(this->getDeviceInfoInt64(CL_DEVICE_LOCAL_MEM_SIZE) / 1024);
}
int CLWarpper::getMaxWorkgroupSize() {
	return (int)this->getDeviceInfoInt64(CL_DEVICE_MAX_WORK_GROUP_SIZE);
}
int CLWarpper::getMaxAllocSizeMB() {
	return (int)(this->getDeviceInfoInt64(CL_DEVICE_MAX_MEM_ALLOC_SIZE) / 1024 / 1024);
}

std::shared_ptr<CLProgram> CLWarpper::buildProgramFromFile(std::string sourcefileName, std::string options)
{
	size_t src_size = 0;
	const char *source_char;
	std::string source = getFileContents(sourcefileName);
	if (source.empty())//use the default buildin kernel source
		if(source_map[sourcefileName].empty())
			throw std::runtime_error("Failed to find the kernel source");
		else
			source = source_map[sourcefileName];

	source_char = source.c_str();
	src_size = strlen(source_char);

	cl_program program = clCreateProgramWithSource(*context, 1, &source_char, &src_size, &error);
	checkError(error);

	error = clBuildProgram(program, 1, &device, options.c_str(), NULL, NULL);
	checkError(error);

	char* build_log;
	size_t log_size;
	error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	checkError(error);
	build_log = new char[log_size + 1];
	error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	checkError(error);
	build_log[log_size] = '\0';
	std::string buildLogMessage = "";
	if (log_size > 2) {
		buildLogMessage = sourcefileName + " build log: " + "\n" + build_log;
		std::cout << buildLogMessage << std::endl;
	}
	delete[] build_log;
	checkError(error);
	std::shared_ptr<CLProgram> res(new CLProgram(program));
	return res;
}

int64_t CLWarpper::getDeviceInfoInt64(cl_device_info name) {
	cl_ulong value = 0;
	clGetDeviceInfo(device, name, sizeof(cl_ulong), &value, 0);
	return static_cast<int64_t>(value);
}


void printPlatformInfoString(std::string valuename, cl_platform_id platformId, cl_platform_info name)
{
	char buffer[256];
	buffer[0] = 0;
	clGetPlatformInfo(platformId, name, 256, buffer, 0);
	std::cout << valuename << ": " << buffer << std::endl;
}
std::string getPlatformInfoString(cl_platform_id platformId, cl_platform_info name) {
	char buffer[257];
	buffer[0] = 0;
	size_t namesize;
	cl_int error = clGetPlatformInfo(platformId, name, 256, buffer, &namesize);
	if (error != CL_SUCCESS) {
		if (error == CL_INVALID_PLATFORM) {
			throw std::runtime_error("Failed to obtain platform info for platform id " + CLWarpper::toString(platformId) + ": invalid platform");
		}
		else if (error == CL_INVALID_VALUE) {
			throw std::runtime_error("Failed to obtain platform info " + CLWarpper::toString(name) + " for platform id " + CLWarpper::toString(platformId) + ": invalid value");
		}
		else {
			throw std::runtime_error("Failed to obtain platform info " + CLWarpper::toString(name) + " for platform id " + CLWarpper::toString(platformId) + ": unknown error code: " + CLWarpper::toString(error));
		}
	}
	return std::string(buffer);
}
void printPlatformInfo(std::string valuename, cl_platform_id platformId, cl_platform_info name) {
	cl_ulong somelong = 0;
	clGetPlatformInfo(platformId, name, sizeof(cl_ulong), &somelong, 0);
	std::cout << valuename << ": " << somelong << std::endl;
}

CLProgram::CLProgram(cl_program prog) :program(prog) {}
cl_kernel CLProgram::getKernel(std::string kernelname)
{
	auto iter = kernels.find(kernelname);
	if (iter == kernels.end())
	{
		cl_int error;
		cl_kernel kernel = clCreateKernel(program, kernelname.c_str(), &error);
		if (error != CL_SUCCESS) {
			std::string exceptionMessage = "";
			switch (error) {
			case -46:
				exceptionMessage = "Invalid kernel name, code -46, kernel " + kernelname + "\n";
				break;
			default:
				exceptionMessage = "Something went wrong with clCreateKernel, OpenCL error code " + CLWarpper::toString(error) + "\n";
				break;
			}

			std::cout << "kernel build error:\n" << exceptionMessage << std::endl;
			throw std::runtime_error(exceptionMessage);
		}
		kernels[kernelname] = kernel;
		return kernel;
	}
	else
		return iter->second;
}
CLProgram::~CLProgram()
{
	for (auto iter = kernels.begin(); iter != kernels.end(); iter++)
		clReleaseKernel(iter->second);
	clReleaseProgram(program);
}

#endif