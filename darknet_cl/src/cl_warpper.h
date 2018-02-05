#ifndef CL_WARPPER_H
#define CL_WARPPER_H
#ifdef GPU
//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "darknet.h"
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <map>
#include <memory>


class CLProgram {
public:
	CLProgram(cl_program);
	cl_kernel getKernel(std::string kernelname);
	~CLProgram();
private:
	std::map<std::string, cl_kernel> kernels;
	cl_program program;
};

class CLWarpper {
public:
	bool verbose;

	cl_int error;  

	cl_platform_id platform_id;
	cl_device_id device;

	cl_context *context;
	cl_command_queue *queue;

	template<typename T>
	static std::string toString(T val) {
		std::ostringstream myostringstream;
		myostringstream << val;
		return myostringstream.str();
	}

	void commonConstructor(cl_platform_id platform_id, cl_device_id device);
	CLWarpper(int gpu);
	CLWarpper();
	CLWarpper(cl_platform_id platformId, cl_device_id deviceId);

	virtual ~CLWarpper();

	static int roundUp(int quantization, int minimum);

	static int getPower2Upperbound(int value);// eg pass in 320, it will return: 512
	//I would like to choose gpu,so ignore other device
	static std::shared_ptr<CLWarpper> createForFirstGpu();
	static std::shared_ptr<CLWarpper> createForIndexedGpu(int gpu);
	
	static std::shared_ptr<CLWarpper> createForPlatformDeviceIndexes(int platformIndex, int deviceIndex);
	static std::shared_ptr<CLWarpper> createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId);

	static std::string errorMessage(cl_int error);
	static void checkError(cl_int error);

	void gpu(int gpuIndex);
	void init(int gpuIndex);
	void finish();

	int getComputeUnits();
	int getLocalMemorySize();
	int getLocalMemorySizeKB();
	int getMaxWorkgroupSize();
	int getMaxAllocSizeMB();

	std::shared_ptr<CLProgram> buildProgramFromFile(std::string sourcefileName, std::string options);

private:
	static int instance_count;
	static std::string getFileContents(std::string filename);
	int64_t getDeviceInfoInt64(cl_device_info name);
};






void printPlatformInfoString(std::string valuename, cl_platform_id platformId, cl_platform_info name);
void printPlatformInfo(std::string valuename, cl_platform_id platformId, cl_platform_info name);
std::string getPlatformInfoString(cl_platform_id platformId, cl_platform_info name);

#endif
#endif