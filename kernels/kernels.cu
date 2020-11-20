#include "kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>


template<typename T = std::uint8_t, typename RES = std::uint32_t>
__global__ void hist(const T* data, RES N, RES* result, T fromValue, T toValue)
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;

	// TODO
}

template<typename T, typename RES>
void run_hist(const T* data, std::size_t N, RES* result, T fromValue, T toValue)
{
	constexpr unsigned int blockSize = 256;
	hist<T, RES><<<1, blockSize >>>(data, (RES)N, result, fromValue, toValue);
}

template void run_hist<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue);
