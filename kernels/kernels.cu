#include "kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

template<typename T = std::uint8_t, typename RES = std::uint32_t>
__global__ void naive(const T* data, RES N, RES* result, T fromValue, T toValue)
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < (toValue - fromValue))
	{
		T threadValue = fromValue + idx;
		for (RES i = 0; i < N; ++i) {
			// Watch for signed/unsigned comparison
			if (data[i] == threadValue) {
				result[idx]++;
			}
		}
	}

}

template<typename T = std::uint8_t, typename RES = std::uint32_t>
__global__ void atomic(const T* data, RES N, RES* result, T fromValue, T toValue)
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		T value = data[idx];
		if (value >= fromValue && value < toValue) {
			atomicAdd(result + (value - fromValue), 1);
		}
	}
}



template<typename T = std::uint8_t, typename RES = std::uint32_t>
__global__ void privatized(const T* data, RES N, RES* result, T fromValue, T toValue, const int copiesPerBlock)
{
	extern __shared__ RES hist[];

	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int histValues = toValue - fromValue + 1;
	const int histSize = histValues % 32 != 0 && 32 % histValues != 0 ? histValues : histValues + 1;

	// Clear the shared memory
	for (int i = threadIdx.x; i < histSize*copiesPerBlock; i += blockDim.x) {
		hist[i] = 0;
	}

	__syncthreads();

	RES *histPart = hist + (histSize) * (threadIdx.x % copiesPerBlock);
	if (idx < N) {
		T value = data[idx];
		if (value >= fromValue && value <= toValue) {
			atomicAdd(histPart + (value - fromValue), 1);
		}
	}

	__syncthreads();
	// Aggregate the resutls into the first histogram copy
	for (int i = threadIdx.x; i < histValues; i += blockDim.x) {
		for (int copy = 1; copy < copiesPerBlock; ++copy) {
			hist[i] += hist[copy*histSize + i];
		}

		atomicAdd(result + i, hist[i]);
	}
}

template<typename T = std::uint8_t, typename RES = std::uint32_t>
__global__ void aggregated(const T* data, RES N, RES* result, T fromValue, T toValue, const int itemsPerThread)
{
	auto start = threadIdx.x + blockIdx.x * blockDim.x * itemsPerThread;
	auto end = min(threadIdx.x + (blockIdx.x + 1) * blockDim.x * itemsPerThread, N);
	for (int i = start; i < end; i += blockDim.x) {
		T value = data[i];
		if (value >= fromValue && value <= toValue) {
			atomicAdd(result + (value - fromValue), 1);
		}
	}
}

template<typename T = std::uint8_t, typename RES = std::uint32_t>
__global__ void atomic_shm(const T* data, RES N, RES* result, T fromValue, T toValue, const int copiesPerBlock, const int itemsPerThread)
{
	extern __shared__ RES hist[];

	auto start = threadIdx.x + blockIdx.x * blockDim.x * itemsPerThread;
	auto end = min(threadIdx.x + (blockIdx.x + 1) * blockDim.x * itemsPerThread, N);

	const int histValues = toValue - fromValue + 1;
	const int histSize = histValues % 32 != 0 && 32 % histValues != 0 ? histValues : histValues + 1;

	// Clear the shared memory
	for (int i = threadIdx.x; i < histSize*copiesPerBlock; i += blockDim.x) {
		hist[i] = 0;
	}

	__syncthreads();

	RES *histPart = hist + (histSize) * (threadIdx.x % copiesPerBlock);
	for (int i = start; i < end; i += blockDim.x) {
		T value = data[i];
		if (value >= fromValue && value <= toValue) {
			atomicAdd(histPart + (value - fromValue), 1);
		}
	}

	__syncthreads();

	// Aggregate the resutls into the first histogram copy
	for (int i = threadIdx.x; i < histValues; i += blockDim.x) {
		for (int copy = 1; copy < copiesPerBlock; ++copy) {
			hist[i] += hist[copy*histSize + i];
		}

		atomicAdd(result + i, hist[i]);
	}
}

template<typename T, typename RES>
void run_naive(const T* data, std::size_t N, RES* result, T fromValue, T toValue, int blockSize){
	const T numBins = toValue - fromValue;
	const std::size_t numBlocks = (numBins / blockSize) + (numBins % blockSize == 0 ? 0 : 1);
	naive<T,RES><<<numBlocks, blockSize>>>(data, (RES)N, result, fromValue, toValue);
}

template<typename T, typename RES>
void run_atomic(const T* data, std::size_t N, RES* result, T fromValue, T toValue, int blockSize) {
	const std::size_t numBlocks = (N / blockSize) + (N % blockSize == 0 ? 0 : 1);
	atomic<T,RES><<<numBlocks, blockSize>>>(data, (RES)N, result, fromValue, toValue);
}

template<typename T, typename RES>
void run_privatized(const T* data, std::size_t N, RES* result, T fromValue, T toValue, int blockSize, int copiesPerBlock){
	const std::size_t numBlocks = (N / blockSize) + (N % blockSize == 0 ? 0 : 1);
	const std::size_t histValues = toValue - fromValue + 1;
	const std::size_t histSize = (histValues % 32 != 0 && 32 % histValues != 0 ? histValues : histValues + 1) * sizeof(RES);
	privatized<T,RES><<<numBlocks, blockSize, histSize * copiesPerBlock>>>(data, (RES)N, result, fromValue, toValue, copiesPerBlock);
}

/*
* By itself, this type of optimalization is just slightly faster than the atomic, as the bottleneck in the atomic solution
* is access to global memory
*/
template<typename T, typename RES>
void run_aggregated(const T* data, std::size_t N, RES* result, T fromValue, T toValue, int blockSize, int itemsPerThread) {
	const int itemsPerBlock = blockSize * itemsPerThread;
	const std::size_t numBlocks = (N / itemsPerBlock) + (N % itemsPerBlock == 0 ? 0 : 1);
	aggregated<T,RES><<<numBlocks, blockSize>>>(data, (RES)N, result, fromValue, toValue, itemsPerThread);
}

template<typename T, typename RES>
void run_atomic_shm(const T* data, std::size_t N, RES* result, T fromValue, T toValue, int blockSize, int copiesPerBlock, int itemsPerThread, cudaStream_t stream) {
	const int itemsPerBlock = blockSize * itemsPerThread;
	const std::size_t numBlocks = (N / itemsPerBlock) + (N % itemsPerBlock == 0 ? 0 : 1);
	const std::size_t histValues = toValue - fromValue + 1;
	const std::size_t histSize = (histValues % 32 != 0 && 32 % histValues != 0 ? histValues : histValues + 1) * sizeof(RES);
	atomic_shm<T,RES><<<numBlocks, blockSize, histSize * copiesPerBlock, stream>>>(data, (RES)N, result, fromValue, toValue, copiesPerBlock, itemsPerThread);
}

template void run_naive<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue, int blockSize);
template void run_atomic<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue, int blockSize);
template void run_privatized<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue, int blockSize, int copiesPerBlock);
template void run_aggregated<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue, int blockSize, int itemsPerThread);
template void run_atomic_shm<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue, int blockSize, int copiesPerBlock, int itemsPerThread, cudaStream_t stream);