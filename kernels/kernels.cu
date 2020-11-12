#include "kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

template<typename T = std::uint8_t, typename RES = std::uint32_t>
__global__ void bin_parallel(const T* data, RES N, RES* result, T fromValue, T toValue)
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
__global__ void atomics(const T* data, RES N, RES* result, T fromValue, T toValue)
{
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N) {
		T value = data[idx];
		if (value >= fromValue && value < toValue) {
			atomicAdd(result + (value - fromValue), 1);
		}
	}
}

extern __shared__ int shared[];

template<typename T = std::uint8_t, typename RES = std::uint32_t>
__global__ void privatized(const T* data, RES N, RES* result, T fromValue, T toValue, const int copiesPerBlock)
{
	RES *hist = reinterpret_cast<RES*>(shared);
	auto idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int histSize = toValue - fromValue + 1;

	// Clear the shared memory
	for (int i = threadIdx.x; i < histSize*copiesPerBlock; i += blockDim.x) {
		hist[i] = 0;
	}

	__syncthreads();

	RES *histPart = hist + (histSize) * ((threadIdx.x * copiesPerBlock) / blockDim.x);
	if (idx < N) {
		T value = data[idx];
		if (value >= fromValue && value <= toValue) {
			atomicAdd(histPart + (value - fromValue), 1);
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < histSize*copiesPerBlock; i += blockDim.x) {
		atomicAdd(result + (i % histSize), hist[i]);
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
__global__ void privatized_aggregated(const T* data, RES N, RES* result, T fromValue, T toValue, const int copiesPerBlock, const int itemsPerThread)
{
	RES *hist = reinterpret_cast<RES*>(shared);
	auto start = threadIdx.x + blockIdx.x * blockDim.x * itemsPerThread;
	auto end = min(threadIdx.x + (blockIdx.x + 1) * blockDim.x * itemsPerThread, N);

	const int histSize = toValue - fromValue + 1;

	// Clear the shared memory
	for (int i = threadIdx.x; i < histSize*copiesPerBlock; i += blockDim.x) {
		hist[i] = 0;
	}

	__syncthreads();

	RES *histPart = hist + (histSize) * ((threadIdx.x * copiesPerBlock) / blockDim.x);
	for (int i = start; i < end; i += blockDim.x) {
		T value = data[i];
		if (value >= fromValue && value <= toValue) {
			atomicAdd(histPart + (value - fromValue), 1);
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < histSize*copiesPerBlock; i += blockDim.x) {
		atomicAdd(result + (i % histSize), hist[i]);
	}
}

template<typename T, typename RES>
void run_bin_parallel(const T* data, std::size_t N, RES* result, T fromValue, T toValue){
	constexpr unsigned int blockSize = 256;

	const T numBins = toValue - fromValue;
	const std::size_t numBlocks = (numBins / blockSize) + (numBins % blockSize == 0 ? 0 : 1);
	bin_parallel<T,RES><<<numBlocks, blockSize>>>(data, (RES)N, result, fromValue, toValue);
}

template<typename T, typename RES>
void run_atomics(const T* data, std::size_t N, RES* result, T fromValue, T toValue) {
	constexpr unsigned int blockSize = 256;
	const std::size_t numBlocks = (N / blockSize) + (N % blockSize == 0 ? 0 : 1);
	atomics<T,RES><<<numBlocks, blockSize>>>(data, (RES)N, result, fromValue, toValue);
}

/*
* Nema cenu delat vice privatnich kopii, protoze do ty shared pameti pristupujou ty vlakna dost na random
* takze ke kolizim nedojde tak casto a jenom tim pridavame pak kolize pri pristupu do globalni pameti
*/
template<typename T, typename RES>
void run_privatized(const T* data, std::size_t N, RES* result, T fromValue, T toValue, int blockSize, int copiesPerBlock){
	const std::size_t numBlocks = (N / blockSize) + (N % blockSize == 0 ? 0 : 1);
	const std::size_t histSize = (toValue - fromValue + 1) * sizeof(RES);
	privatized<T,RES><<<numBlocks, blockSize, histSize * copiesPerBlock>>>(data, (RES)N, result, fromValue, toValue, copiesPerBlock);
}

/*
* Tohle je samo o sobe moc pomaly, protoze tam je bottleneck pristup do globalni pameti
* takze jen o trosku rychlejsi nez atomic
*/
template<typename T, typename RES>
void run_aggregated(const T* data, std::size_t N, RES* result, T fromValue, T toValue, int blockSize, int itemsPerThread) {
	const int itemsPerBlock = blockSize * itemsPerThread;
	const std::size_t numBlocks = (N / itemsPerBlock) + (N % itemsPerBlock == 0 ? 0 : 1);
	aggregated<T,RES><<<numBlocks, blockSize>>>(data, (RES)N, result, fromValue, toValue, itemsPerThread);
}

/*
* Samozrejme nejrychlejsi, optimum je 128 itemsPerThread, stejne jako v privatized nema cenu delat vice kopii, jenom
* tim pridavame praci a kolize pri pristupu do globalni pameti, protoze ten se dela dost sekvencne
* oproti tomu pristup do shared se dela random
*/
template<typename T, typename RES>
void run_privatized_aggregated(const T* data, std::size_t N, RES* result, T fromValue, T toValue, int blockSize, int copiesPerBlock, int itemsPerThread) {
	const int itemsPerBlock = blockSize * itemsPerThread;
	const std::size_t numBlocks = (N / itemsPerBlock) + (N % itemsPerBlock == 0 ? 0 : 1);
	const std::size_t histSize = (toValue - fromValue + 1) * sizeof(RES);
	privatized_aggregated<T,RES><<<numBlocks, blockSize, histSize * copiesPerBlock>>>(data, (RES)N, result, fromValue, toValue, copiesPerBlock, itemsPerThread);
}

template void run_bin_parallel<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue);
template void run_atomics<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue);
template void run_privatized<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue, int blockSize, int copiesPerBlock);
template void run_aggregated<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue, int blockSize, int itemsPerThread);
template void run_privatized_aggregated<std::uint8_t, std::uint32_t>(const std::uint8_t* data, std::size_t N, std::uint32_t* result, std::uint8_t fromValue, std::uint8_t toValue, int blockSize, int copiesPerBlock, int itemsPerThread);