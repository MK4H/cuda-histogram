#ifndef HISTOGRAM_CUDA_HPP
#define HISTOGRAM_CUDA_HPP

#include "kernels.cuh"
#include "interface.hpp"

#include "cuda/cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cstdint>
#include <algorithm>


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaHistogramAlgorithm : public IHistogramAlgorithm<T, RES>
{
protected:
	const T* mData;
	std::size_t mN;

	T* dData;
	RES* dResults;

	int blockSize;
	int copiesPerBlock;
	int itemsPerThread;
	int chunkSize;
	bool pinned;
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		IHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue, args);
		mData = data;
		mN = N;

		blockSize = args.getArgInt("blockSize").getValue();
		copiesPerBlock = args.getArgInt("privCopies").getValue();
		itemsPerThread = args.getArgInt("itemsPerThread").getValue();
		chunkSize = args.getArgInt("chunkSize").getValue();
		pinned = args.getArgBool("pinned").getValue();
	}
};


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaSimpleAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		CudaHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue, args);

		CUCH(cudaSetDevice(0));

		CUCH(cudaMalloc(&this->dData, this->mN * sizeof(T)));
		CUCH(cudaMalloc(&this->dResults, this->mResult.size() * sizeof(RES)));
	}

	virtual void prepare() override
	{
		if (!this->mData || !this->mN) return;

		CUCH(cudaMemcpy(this->dData, this->mData, this->mN * sizeof(T), cudaMemcpyHostToDevice));
		CUCH(cudaMemset(this->dResults, 0, this->mResult.size() * sizeof(RES)));
	}

	virtual void finalize() override
	{
		CUCH(cudaMemcpy(this->mResult.data(), this->dResults, this->mResult.size() * sizeof(RES), cudaMemcpyDeviceToHost));

		CUCH(cudaFree(this->dData));
		CUCH(cudaFree(this->dResults));
	}
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaNaiveAlgorithm : public CudaSimpleAlgorithm<T, RES>
{
public:
	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_naive(this->dData, this->mN, this->dResults, this->mFromValue, this->mToValue, this->blockSize);
		CUCH(cudaDeviceSynchronize());
	}
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaAtomicAlgorithm : public CudaSimpleAlgorithm<T, RES>
{
public:
	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_atomic(this->dData, this->mN, this->dResults, this->mFromValue, this->mToValue, this->blockSize);
		CUCH(cudaDeviceSynchronize());
	}
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaPrivatizedAlgorithm : public CudaSimpleAlgorithm<T, RES>
{
public:

	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_privatized(
			this->dData,
			this->mN,
			this->dResults,
			this->mFromValue,
			this->mToValue,
			this->blockSize,
			this->copiesPerBlock
		);
		CUCH(cudaDeviceSynchronize());
	}
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaAggregatedAlgorithm : public CudaSimpleAlgorithm<T, RES>
{
public:

	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_aggregated(
			this->dData,
			this->mN,
			this->dResults,
			this->mFromValue,
			this->mToValue,
			this->blockSize,
			this->itemsPerThread
		);
		CUCH(cudaDeviceSynchronize());
	}
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaAtomicShmAlgorithm : public CudaSimpleAlgorithm<T, RES>
{
public:
	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_atomic_shm(
			this->dData,
			this->mN,
			this->dResults,
			this->mFromValue,
			this->mToValue,
			this->blockSize,
			this->copiesPerBlock,
			this->itemsPerThread
		);
		CUCH(cudaDeviceSynchronize());
	}
};


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaOverlapAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
const T* hostData;

public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		CudaHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue, args);

		CUCH(cudaSetDevice(0));
		if (this->pinned) {
			T* pinnedData;
			CUCH(cudaHostAlloc(&pinnedData, this->mN * sizeof(T), cudaHostAllocWriteCombined));
			std::copy(data, data + this->mN, pinnedData);
			hostData = pinnedData;
		}
		else {
			hostData = this->mData;
		}
		CUCH(cudaMalloc(&this->dData, 2 * this->chunkSize * sizeof(T)));
		CUCH(cudaMalloc(&this->dResults, this->mResult.size() * sizeof(RES)));
	}

	virtual void prepare() override
	{
		if (!this->mData || !this->mN) return;

		CUCH(cudaMemset(this->dResults, 0, this->mResult.size() * sizeof(RES)));
	}

	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		const std::size_t numChunks = this->mN / this->chunkSize + (this->mN % this->chunkSize == 0 ? 0 : 1) ;
		for (std::size_t chunk = 0; chunk < numChunks - 1; ++chunk) {
			// Copy data chunk
			CUCH(cudaMemcpy(this->dData + ((chunk % 2) * this->chunkSize), hostData + (chunk * this->chunkSize), this->chunkSize * sizeof(T), cudaMemcpyHostToDevice));
			// Execute
			run_atomic_shm(
				this->dData + ((chunk % 2) * this->chunkSize),
				this->chunkSize,
				this->dResults,
				this->mFromValue,
				this->mToValue,
				this->blockSize,
				this->copiesPerBlock,
				this->itemsPerThread
			);
		}
		// Last chunk, possibly incomplete
		// Copy data chunk
		const std::size_t lastSize = this->mN - (numChunks - 1) * this->chunkSize;
		CUCH(cudaMemcpy(this->dData + (((numChunks - 1) % 2) * this->chunkSize), hostData + ((numChunks - 1) * this->chunkSize), lastSize * sizeof(T), cudaMemcpyHostToDevice));
		// Execute
		run_atomic_shm(
			this->dData + (((numChunks - 1) % 2) * this->chunkSize),
			lastSize,
			this->dResults,
			this->mFromValue,
			this->mToValue,
			this->blockSize,
			this->copiesPerBlock,
			this->itemsPerThread
		);

		CUCH(cudaDeviceSynchronize());
	}

	virtual void finalize() override
	{
		CUCH(cudaMemcpy(this->mResult.data(), this->dResults, this->mResult.size() * sizeof(RES), cudaMemcpyDeviceToHost));

		CUCH(cudaFree(this->dData));
		CUCH(cudaFree(this->dResults));

		if (this->pinned) {
			CUCH(cudaFreeHost(const_cast<T*>(hostData)));
		}
	}
};


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaFinalAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		CudaHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue, args);

		CUCH(cudaSetDevice(0));
		CUCH(cudaMallocManaged(&this->dData, this->mN * sizeof(T)));
		CUCH(cudaMallocManaged(&this->dResults, this->mResult.size() * sizeof(RES)));
	}

	virtual void prepare() override
	{
		if (!this->mData || !this->mN) return;

		// Best would be to read the data straight from the file to the unified memory
		// but I don't wnat to mess with the code in histogram.cpp
		std::copy(this->mData, this->mData + this->mN, this->dData);
		CUCH(cudaMemset(this->dResults, 0, this->mResult.size() * sizeof(RES)));
	}

	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_atomic_shm(
			this->dData,
			this->mN,
			this->dResults,
			this->mFromValue,
			this->mToValue,
			this->blockSize,
			this->copiesPerBlock,
			this->itemsPerThread
		);
		CUCH(cudaDeviceSynchronize());
	}

	virtual void finalize() override
	{
		std::copy(this->dResults, this->dResults + this->mResult.size(), this->mResult.data());

		CUCH(cudaFree(this->dData));
		CUCH(cudaFree(this->dResults));
	}
};


#endif
