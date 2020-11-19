#ifndef HISTOGRAM_CUDA_HPP
#define HISTOGRAM_CUDA_HPP

#include "kernels.cuh"
#include "interface.hpp"

#include "cuda/cuda.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cstdint>


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaHistogramAlgorithm : public IHistogramAlgorithm<T, RES>
{
protected:
	const T* mData;
	std::size_t mN;

	T* dData;
	RES* dResults;
	int blockSize;
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		IHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue, args);
		mData = data;
		mN = N;

		CUCH(cudaSetDevice(0));

		blockSize = args.getArgInt("blockSize").getValue();
		CUCH(cudaMalloc(&dData, mN * sizeof(T)));
		CUCH(cudaMalloc(&dResults, this->mResult.size() * sizeof(RES)));
	}

	virtual void prepare() override
	{
		if (!mData || !mN) return;

		CUCH(cudaMemcpy(dData, mData, mN * sizeof(T), cudaMemcpyHostToDevice));
		CUCH(cudaMemset(dResults, 0, this->mResult.size() * sizeof(RES)));
	}

	virtual void finalize() override
	{
		CUCH(cudaMemcpy(this->mResult.data(), dResults, this->mResult.size() * sizeof(RES), cudaMemcpyDeviceToHost));

		CUCH(cudaFree(dData));
		CUCH(cudaFree(dResults));
	}
};


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaNaiveAlgorithm : public CudaHistogramAlgorithm<T, RES>
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
class CudaAtomicAlgorithm : public CudaHistogramAlgorithm<T, RES>
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
class CudaPrivatizedAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		CudaHistogramAlgorithm<T,RES>::initialize(data, N, fromValue, toValue, args);
		copiesPerBlock = args.getArgInt("privCopies").getValue();
	}

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
			copiesPerBlock
		);
		CUCH(cudaDeviceSynchronize());
	}
private:
	int copiesPerBlock;
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaAggregatedAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		CudaHistogramAlgorithm<T,RES>::initialize(data, N, fromValue, toValue, args);
		itemsPerThread = args.getArgInt("itemsPerThread").getValue();
	}

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
			itemsPerThread
		);
		CUCH(cudaDeviceSynchronize());
	}
private:
	int itemsPerThread;
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaAtomicShmAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		CudaHistogramAlgorithm<T,RES>::initialize(data, N, fromValue, toValue, args);
		copiesPerBlock = args.getArgInt("privCopies").getValue();
		itemsPerThread = args.getArgInt("itemsPerThread").getValue();
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
			copiesPerBlock,
			itemsPerThread
		);
		CUCH(cudaDeviceSynchronize());
	}
private:
	int copiesPerBlock;
	int itemsPerThread;
};

#endif
