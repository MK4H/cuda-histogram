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
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		IHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue, args);
		mData = data;
		mN = N;

		CUCH(cudaSetDevice(0));


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
class CudaBinParallelAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_bin_parallel(this->dData, this->mN, this->dResults, this->mFromValue, this->mToValue);
		CUCH(cudaDeviceSynchronize());
	}
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaAtomicsAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_atomics(this->dData, this->mN, this->dResults, this->mFromValue, this->mToValue);
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
		blockSize = args.getArgInt("blockSize").getValue();
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
			blockSize,
			copiesPerBlock
		);
		CUCH(cudaDeviceSynchronize());
	}
private:
	int blockSize;
	int copiesPerBlock;
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaAggregatedAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		CudaHistogramAlgorithm<T,RES>::initialize(data, N, fromValue, toValue, args);
		blockSize = args.getArgInt("blockSize").getValue();
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
			blockSize,
			itemsPerThread
		);
		CUCH(cudaDeviceSynchronize());
	}
private:
	int blockSize;
	int itemsPerThread;
};

template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaPrivatizedAggregatedAlgorithm : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		CudaHistogramAlgorithm<T,RES>::initialize(data, N, fromValue, toValue, args);
		blockSize = args.getArgInt("blockSize").getValue();
		copiesPerBlock = args.getArgInt("privCopies").getValue();
		itemsPerThread = args.getArgInt("itemsPerThread").getValue();
	}

	virtual void run() override
	{
		if (!this->mData || !this->mN) return;

		// Execute
		run_privatized_aggregated(
			this->dData,
			this->mN,
			this->dResults,
			this->mFromValue,
			this->mToValue,
			blockSize,
			copiesPerBlock,
			itemsPerThread
		);
		CUCH(cudaDeviceSynchronize());
	}
private:
	int blockSize;
	int copiesPerBlock;
	int itemsPerThread;
};

#endif
