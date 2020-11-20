#ifndef HISTOGRAM_CUDA_HPP
#define HISTOGRAM_CUDA_HPP

#include "kernels.cuh"
#include "interface.hpp"

#include "cuda/cuda.hpp"

#include <vector>
#include <cstdint>


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaHistogramAlgorithm : public IHistogramAlgorithm<T, RES>
{
protected:
	const T* mData;
	std::size_t mN;

public:
	virtual void initialize(const T* data, std::size_t N, T fromValue, T toValue, bpp::ProgramArguments& args) override
	{
		IHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue, args);
		mData = data;
		mN = N;

		CUCH(cudaSetDevice(0));
		
		// TODO alocate buffers
	}

	virtual void prepare() override
	{
		if (!mData || !mN) return;
		
		// TODO copy data
	}

	virtual void finalize() override
	{
		// TODO copy data
	}
};


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class CudaHistogramAlgorithmFirst : public CudaHistogramAlgorithm<T, RES>
{
public:
	virtual void run() override
	{
		if (!this->mData || !this->mN) return;
		
		// Execute

		CUCH(cudaDeviceSynchronize());
	}
};


#endif
