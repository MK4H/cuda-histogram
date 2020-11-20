#ifndef HISTOGRAM_SERIAL_HPP
#define HISTOGRAM_SERIAL_HPP

#include "interface.hpp"

#include <vector>
#include <cstdint>


template<typename T = std::uint8_t, typename RES = std::uint32_t>
class SerialHistogramAlgorithm : public IHistogramAlgorithm<T, RES>
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
	}

	virtual void run() override
	{
		if (!mData || !mN) return;

		for (std::size_t i = 0; i < mN; ++i) {
			T val = mData[i];
			if (val >= this->mFromValue && val <= this->mToValue) {
				this->mResult[val - this->mFromValue]++;
			}
		}
	}
};


#endif
