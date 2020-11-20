#ifndef HISTOGRAM_KEKRNELS_CUH
#define HISTOGRAM_KEKRNELS_CUH

#include <cstddef>
#include <cstdint>

template<typename T, typename RES>
void run_hist(const T* data, std::size_t N, RES* result, T fromValue, T toValue);

#endif
