#ifndef HISTOGRAM_KEKRNELS_CUH
#define HISTOGRAM_KEKRNELS_CUH

#include <cstddef>
#include <cstdint>

template<typename T, typename RES>
void run_bin_parallel(const T* data, std::size_t N, RES* result, T fromValue, T toValue);

template<typename T, typename RES>
void run_atomics(const T* data, std::size_t N, RES* result, T fromValue, T toValue);

template<typename T, typename RES>
void run_privatized(
    const T*    data,
    std::size_t N,
    RES*        result,
    T           fromValue,
    T           toValue,
    int         blockSize,
    int         copiesPerBlock
);

template<typename T, typename RES>
void run_aggregated(
    const T*    data,
    std::size_t N,
    RES*        result,
    T           fromValue,
    T           toValue,
    int         blockSize,
    int         itemsPerThread
);

template<typename T, typename RES>
void run_privatized_aggregated(
    const T*    data,
    std::size_t N,
    RES*        result,
    T           fromValue,
    T           toValue,
    int         blockSize,
    int         copiesPerBlock,
    int         itemsPerThread
);

#endif
