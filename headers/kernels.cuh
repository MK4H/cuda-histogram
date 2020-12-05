#ifndef HISTOGRAM_KEKRNELS_CUH
#define HISTOGRAM_KEKRNELS_CUH

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

template<typename T, typename RES>
void run_naive(
    const T*    data,
    std::size_t N,
    RES*        result,
    T           fromValue,
    T           toValue,
    int         blockSize
);

template<typename T, typename RES>
void run_atomic(
    const T*    data,
    std::size_t N,
    RES*        result,
    T           fromValue,
    T           toValue,
    int         blockSize
);

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
void run_atomic_shm(
    const T*        data,
    std::size_t     N,
    RES*            result,
    T               fromValue,
    T               toValue,
    int             blockSize,
    int             copiesPerBlock,
    int             itemsPerThread,
    cudaStream_t    stream = (cudaStream_t)0
);

#endif
