# Assignment 2 - CUDA Histogram

The assignment is started in labs as a three episodes saga:
- [Part 1](https://www.ksi.mff.cuni.cz/teaching/nprg058-web/pages/assignments/cuda1) the atomic instructions
- [Part 2](https://www.ksi.mff.cuni.cz/teaching/nprg058-web/pages/assignments/cuda2) - kernel optimizations
- [Part 3](https://www.ksi.mff.cuni.cz/teaching/nprg058-web/pages/assignments/cuda3) - GPU-host memory transfers

The last part contains the details required for the implementation.

Write a CUDA accelerated implementation that will compute a histogram (frequency analysis) of a text file. The input is a plain text where one char is one byte (we do not bother with encoding details). The histogram holds number of occurrences for each character. The range of characters in the histogram may be adjusted by algorithm parameters, range 0-127 is computed by default.

Your solution needs to implement following algorithms (please use the designated names in your CLI arguments):

- serial - CPU sequential baseline (already implemented in given source codes)
- naive - solution without any synchronization where one thread computes one histogram bin
- atomic - simple atomic updates (no privatization, no shared memory)
- atomic_shm - atomic updates with private copy in shared memory (no privatization in global memory, --itemsPerThread indicate how many input chars are processed in one thread, and --privCopies indicate how many histogram copies are placed in shared memory)
- overlap - same as atomic_shm, but the input data are transferred in chunks and overlap with computations (add CLI arguments: --chunkSize = number of chars in one chunk and --pinned = bool flag that indicates the chunk buffers has to be allocated as portable memory)
- final - same as atomic_shm, but either using memory mapping or unified memory (your choice should be mentioned in readme file in your repository).

All algorithms should also reflect --blockSize that sets number of threads in each CUDA block.


# Solution

The solution includes all the algorithms required by the assignment, namely:
- naive
- atomic
- atomic_shm
- overlap
- final

Additionaly, there are two more algorithms implemented as part of the CUDA Histogram II assignment.
These are:
- privatized
- aggregated

Privatized extends the atomic solution by using private copies of the output histogram for each block.
To merge the block private copies, each thread in the block gets one or more values which it
merges the results of from all the copies. This ensures that we need no synchronization when
merging the private histogram copies. When results are merged for the given value, the
global histogram is updated using a single atomic instaruction.

This could probably be done in a few different ways, but I have chosen this implementation due to it's minimal global memory requirements and simple evolution from the atomic algorithm implementation.

Aggregated extends the atomic solution by processing multiple input characters in each thread.
All warp threads access consecutive input characters, optimising memory access paterns.
The input is read using the following pattern (blockID.threadID):
1.1 1.2 1.1 1.2 2.1 2.2 2.1 2.2 3.1 3.2 3.1 3.2

Each block receives a separate part of the input, processing the given
part while maintaining the correct access pattern for each warp.

These two optimizations are then combined into the atomic_shm algorithm, which is part
of the final assignment.


atomic_shm, overlap and final algorithms use the exact same kernel implementation,
the atomic_shm implementation. The only difference is allocation and access patterns
when working with the GPU memory.


The atomic_shm solution uses explicit cudaMemcopy calls to copy the input during
Preparations phase and output during the Finalization phase.

Overlap solution postpones copying of the input data to Execution phase,
so that the data transfer can be overlaped with kernel execution.
The input data is transfered in chunks, running the CUDA kernel for each
chunk separately and in parallel with the data transfer of the following chunk.

The time to transfer the data is much larger than the time needed to process the data,
as can be seen in the atomic_shm algorithm measurement, where the Preparations phase
is basically just data transfer and Execution phase is just the kernel execution.
This makes the overlap solution much less useful.

The final solution uses unified memory, allocated using cudaMallocManaged, to work
with the GPU memory. This allows us to access the pointer both in the host code and
in the CUDA kernel code. To demonstrate this, we use std::copy to copy the data
from the input file to the GPU buffer, instead of cudaMemcpy as is used in the atomic_shm
algorithm.

As we can see from the measurment, this way of working with the buffer is far from optimal.

## Measurements

| Algorithm | Preparations | Execution | Finalization |
| --------- | ------------ | --------- | ------------ |
| serial | 0.000196 ms | 2247.3 ms | 0.000144 ms |
| naive | 483.871 ms | 156536 ms | 2.64712 ms |
| atomic | 483.869 ms | 819.614 ms | 2.61886 ms |
| atomic_shm | 489.821 ms | 6.34248 ms | 2.62634 ms |
| overlap | 0.024907 ms | 235.085 ms | 196.242 ms |
| final | 1160.67 ms | 424.686 ms | 128.244 ms |

The measurements were done using the `measure.sh` script. If you
want to generate the table again from new measurements,
use the AWK script `output_to_table.awk`.
