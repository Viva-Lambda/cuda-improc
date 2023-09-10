#ifndef CUDACHECK_H
#define CUDACHECK_H
#include <cuda_runtime.h>
#include <iostream>
namespace cuda_imgproc {
void print_cuda_error(cudaError_t res,
                      const char *const func_name,
                      const char *const file_name,
                      const int line) {
  if (res != cudaSuccess) {
    std::cout << "CUDA ERROR :: "
              << static_cast<unsigned int>(res) << " "
              << cudaGetErrorName(res) << " in function "
              << func_name << " at line " << line
              << " of file " << file_name << std::endl;
    cudaDeviceReset();
    exit(99);
  }
};
} // namespace cuda_imgproc

#define CUDA_CHECK(v)                                      \
  cuda_imgproc::print_cuda_error((v), #v, __FILE__,        \
                                 __LINE__)

#endif
