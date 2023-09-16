// simple
#include <cudaimproc/cudacheck.h>
#include <cudaimproc/execonfig.h>
#include <cudaimproc/imgio.h>
//
#include <cuda_runtime.h>
#include <iostream>
#include <optional>

namespace cudaimproc {
__global__ void
gen_gradient_async(unsigned char *pixels, const int imwidth,
                   const int imheight,
                   const std::size_t stream_offset) {
  //
  int col = threadIdx.x; // local thread index
  col +=
      blockIdx.x * blockDim.x; // thread block id * number
                               // of threads_per_block
  if (col >= imwidth) {
    return;
  }

  if (rgb_offset == 2) {
  }
  int bytes_per_pixel = 3;
  int per_scanline = imwidth * bytes_per_pixel;

  float g = static_cast<float>(col) /
            static_cast<float>(imwidth - 1);
  unsigned char green =
      static_cast<unsigned char>(g * 255.99);

  float b = 0.25f;
  unsigned char blue = static_cast<unsigned char>(b * 255.99);

  for (int row = 0; row < imheight; ++row) {
    //
    float r = static_cast<float>(row) /
              static_cast<float>(imheight - 1);
    unsigned char red =
        static_cast<unsigned char>(r * 255.99);
    // TODO write offset code
  }
}
}; // namespace cudaimproc

int main(void) { // yep this is (void) type of day
                 //

  // image config
  const float aspect_ratio = 16.0f / 9.0f;
  const std::size_t imwidth = 640;
  const std::size_t imheight = static_cast<int>(
      static_cast<float>(imwidth) / aspect_ratio);
  const std::size_t bytes_per_line = imwidth * 3;
  const std::size_t imsize = bytes_per_line * imheight;
  const std::size_t imsizeInBytes =
      imsize * sizeof(unsigned char);

  // kernel config

  const std::size_t threads_per_block = 64;
  const std::size_t nb_streams =
      3; // 1 for each rgb component
  cudaimproc::ExecutionConfig1D config(
      imheight, threads_per_block, nb_streams);
  const std::size_t streamSize =
      imsize / config.nb_streams();
  cudaStream_t streams[nb_streams];

  // create cuda stream
  for (int i = 0; i < nb_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  unsigned char *pixels_device{nullptr};
  //
  // cuda malloc
  CUDA_CHECK(
      cudaMalloc((void **)(&pixels_device), imsizeInBytes));

  //

  for (int i = 0; i < nb_streams; ++i) {
    std::size_t sblock_nb =
        streamSize / config.nb_threads();
    cudaimproc::gen_gradient_async<<<
        sblock_nb, config.nb_threads(), 0, streams[i]>>>(
        pixels_device, imwidth, imheight, i);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  //
  unsigned char *pixels_host = new unsigned char[imsize];
  CUDA_CHECK(cudaMemcpy(pixels_host, pixels_device,
                        imsizeInBytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(pixels_device));

  //
  cudaimproc::render(std::make_optional(pixels_host),
                     imheight, imwidth, 3, "asyncimg");
  // destroy resources
  for (int i = 0; i < nb_streams; ++i) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
  delete[] pixels_host;
  return 0;
}
