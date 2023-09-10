// simple
#include <cudaimproc/cudacheck.h>
#include <cudaimproc/imgio.h>
//
#include <cuda_runtime.h>
#include <iostream>
#include <optional>

namespace cudaimproc {

__global__ void gen_gradient(unsigned char *pixels,
                             const int imwidth,
                             const int imheight) {
  //
  int row = threadIdx.x; // local thread index
  row +=
      blockIdx.x * blockDim.x; // thread block id * number
                               // of threads_per_block
  int bytes_per_pixel = 3;
  int per_scanline = imwidth * bytes_per_pixel;

  float g = static_cast<float>(row) /
            static_cast<float>(imheight - 1);
  if (row >= imheight)
    return;
  for (int col = 0; col < imwidth; ++col) {
    //
    float r = static_cast<float>(col) /
              static_cast<float>(imwidth - 1);
    float b = 0.25f;

    unsigned char red =
        static_cast<unsigned char>(r * 255.99);
    unsigned char green =
        static_cast<unsigned char>(g * 255.99);
    unsigned char blue =
        static_cast<unsigned char>(b * 255.99);
    //
    int index = col * bytes_per_pixel + row * per_scanline;
    pixels[index] = red;
    pixels[index + 1] = green;
    pixels[index + 2] = blue;
  }
}

}; // namespace cudaimproc

int main(void) { // yep this is (void) type of day
                 //
  float aspect_ratio = 16.0f / 9.0f;
  const std::size_t imwidth = 640;
  const std::size_t imheight = static_cast<int>(
      static_cast<float>(imwidth) / aspect_ratio);
  std::size_t threads_per_block = 64;
  int remainder = imheight % threads_per_block;
  std::size_t nb_blocks =
      (remainder) != 0
          ? (imheight + threads_per_block - remainder) /
                threads_per_block
          : imheight / threads_per_block;
  std::size_t bytes_per_line = imwidth * 3;
  std::size_t imsize = bytes_per_line * imheight;
  unsigned char *pixels_device{nullptr};
  //
  // cuda malloc
  CUDA_CHECK(cudaMalloc((void **)(&pixels_device),
                        imsize * sizeof(unsigned char)));

  //
  cudaimproc::
      gen_gradient<<<nb_blocks, threads_per_block>>>(
          pixels_device, imwidth, imheight);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  //
  unsigned char *pixels_host = new unsigned char[imsize];
  CUDA_CHECK(cudaMemcpy(pixels_host, pixels_device,
                        imsize * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(pixels_device));

  //
  cudaimproc::render(std::make_optional(pixels_host),
                       imheight, imwidth);
  // cuda_imgproc::render(std::nullopt, imheight, imwidth);
  delete[] pixels_host;
  return 0;
}
