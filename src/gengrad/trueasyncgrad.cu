// simple
#include <cudaimproc/cudacheck.h>
#include <cudaimproc/execonfig.h>
#include <cudaimproc/imgio.h>
//
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
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

  int bytes_per_pixel = 3;
  int per_scanline = imwidth * bytes_per_pixel;

  float g = static_cast<float>(col) /
            static_cast<float>(imwidth - 1);
  unsigned char green =
      static_cast<unsigned char>(g * 255.99);

  float b = 0.25f;
  unsigned char blue =
      static_cast<unsigned char>(b * 255.99);

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

int main() {
  std::filesystem::path img_dir(IMAGE_DIR);
  std::filesystem::path imname("owl.jpg");
  std::filesystem::path imgp = img_dir / imname;

  // image config
  cudaimproc::img_info info = cudaimproc::imread(imgp);
  const std::size_t imgSize =
      info.width * info.height * info.channels;
  const std::size_t imgSizeByte =
      imgSize * sizeof(unsigned char);

  cudaError_t res = cudaGetLastError();
  std::cout << "CUDA ERROR :: " << cudaGetErrorName(res)
            << std::endl;

  // kernel execution config
  const std::size_t threads_per_block = 64;
  const std::size_t nb_streams =
      3; // 1 for each rgb component
  cudaimproc::ExecutionConfig1D config(
      info.height, threads_per_block, nb_streams);
  const std::size_t streamSize =
      imgSize / config.nb_streams();

  cudaStream_t streams[nb_streams];

  // create cuda stream
  for (int i = 0; i < nb_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }
  //

  // pin host and device memory

  //
  std::cout << " height " << info.height << " width "
            << info.width << std::endl;
  for (auto [start, end] : config.chunks()) {
    std::size_t stream_size =
        info.width * (end - start) * sizeof(unsigned char);
    //
    std::cout << "start " << start << " end " << end
              << std::endl;
  }
  //
  // cuda malloc
  // destroy resources
  for (int i = 0; i < nb_streams; ++i) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
  return 0;
}
