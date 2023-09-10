#include <cudaimproc/cudacheck.h>
#include <cudaimproc/imgio.h>
//
#include <matrix/matrix.cuh>
//
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <optional>

namespace cudaimproc {
__device__ cudamat::MatN<float, 3, 3> mk_ymat() {
  float sobel_1[] = {
      1, 2, 1, // first row
  };
  float sobel_2[] = {
      0, 0, 0, // second row
  };
  float sobel_3[] = {
      -1, -2, -1 // third row
  };
  cudamat::MatN<float, 3, 3> sobel_m;
  sobel_m.set_row(0, sobel_1);
  sobel_m.set_row(1, sobel_2);
  sobel_m.set_row(2, sobel_3);
  return sobel_m;
}
__device__ cudamat::MatN<float, 3, 3> mk_xmat() {
  float sobel_1[] = {
      1, 0, -1, // first row
  };
  float sobel_2[] = {
      2, 0, -2, // second row
  };
  float sobel_3[] = {
      1, 0, -1 // third row
  };
  cudamat::MatN<float, 3, 3> sobel_m;
  sobel_m.set_row(0, sobel_1);
  sobel_m.set_row(1, sobel_2);
  sobel_m.set_row(2, sobel_3);
  return sobel_m;
}
//
__global__ void sobelXY_3x3(unsigned char *in_img,
                            unsigned char *out_img,
                            const int imwidth,
                            const int imheight,
                            const int rgb_offset) {
  //

  unsigned int scol_nb = 0;
  unsigned int srow_nb = 0;
  cudamat::MatN<float, 3, 3> sobel_y = mk_ymat();
  sobel_y.col_nb(scol_nb);
  sobel_y.row_nb(srow_nb);
  cudamat::MatN<float, 3, 3> sobel_x = mk_xmat();
  int col = threadIdx.x; // local thread index
  col +=
      blockIdx.x * blockDim.x; // thread block id * number
                               // of threads_per_block
  if (col >= imwidth) {
    return;
  }
  if (col < scol_nb) {
    // do padding or nothing
    return;
  }
  int bytes_per_pixel = 3;
  int per_scanline = imwidth * bytes_per_pixel;

  for (int row = srow_nb; row < imheight; ++row) {
    //
    float gx = 0.0f;
    float gy = 0.0f;
    for (unsigned int sr = 0; sr < srow_nb; ++sr) {
      for (unsigned int sc = 0; sc < scol_nb; ++sc) {
        //
        int index = (col - sc) * bytes_per_pixel +
                    (row - sr) * per_scanline;
        float scell;
        sobel_y.get(sr, sc, scell);
        gy += scell * in_img[index + rgb_offset];
        sobel_x.get(sr, sc, scell);
        gx += scell * in_img[index + rgb_offset];
      }
    }
    int index = col * bytes_per_pixel + row * per_scanline;
    float g = std::sqrt(gx * gx + gy * gy);
    out_img[index + rgb_offset] =
        static_cast<unsigned char>(g);
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
  //
  // execution config
  const std::size_t threads_per_block = 64;
  const std::size_t nb_streams =
      3; // 1 for each rgb component
  const std::size_t streamSize = imgSize / nb_streams;
  cudaStream_t streams[nb_streams];

  // create cuda stream
  for (int i = 0; i < nb_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  //
  unsigned char *d_out_img{nullptr};
  unsigned char *d_in_img{nullptr};
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&d_out_img), imgSizeByte));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void **>(&d_in_img), imgSizeByte));

  CUDA_CHECK(
      cudaMemcpy(reinterpret_cast<void *>(d_in_img),
                 reinterpret_cast<const void *>(info.data),
                 imgSizeByte, cudaMemcpyHostToDevice));

  // launch kernel
  for (int i = 0; i < nb_streams; ++i) {
    cudaimproc::
        sobelXY_3x3<<<streamSize / threads_per_block,
                      threads_per_block, 0, streams[i]>>>(
            d_in_img, d_out_img, info.width, info.height,
            i);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // destroy cuda stream
  for (int i = 0; i < nb_streams; ++i) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
  CUDA_CHECK(cudaFree(d_in_img));
  unsigned char *out_img = new unsigned char[imgSize];
  CUDA_CHECK(
      cudaMemcpy(reinterpret_cast<void *>(out_img),
                 reinterpret_cast<const void *>(d_out_img),
                 imgSizeByte, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out_img));
  //
  cudaimproc::render(std::make_optional(out_img),
                     info.height, info.width, info.channels,
                     "sobelXY_img");
  // cuda_imgproc::render(std::nullopt, imheight, imwidth);
  delete[] out_img;
  return 0;
}
