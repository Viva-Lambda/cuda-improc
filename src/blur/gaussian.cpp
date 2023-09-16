#include <cudaimproc/cudacheck.h>
#include <cudaimproc/imgio.h>
//
#include <matrix/matrix.cuh>
//
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <optional>

namespace cudaimproc {}

int main() {

  cudaimproc::img_info info = cudaimproc::owl_img();
}
