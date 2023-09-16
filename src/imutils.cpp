#include <cudaimproc/imutils.h>

namespace cudaimproc {

img_info owl_img() {
  std::filesystem::path img_dir(IMAGE_DIR);
  std::filesystem::path imname("owl.jpg");
  std::filesystem::path imgp = img_dir / imname;

  // image config
  img_info info = cudaimproc::imread(imgp);
  return info;
}
} // namespace cudaimproc
