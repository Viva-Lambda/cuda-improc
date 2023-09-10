#include <cudaimproc/imgio.h>

#include <iostream>

//
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace cudaimproc {

void render(std::optional<unsigned char *> pixels_opt,
            int imheight, int imwidth, int channel,
            const char *imname) {
  if (pixels_opt.has_value()) {
    unsigned char *pixels_host = *pixels_opt;
    int bytes_per_pixel = channel;
    int bytes_per_line = imwidth * bytes_per_pixel;
    std::string name(imname);
    name += ".png";
    const char *fname = name.c_str();
    stbi_write_png(fname, imwidth, imheight,
                   bytes_per_pixel, pixels_host,
                   bytes_per_line);

  } else {
    std::cout << "P3" << std::endl;
    std::cout << imwidth << " " << imheight << std::endl;
    std::cout << "255" << std::endl;
    for (int j = imheight - 1; j >= 0; --j) {
      for (int i = 0; i < imwidth; ++i) {
        auto r = static_cast<double>(i) / (imwidth - 1);
        auto g = static_cast<double>(j) / (imheight - 1);
        auto b = 0.25;

        int ir = static_cast<int>(255.999 * r);
        int ig = static_cast<int>(255.999 * g);
        int ib = static_cast<int>(255.999 * b);
        std::cout << ir << ' ' << ig << ' ' << ib
                  << std::endl;
      }
    }
  }
}

} // namespace cuda_imgproc
