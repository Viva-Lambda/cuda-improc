#include <cudaimproc/imgio.h>

#include <iostream>

//
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

//
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

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

img_info::img_info(unsigned char *d, std::size_t w,
                   std::size_t h, std::size_t c,
                   const char *n)
    : data{nullptr}, width(w), height(h), channels(c),
      name(n) {
  std::size_t imsize = width * height * channels;
  data = new unsigned char[imsize];
  std::copy(d, d + (width * h * c),
            data); // copy the data into p2
}

img_info imread(std::filesystem::path impath) {
  //
  int w, h, c;
  std::filesystem::path im_p = impath.make_preferred();
  const char *img_p = im_p.c_str();
  unsigned char *img = stbi_load(img_p, &w, &h, &c, 0);
  img_info info(img, w, h, c, "in_image");
  return info;
}

} // namespace cudaimproc
