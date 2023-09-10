#pragma once

#include <filesystem>
#include <optional>

namespace cudaimproc {

struct img_info {
  unsigned char *data;
  std::size_t width, height, channels;
  const char *name;
  img_info() = delete;
  img_info(unsigned char *d, std::size_t w, std::size_t h,
           std::size_t c, const char *n);
  ~img_info() { delete[] data; }
};

void render(std::optional<unsigned char *> pixels_opt,
            int imheight, int imwidth, int channel = 3,
            const char *imname = "imgrad");

img_info imread(std::filesystem::path impath);
} // namespace cudaimproc
