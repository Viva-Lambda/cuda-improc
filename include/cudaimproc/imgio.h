#pragma once

#include <optional>

namespace cudaimproc {

void render(std::optional<unsigned char *> pixels_opt,
            int imheight, int imwidth, int channel = 3,
            const char *imname = "imgrad");
}
