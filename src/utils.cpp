#include <cudaimproc/utils.h>

namespace cudaimproc {
std::vector<std::pair<unsigned int, unsigned int>>
chunker(const unsigned int seq_size,
        unsigned int chunk_size) {
  std::vector<unsigned int> indices;
  for (unsigned int i = 0; i < seq_size; i++) {
    indices.push_back(i);
  }

  std::vector<std::pair<unsigned int, unsigned int>>
      start_ends;
  auto fn = [&](unsigned int b, unsigned int e) {
    auto pos = std::make_pair(b, e);
    start_ends.emplace_back(pos);
  };
  chunks(indices, chunk_size, fn);
  return start_ends;
}

} // namespace cudaimproc
