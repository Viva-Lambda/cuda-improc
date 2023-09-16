#ifndef UTILS_H
#define UTILS_H

#include <tuple>
#include <vector>

namespace cudaimproc {
std::vector<std::pair<unsigned int, unsigned int>>
chunker(const unsigned int seq_size,
        unsigned int chunk_size);
// adapted from https://stackoverflow.com/a/9943098
// divides the container into inclusive chunks where
// each chunk contains k elements except the last one
// chunks([0,1,2,3], 3) would give [(0,1,2), (3)]
// the Fn should be a function that accumulates chunks into
// list
template <typename Container, typename Fn>
void chunks(const Container &seq, std::size_t k, Fn f) {
  auto size = seq.size();
  std::size_t i = 0;
  bool last_added = false;

  if (size > k) {
    while (i < (size - k)) {
      std::size_t start = i;
      std::size_t end = i + k - 1;
      std::size_t next_start = i + k;
      if (next_start >= (size - 1)) {
        f(i, size - 1);
        last_added = true;
        break;
      }
      f(start, end);
      i = next_start;
    }
  }
  //
  if (last_added == false) {
    f(i, size - 1);
  }
}
} // namespace cudaimproc
#endif
