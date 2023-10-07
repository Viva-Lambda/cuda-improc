#ifndef EXECONFIG_H
#define EXECONFIG_H
// execution config handler for cuda kernels
#include <tuple>
#include <vector>

namespace cudaimproc {

struct ExecutionConfig1D {

  ExecutionConfig1D() = delete;

  //! \brief cuda stream based execution config
  ExecutionConfig1D(std::size_t nb_elements_to_process,
                    const std::size_t threads_per_b = 32,
                    const std::size_t nb_s = 1);

  std::size_t block_nb(std::size_t stream_index = 0) const;
  std::size_t nb_threads() const;
  std::size_t nb_streams() const;
  std::vector<std::pair<unsigned int, unsigned int>>
  chunks() const;

private:
  std::size_t
  find_nb_blocks(std::size_t nb_elements_to_process) const;

  void mk_stream_chunks(
      std::size_t total_nb_elements_to_process);

  const std::size_t threads_per_block{32};
  const std::size_t stream_nb{1};

  std::vector<std::pair<unsigned int, unsigned int>>
      stream_chunks;
};

} // namespace cudaimproc

#endif
