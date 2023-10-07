#include <cudaimproc/execonfig.h>
#include <cudaimproc/utils.h>
#include <stdexcept>
#include <string>

namespace cudaimproc {

std::vector<std::pair<unsigned int, unsigned int>>
ExecutionConfig1D::chunks() const {
  return stream_chunks;
}

// find total number of blocks when give number of elements
// to process
std::size_t ExecutionConfig1D::find_nb_blocks(
    std::size_t nb_elements_to_process) const {
  int remainder =
      nb_elements_to_process % threads_per_block;

  std::size_t total_b = 1;
  if (remainder == 0) {
    total_b = static_cast<std::size_t>(
        nb_elements_to_process / threads_per_block);
  } else if (remainder != 0) {
    std::size_t temp =
        nb_elements_to_process + threads_per_block;
    total_b = static_cast<std::size_t>((temp - remainder) /
                                       remainder);
  }
  return total_b;
}

// finds number of elements to be processed by each stream
void ExecutionConfig1D::mk_stream_chunks(
    std::size_t total_nb_elements_to_process) {
  std::size_t stream_size = static_cast<std::size_t>(
      total_nb_elements_to_process / stream_nb);
  stream_chunks.clear();
  stream_chunks =
      chunker(total_nb_elements_to_process, stream_size);
}
std::size_t ExecutionConfig1D::nb_threads() const {
  return threads_per_block;
}
std::size_t ExecutionConfig1D::nb_streams() const {
  return stream_nb;
}

ExecutionConfig1D::ExecutionConfig1D(
    std::size_t nb_elements_to_process,
    const std::size_t threads_per_b, const std::size_t nb_s)
    : threads_per_block(threads_per_b), stream_nb(nb_s) {
  mk_stream_chunks(nb_elements_to_process);
  // distribute this to streams
}

// get block number given streams index
std::size_t ExecutionConfig1D::block_nb(
    std::size_t stream_index) const {
  if (stream_index >= stream_nb) {
    std::string indx = std::to_string(stream_index);
    std::string nbs = std::to_string(stream_nb);
    std::string msg = "given stream index " + indx;
    msg += " is >= available number of streams " + nbs;
    throw std::runtime_error(msg);
  }
  std::pair<unsigned int, unsigned int> stream_start_end =
      stream_chunks[stream_index];
  auto start = stream_start_end.first;
  auto end = stream_start_end.second;
  std::size_t nb_elements_to_process_by_stream =
      (end + 1) - start;
  std::size_t block_size =
      find_nb_blocks(nb_elements_to_process_by_stream);
  return block_size;
}
} // namespace cudaimproc
