#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <math.h>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <vector>

namespace cudamat {

enum mstatus_t : std::uint_least8_t {
  SUCCESS = 1,
  INDEX_ERROR = 2,
  SIZE_ERROR = 3,
  ARG_ERROR = 4,
  LU_ERROR = 5,
  NOT_IMPLEMENTED = 6
};

struct MResult {
  //
  unsigned int line_info = 0;
  const char *file_name = "";
  const char *fn_name = "";
  const char *call_name = "";
  const char *duration_info = "";

  mstatus_t status;
  bool success = false;

  __host__ __device__ MResult() {}
  __host__ __device__ MResult(unsigned int line,
                              const char *fname,
                              const char *funcname,
                              const char *cname,
                              mstatus_t op)
      : line_info(line), file_name(fname),
        fn_name(funcname), call_name(cname), status(op),
        success(op == SUCCESS) {}
};

template <class T = float, unsigned int RowNb = 1,
          unsigned int ColNb = RowNb>
class MatN {
  /** holds the vector data*/
  T data[ColNb * RowNb];
  static const unsigned int nb_rows = RowNb;
  static const unsigned int nb_cols = ColNb;
  static const unsigned int size = ColNb * RowNb;
  static const unsigned int sizeInBytes =
      ColNb * RowNb * sizeof(T);

public:
  __host__ __device__ MatN() {
    for (unsigned int i = 0; i < RowNb * ColNb; i++) {
      data[i] = static_cast<T>(0);
    }
    // lu_decomposition = LUdcmp<T, RowNb, ColNb>(data);
  }
  __host__ __device__ ~MatN() { delete[] data; }
  __host__ __device__ MatN(const T vd[RowNb * ColNb])
      : data(vd) {}

  __host__ __device__ MatN(T fill_value) {
    for (unsigned int i = 0; i < size; i++) {
      set(i, fill_value);
    }
  }
  template <class K, unsigned int R, unsigned int C>
  friend __host__ __device__ std::stringstream &
  operator<<(std::stringstream &out, MatN<K, R, C> m);

  /**\brief Create matrix based on argument matrix*/
  template <unsigned int OutRowNb = RowNb,
            unsigned int OutColNb = ColNb>
  __host__ __device__ static MResult
  from_row_cols(MatN<T, OutRowNb, OutColNb> &out) {
    out = MatN<T, OutRowNb, OutColNb>(static_cast<T>(0));
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "from_row_cols", SUCCESS);
  }
  template <unsigned int OutRowNb = RowNb,
            unsigned int OutColNb = ColNb>
  __host__ __device__ static MResult
  from_row_cols(T v, MatN<T, OutRowNb, OutColNb> &out) {
    MatN<T, OutRowNb, OutColNb> mat(v);
    out = mat;
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "from_row_cols", SUCCESS);
  }
  template <unsigned int OutRowNb = RowNb,
            unsigned int OutColNb = ColNb>
  __host__ __device__ static MResult
  identity(unsigned int nb,
           MatN<T, OutRowNb, OutColNb> &out) {
    MatN<T, OutRowNb, OutColNb> mat;
    auto r = from_row_cols<OutRowNb, OutColNb>(mat);
    if (r.status != SUCCESS)
      return r;
    for (unsigned int i = 0; i < nb; i++) {
      mat.set(i, i, static_cast<T>(1));
    }
    out = mat;
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "identity", SUCCESS);
  }
  __host__ __device__ MResult
  apply(const MatN<T, RowNb, ColNb> &vmat,
        const std::function<T(T, T)> &fn,
        MatN<T, RowNb, ColNb> &out) const {
    for (unsigned int i = 0; i < size; i++) {
      T tout = static_cast<T>(0);
      vmat.get(i, tout);
      T val = fn(data[i], tout);
      auto r = out.set(i, val);
      if (r.status != SUCCESS)
        return r;
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "apply", SUCCESS);
  }
  __host__ __device__ MResult
  apply(const T &v, const std::function<T(T, T)> &fn,
        MatN<T, RowNb, ColNb> &out) const {
    for (unsigned int i = 0; i < size; i++) {
      T val = fn(data[i], v);
      auto r = out.set(i, val);
      if (r.status != SUCCESS)
        return r;
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "apply", SUCCESS);
  }
  // tested
  template <unsigned int OutRowNb = RowNb,
            unsigned int OutColNb = ColNb>
  __host__ __device__ MResult
  fill(T v, MatN<T, OutRowNb, OutColNb> &out) const {
    unsigned int s = 0;
    out.get_size(s);
    for (unsigned int i = 0; i < s; i++) {
      out.set(i, v);
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__, "fill",
                   SUCCESS);
  }
  // tested
  __host__ __device__ MResult
  transpose(MatN<T, ColNb, RowNb> &out) const {

    for (unsigned int i = 0; i < nb_rows; i++) {
      for (unsigned int j = 0; j < nb_cols; j++) {
        T tout = static_cast<T>(0);
        get(i, j, tout);
        out.set(j, i, tout);
      }
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "transpose", SUCCESS);
  }
  // tested
  __host__ __device__ MResult
  col_nb(unsigned int &v) const {
    v = nb_cols;
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "col_nb", SUCCESS);
  }
  // tested
  __host__ __device__ MResult
  row_nb(unsigned int &v) const {
    v = nb_rows;
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "row_nb", SUCCESS);
  }
  // tested
  __host__ __device__ MResult
  get_size(unsigned int &out) const {
    out = size;
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "get_size", SUCCESS);
  }
  __host__ __device__ MResult get(unsigned int row,
                                  unsigned int col,
                                  T &out) const {
    unsigned int index = row * nb_cols + col;
    MResult r = get(index, out);
    if (r.status != SUCCESS)
      return r;
    return MResult(__LINE__, __FILE__, __FUNCTION__, "get",
                   SUCCESS);
  }
  __host__ __device__ MResult get(unsigned int index,
                                  T &out) const {
    if (index >= size)
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "get", INDEX_ERROR);
    out = data[index];
    return MResult(__LINE__, __FILE__, __FUNCTION__, "get",
                   SUCCESS);
  }
  __host__ __device__ MResult
  get(T out[RowNb * ColNb]) const {
    out = data;
    return MResult(__LINE__, __FILE__, __FUNCTION__, "get",
                   SUCCESS);
  }
  __host__ __device__ MResult set(unsigned int row,
                                  unsigned int col, T el) {
    unsigned int index = row * nb_cols + col;
    auto r = set(index, el);
    if (r.status != SUCCESS)
      return r;
    return MResult(__LINE__, __FILE__, __FUNCTION__, "set",
                   SUCCESS);
  }
  __host__ __device__ MResult set(unsigned int index,
                                  T el) {
    if (index >= size)
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "set", INDEX_ERROR);

    data[index] = el;
    return MResult(__LINE__, __FILE__, __FUNCTION__, "set",
                   SUCCESS);
  }
  __host__ __device__ MResult
  column(unsigned int index, T out[RowNb]) const {
    if (index >= ColNb) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "column", INDEX_ERROR);
    }
    for (unsigned int i = 0; i < RowNb; i++) {
      out[i] = data[i * ColNb + index];
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "column", SUCCESS);
  }
  __host__ __device__ MResult
  set_column(unsigned int index, const T idata[RowNb]) {
    if (index >= ColNb) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "set_column", INDEX_ERROR);
    }
    for (unsigned int i = 0; i < RowNb; i++) {
      data[i * ColNb + index] = idata[i];
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "set_column", SUCCESS);
  }
  __host__ __device__ MResult row(unsigned int index,
                                      T out[ColNb]) const {
    if (index >= RowNb) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "row", INDEX_ERROR);
    }
    for (unsigned int i = 0; i < ColNb; i++) {
      out[i] = data[index * ColNb + i];
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "row", SUCCESS);
  }
  __host__ __device__ MResult
  set_row(unsigned int index, const T idata[ColNb]) {
    if (index >= RowNb) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "set_row", INDEX_ERROR);
    }
    for (unsigned int i = 0; i < ColNb; i++) {
      data[index * ColNb + i] = idata[i];
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "set_row", SUCCESS);
  }

  /**Obtain submatrix TODO*/
  __host__ __device__ MResult
  submat(unsigned int row_start, unsigned int col_start,
         MatN<T, RowNb, ColNb> &out) const {
    unsigned int row_size = nb_rows - row_start;
    unsigned int col_size = nb_cols - col_start;
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "submat", NOT_IMPLEMENTED);
  }
  __host__ __device__ MResult
  add(const MatN<T, RowNb, ColNb> &v,
      MatN<T, RowNb, ColNb> &out) const {
    auto fn = [](T matv, T val) { return matv + val; };
    return apply(v, fn, out);
  }
  __host__ __device__ MResult
  add(T v, MatN<T, RowNb, ColNb> &out) const {
    auto fn = [](T matv, T val) { return matv + val; };
    return apply(v, fn, out);
  }
  __host__ __device__ MResult
  subtract(const MatN<T, RowNb, ColNb> &v,
           MatN<T, RowNb, ColNb> &out) const {
    auto fn = [](T matv, T val) { return matv - val; };
    return apply(v, fn, out);
  }
  __host__ __device__ MResult
  subtract(T v, MatN<T, RowNb, ColNb> &out) const {
    auto fn = [](T matv, T val) { return matv - val; };
    return apply(v, fn, out);
  }
  __host__ __device__ MResult
  hadamard_product(const MatN<T, RowNb, ColNb> &v,
                   MatN<T, RowNb, ColNb> &out) const {
    auto fn = [](T matv, T val) { return matv * val; };
    return apply(v, fn, out);
  }
  __host__ __device__ MResult
  hadamard_product(T v, MatN<T, RowNb, ColNb> &out) const {
    auto fn = [](T matv, T val) { return matv * val; };
    return apply(v, fn, out);
  }
  __host__ __device__ MResult
  divide(const MatN<T, RowNb, ColNb> &v,
         MatN<T, RowNb, ColNb> &out) const {
    unsigned int osize = 0;
    v.get_size(osize);
    for (unsigned int i = 0; i < osize; i++) {
      T tout = static_cast<T>(0);
      v.get(i, tout);
      if (tout == static_cast<T>(0)) {
        // zero division risk
        return MResult(__LINE__, __FILE__, __FUNCTION__,
                       "divide", ARG_ERROR);
      }
    }
    auto fn = [](T matv, T val) { return matv / val; };
    return apply(v, fn, out);
  }
  __host__ __device__ MResult
  divide(T v, MatN<T, RowNb, ColNb> &out) const {
    if (v == static_cast<T>(0)) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "divide", ARG_ERROR);
    }
    auto fn = [](T matv, T val) { return matv / val; };
    return apply(v, fn, out);
  }
  /**Declares inner vector product*/
  template <unsigned int N = RowNb>
  __host__ __device__ MResult vdot(const T x[N],
                                   const T y[N],
                                   T &out) const {
    if (N == 0) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "vdot", SIZE_ERROR);
    }

    out = static_cast<T>(0);
    for (unsigned int i = 0; i < N; i++) {
      out += x[i] * y[i];
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__, "vdot",
                   SUCCESS);
  }

  /**Declares inner vector product with scalars*/
  template <unsigned int N = RowNb>
  __host__ __device__ MResult vdot_s(const T x[N],
                                     const T &a,
                                     T out[N]) const {

    if (N == 0) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "vdot_s", SIZE_ERROR);
    }
    for (unsigned int i = 0; i < N; i++) {
      out[i] = x[i] * a;
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "vdot_s", SUCCESS);
  }
  /**Implements saxpy algorithm from Golub, Van Loan 2013,
   * p. 4 alg.1.1.2*/
  template <unsigned int N = RowNb>
  __host__ __device__ MResult saxpy(const T &a,
                                    const T x[N],
                                    T y[N]) const {
    if (N == 0) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "saxpy", SIZE_ERROR);
    }
    for (unsigned int i = 0; i < N; i++) {
      y[i] += x[i] * a; //
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "saxpy", SUCCESS);
  }
  /**
    Implements gaxpy algorithm from Golub, Van Loan 2013, p.
    4 alg.1.1.3

    as specified in p. 6-7
   */
  __host__ __device__ MResult gaxpy(const T x[ColNb],
                                    T y[RowNb]) const {
    for (unsigned int j = 0; j < ColNb; j++) {
      T c_j[RowNb];
      column(j, c_j);
      saxpy<RowNb>(x[j], c_j, y);
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "gaxpy", SUCCESS);
  }
  /**
     Implements outer product update from Golub, Van Loan
     2013, p. 7 as a series of saxpy operations
    */
  template <unsigned int Rn, unsigned int Cn>
  __host__ __device__ MResult
  outer_product(const T x[Rn], const T y[Cn],
                MatN<T, Rn, Cn> &out) const {
    for (unsigned int i = 0; i < Rn; i++) {
      T A_i[Cn];
      out.row(i, A_i);
      saxpy<Cn>(x[i], y, A_i);
      out.set_row(i, A_i);
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "outer_product", SUCCESS);
  }
  template <unsigned int OutColNb = RowNb>
  __host__ __device__ MResult
  multiply(T v, MatN<T, RowNb, OutColNb> &out) const {
    // m x n \cdot  vmat (n x l) = out (m x l)
    // RowNb x ColNb \codt (n x l) = out (OutRowNb x
    // OutColNb)
    MatN<T, ColNb, OutColNb> vmat(v);

    auto r = multiply(vmat, out);
    if (r.status != SUCCESS)
      return r;
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "multiply", SUCCESS);
  }
  /*matrix to matrix multiplication*/
  template <unsigned int OutColNb = RowNb>
  __host__ __device__ MResult
  dot(const MatN<T, ColNb, OutColNb> &v,
      MatN<T, RowNb, OutColNb> &out) const {
    return multiply(v, out);
  }
  /*matrix to scalar multiplication*/
  template <unsigned int OutColNb = RowNb>
  __host__ __device__ MResult
  dot(T v, MatN<T, RowNb, OutColNb> &out) const {
    return multiply(v, out);
  }
  /*matrix to vector multiplication*/
  __host__ __device__ MResult
  dot(const T v[ColNb], MatN<T, RowNb, 1> &out) const {
    MatN<T, ColNb, 1> vmat(v);
    auto r = multiply<1>(vmat, out);
    if (r.status != SUCCESS)
      return r;
    return MResult(__LINE__, __FILE__, __FUNCTION__, "dot",
                   SUCCESS);
  }

  /**
    m x n \cdot  vmat (n x l) = out (m x l)
    RowNb x ColNb \codt (OutRowNb x OutColNb) = out (RowNb x
    OutColNb)

    We are using the kij (row outer product) variant from
    Golub, van Loan 2013, p. 11 alg. 1.1.8 due to
    implementing this algorithm in C++. For fortran etc one
    should use jki since it access matrices by column.  For
    a comparison of algorithms see table 1.1.1 in p. 9

    tested
   */
  template <unsigned int OutColNb = RowNb>
  __host__ __device__ MResult
  multiply(const MatN<T, ColNb, OutColNb> &B,
           MatN<T, RowNb, OutColNb> &out) const {

    // fill out matrix with zero
    out = MatN<T, RowNb, OutColNb>(static_cast<T>(0));
    for (unsigned int k = 0; k < ColNb; k++) {
      // x vector
      T A_k[RowNb];
      column(k, A_k);

      // y vector
      T B_k[OutColNb];
      B.row(k, B_k);

      // compute their outer product
      outer_product<RowNb, OutColNb>(A_k, B_k, out);
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "multiply", SUCCESS);
  }
  /**
    add row
   */
  __host__ __device__ MResult
  add_row(const T r_data[ColNb],
          MatN<T, RowNb + 1, ColNb> &out) const {
    return add_rows<ColNb>(r_data, out);
  }
  /**
    add rows if the incoming data has a size of multiple of
    number of columns
    of this array
  */
  template <unsigned int InRow = ColNb>
  __host__ __device__ MResult add_rows(
      const T r_data[InRow],
      MatN<T, RowNb + (InRow / ColNb), ColNb> &out) const {
    if ((InRow % ColNb) != 0) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "add_rows", SIZE_ERROR);
    }
    // fill output matrix with zeros
    from_row_cols(out);

    // fill with the output matrix with current matrix
    // elements
    unsigned int i = 0;
    unsigned int j = 0;
    for (i = 0; i < RowNb; i++) {
      for (j = 0; j < ColNb; j++) {
        T value = static_cast<T>(0);
        get(i, j, value);
        out.set(i, j, value);
      }
    }

    // fill from r_data the remaining values
    unsigned int nb_of_rows_to_add =
        static_cast<unsigned int>(InRow / ColNb);
    for (i = 0; i <= nb_of_rows_to_add; i++) {
      unsigned int row = RowNb + i;
      for (unsigned int j = 0; j < ColNb; j++) {
        T row_val = r_data[i * ColNb + j];
        out.set(row, j, row_val);
      }
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "add_rows", SUCCESS);
  }
  /**
    add column
  */
  __host__ __device__ MResult
  add_column(const T r_data[RowNb],
             MatN<T, RowNb, ColNb + 1> &out) const {
    return add_columns<RowNb>(r_data, out);
  }
  template <unsigned int InCol = RowNb>
  __host__ __device__ MResult add_columns(
      const T c_data[InCol],
      MatN<T, RowNb, ColNb + (InCol / RowNb)> &out) const {
    if ((InCol % RowNb) != 0) {
      return MResult(__LINE__, __FILE__, __FUNCTION__,
                     "add_columns", SIZE_ERROR);
    }
    // fill output matrix with zeros
    from_row_cols(out);

    // fill with the output matrix with current matrix
    // elements
    unsigned int i = 0;
    unsigned int j = 0;
    for (i = 0; i < RowNb; i++) {
      for (j = 0; j < ColNb; j++) {
        T value = static_cast<T>(0);
        get(i, j, value);
        out.set(i, j, value);
      }
    }
    // fill from c_data the remaining values
    unsigned int nb_of_cols_to_add =
        static_cast<unsigned int>(InCol / RowNb);

    // even if there are zero columns to add the output
    // should be one
    for (i = 0; i < nb_of_cols_to_add; i++) {
      unsigned int col = ColNb + i;
      for (j = 0; j < RowNb; j++) {
        T col_val = c_data[i * RowNb + j];
        out.set(j, col, col_val);
      }
    }
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "add_columns", SUCCESS);
  }
  __host__ __device__ MResult
  to_double_vec(std::vector<std::vector<T>> &ovec) const {
    //
    std::vector<std::vector<T>> out(
        nb_rows, std::vector<T>(nb_cols));
    for (unsigned int i = 0; i < nb_rows; i++) {
      for (unsigned int j = 0; j < nb_cols; j++) {
        get(i, j, out[i][j]);
      }
    }
    ovec = out;
    return MResult(__LINE__, __FILE__, __FUNCTION__,
                   "to_double_vec", SUCCESS);
  }
};

template <typename T, unsigned int R, unsigned int C = R>
__host__ __device__ std::stringstream &
operator<<(std::stringstream &out, MatN<T, R, C> m) {
  constexpr unsigned int arr_size = R * C;
  T arr[arr_size];
  m.get(arr);
  for (unsigned int i = 0; i < arr_size; i++) {
    if (i % C == 0) {
      out << std::endl;
    }
    if (arr[i] >= 0) {
      out << " " << arr[i] << " ";
    } else {
      out << arr[i] << " ";
    }
  }
  return out;
}
} // namespace cudamat
#endif
