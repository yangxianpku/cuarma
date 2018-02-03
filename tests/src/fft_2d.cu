/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   fft_2d.cu
 *   @coding UTF-8
 *   @brief  Tests the one-dimensional FFT routines.
 *   @brief  ≤‚ ‘£∫∂˛Œ¨FFt≤‚ ‘
 */


#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>

#include "head_data.h"
#include "head_define.h"

#include "cuarma/blas/host_based/fft_operations.hpp"

#ifdef CUARMA_WITH_CUDA
#include "cuarma/blas/cuda/fft_operations.hpp"
#endif
#include "cuarma/blas/fft_operations.hpp"
#include "cuarma/fft.hpp"

typedef float ScalarType;

const ScalarType EPS = ScalarType(0.06f); //use smaller values in double precision

typedef ScalarType (*test_function_ptr)(std::vector<ScalarType>&, std::vector<ScalarType>&, unsigned int, unsigned int, unsigned int);
typedef void (*input_function_ptr)(std::vector<ScalarType>&, std::vector<ScalarType>&, unsigned int&, unsigned int&, unsigned int&, const std::string&);


void set_values_struct(std::vector<ScalarType>& input, std::vector<ScalarType>& output, unsigned int& rows, unsigned int& cols, unsigned int& batch_size, TestData2D& data);

void set_values_struct(std::vector<ScalarType>& input, std::vector<ScalarType>& output,unsigned int& rows, unsigned int& cols, unsigned int& batch_size, TestData2D& data)
{
  unsigned int size = data.col_num * data.batch_num * 2 * data.row_num;
  input.resize(size);
  output.resize(size);
  rows = data.row_num;
  cols = data.col_num;
  batch_size = data.batch_num;
  for (unsigned int i = 0; i < size; i++)
  {
    input[i] = data.input[i];
    output[i] = data.output[i];
  }

}

void read_matrices_pair(std::vector<ScalarType>& input, std::vector<ScalarType>& output,unsigned int& rows, unsigned int& cols, unsigned int& batch_size, const std::string& log_tag);

void read_matrices_pair(std::vector<ScalarType>& input, std::vector<ScalarType>& output,unsigned int& rows, unsigned int& cols, unsigned int& batch_size, const std::string& log_tag)
{
  if (log_tag == "fft:2d::direct::1_arg")
    set_values_struct(input, output, rows, cols, batch_size, direct_2d);
  if (log_tag == "fft:2d::radix2::1_arg")
    set_values_struct(input, output, rows, cols, batch_size, radix2_2d);
  if (log_tag == "fft:2d::direct::big::2_arg")
    set_values_struct(input, output, rows, cols, batch_size, direct_2d_big);
  if (log_tag == "fft::transpose" || log_tag == "fft::transpose_inplace")
      set_values_struct(input, output, rows, cols, batch_size, transposeMatrix);

}

template<typename ScalarType>
ScalarType diff(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref)
{
  ScalarType df = 0.0;
  ScalarType norm_ref = 0;

  for (std::size_t i = 0; i < vec.size(); i++)
  {
    df = df + pow(vec[i] - ref[i], 2);
    norm_ref += ref[i] * ref[i];
  }

  return sqrt(df / norm_ref);
}

template<typename ScalarType>
ScalarType diff_max(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref)
{
  ScalarType df = 0.0;
  ScalarType mx = 0.0;
  ScalarType norm_max = 0;

  for (std::size_t i = 0; i < vec.size(); i++)
  {
    df = std::max<ScalarType>(std::fabs(vec[i] - ref[i]), df);
    mx = std::max<ScalarType>(std::fabs(vec[i]), mx);

    if (mx > 0)
    {
      if (norm_max < df / mx)
        norm_max = df / mx;
    }
  }

  return norm_max;
}


void copy_vector_to_matrix(cuarma::matrix<ScalarType> & input, std::vector<ScalarType> & in,unsigned int row, unsigned int col);

void copy_vector_to_matrix(cuarma::matrix<ScalarType> & input, std::vector<ScalarType> & in,unsigned int row, unsigned int col)
{
  std::vector<std::vector<ScalarType> > my_matrix(row, std::vector<ScalarType>(col * 2));
  for (unsigned int i = 0; i < row; i++)
    for (unsigned int j = 0; j < col * 2; j++)
      my_matrix[i][j] = in[i * col * 2 + j];
  cuarma::copy(my_matrix, input);

}

void copy_matrix_to_vector(cuarma::matrix<ScalarType> & input, std::vector<ScalarType> & in,unsigned int row, unsigned int col);

void copy_matrix_to_vector(cuarma::matrix<ScalarType> & input, std::vector<ScalarType> & in,unsigned int row, unsigned int col)
{
  std::vector<std::vector<ScalarType> > my_matrix(row, std::vector<ScalarType>(col * 2));
  cuarma::copy(input, my_matrix);
  for (unsigned int i = 0; i < row; i++)
    for (unsigned int j = 0; j < col * 2; j++)
      in[i * col * 2 + j] = my_matrix[i][j];
}

ScalarType fft_2d_1arg(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,unsigned int col, unsigned int /*batch_size*/);

ScalarType fft_2d_1arg(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,unsigned int col, unsigned int /*batch_size*/)
{
  cuarma::matrix<ScalarType> input(row, 2 * col);

  std::vector<ScalarType> res(in.size());

  copy_vector_to_matrix(input, in, row, col);

  cuarma::inplace_fft(input);
  //std::cout << input << "\n";
  cuarma::backend::finish();

  copy_matrix_to_vector(input, res, row, col);

  return diff_max(res, out);
}

ScalarType transpose_inplace(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,unsigned int col, unsigned int /*batch_size*/);

ScalarType transpose_inplace(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,unsigned int col, unsigned int /*batch_size*/)
{
  cuarma::matrix<ScalarType> input(row, 2 * col);

  std::vector<ScalarType> res(in.size());

  copy_vector_to_matrix(input, in, row, col);

  cuarma::blas::transpose(input);

  cuarma::backend::finish();

  copy_matrix_to_vector(input, res, row, col);

  return diff_max(res, out);
}

ScalarType transpose(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,unsigned int col, unsigned int /*batch_size*/);

ScalarType transpose(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,unsigned int col, unsigned int /*batch_size*/)
{
  cuarma::matrix<ScalarType> input(row, 2 * col);
  cuarma::matrix<ScalarType> output(row, 2 * col);


  std::vector<ScalarType> res(in.size());

  copy_vector_to_matrix(input, in, row, col);

  cuarma::blas::transpose(input,output);

  cuarma::backend::finish();

  copy_matrix_to_vector(output, res, row, col);

  return diff_max(res, out);
}


ScalarType fft_2d_2arg(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,unsigned int col, unsigned int /*batch_size*/);

ScalarType fft_2d_2arg(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int row,unsigned int col, unsigned int /*batch_size*/)
{
  cuarma::matrix<ScalarType> input(row, 2 * col);
  cuarma::matrix<ScalarType> output(row, 2 * col);

  std::vector<ScalarType> res(in.size());

  copy_vector_to_matrix(input, in, row, col);

  //std::cout << input << "\n";
  cuarma::fft(input, output);
  //std::cout << input << "\n";
  cuarma::backend::finish();

  copy_matrix_to_vector(output, res, row, col);

  return diff_max(res, out);
}

int test_correctness(const std::string& log_tag, input_function_ptr input_function,test_function_ptr func);

int test_correctness(const std::string& log_tag, input_function_ptr input_function,test_function_ptr func)
{

  std::vector<ScalarType> input;
  std::vector<ScalarType> output;

  std::cout << std::endl;
  std::cout << "*****************" << log_tag << "***************************\n";

  unsigned int batch_size;
  unsigned int rows_num, cols_num;

  input_function(input, output, rows_num, cols_num, batch_size, log_tag);
  ScalarType df = func(input, output, rows_num, cols_num, batch_size);
  printf("%7s ROWS=%6d COLS=%6d; BATCH=%3d; DIFF=%3.15f;\n", ((fabs(df) < EPS) ? "[Ok]" : "[Fail]"),
      rows_num, cols_num, batch_size, df);
  std::cout << std::endl;

  if (df > EPS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}

int main()
{
  std::cout << "*" << std::endl;
  std::cout << "* cuarma test: FFT" << std::endl;
  std::cout << "*" << std::endl;

  //2D FFT tests
  if (test_correctness("fft:2d::radix2::1_arg", read_matrices_pair, &fft_2d_1arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft:2d::direct::1_arg", read_matrices_pair, &fft_2d_1arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft:2d::direct::big::2_arg", read_matrices_pair,
      &fft_2d_2arg) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::transpose_inplace", read_matrices_pair, &transpose_inplace) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (test_correctness("fft::transpose", read_matrices_pair, &transpose) == EXIT_FAILURE)
      return EXIT_FAILURE;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}