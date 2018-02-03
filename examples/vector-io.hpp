#pragma  once

/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   vector-io.hpp
 *   @coding UTF-8
 *   @brief  Vector，Matrix  IO
 *   @brief  测试：向量、矩阵 IO测试
 */
#include <string>
#include <iostream>
#include <fstream>

#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/traits/size.hpp"


template<typename MatrixType, typename ScalarType>
void insert(MatrixType & matrix, long row, long col, ScalarType value)
{
  matrix(row, col) = value;
}

template<typename MatrixType>
class my_inserter
{
  public:
    my_inserter(MatrixType & mat) : mat_(mat) {}

    void apply(long row, long col, double value)
    {
      insert(mat_, row, col, value);
    }

  private:
    MatrixType & mat_;
};


template<typename VectorType>
void resize_vector(VectorType & vec, unsigned int size)
{
  vec.resize(size);
}

template<typename VectorType>
bool readVectorFromFile(const std::string & filename, VectorType & vec)
{
  typedef typename cuarma::result_of::value_type<VectorType>::type    ScalarType;

  std::ifstream file(filename.c_str());

  if (!file) return false;

  unsigned int size;
  file >> size;

  resize_vector(vec, size);

  for (unsigned int i = 0; i < size; ++i)
  {
    ScalarType element;
    file >> element;
    vec[i] = element;
  }

  return true;
}


template<class MatrixType>
bool readMatrixFromFile(const std::string & filename, MatrixType & matrix)
{
  typedef typename cuarma::result_of::value_type<MatrixType>::type    ScalarType;

  std::cout << "Reading matrix..." << std::endl;

  std::ifstream file(filename.c_str());

  if (!file) return false;

  std::string id;
  file >> id;
  if (id != "Matrix") return false;


  unsigned int num_rows, num_columns;
  file >> num_rows >> num_columns;
  if (num_rows != num_columns) return false;

  cuarma::traits::resize(matrix, num_rows, num_rows);

  my_inserter<MatrixType> ins(matrix);
  for (unsigned int row = 0; row < num_rows; ++row)
  {
    int num_entries;
    file >> num_entries;
    for (int j = 0; j < num_entries; ++j)
    {
      unsigned int column;
      ScalarType element;
      file >> column >> element;

      ins.apply(row, column, element);
    }
  }

  return true;
}