/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   self_assign.cu
 *   @coding UTF-8
 *   @brief  Tests the correct handling of self-assignments.
 *   @brief  测试：自赋值验证
 */


#include <iostream>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/matrix_proxy.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/compressed_compressed_matrix.hpp"
#include "cuarma/coordinate_matrix.hpp"
#include "cuarma/ell_matrix.hpp"
#include "cuarma/sliced_ell_matrix.hpp"
#include "cuarma/hyb_matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/ilu.hpp"
#include "cuarma/blas/detail/ilu/common.hpp"
#include "cuarma/io/matrix_market.hpp"
#include "cuarma/tools/random.hpp"


//
// -------------------------------------------------------------
//
template<typename NumericT>
NumericT diff(NumericT const & s1, cuarma::scalar<NumericT> const & s2)
{
  if (std::fabs(s1 - s2) > 0)
    return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
  return 0;
}

template<typename NumericT>
NumericT diff(std::vector<NumericT> const & v1, cuarma::vector<NumericT> const & v2)
{
  std::vector<NumericT> v2_cpu(v2.size());
  cuarma::backend::finish();
  cuarma::copy(v2.begin(), v2.end(), v2_cpu.begin());

  for (std::size_t i=0;i<v1.size(); ++i)
  {
    if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
      v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
    else
      v2_cpu[i] = 0.0;

    if (v2_cpu[i] > 0.0001)
    {
      //std::cout << "Neighbor: "      << i-1 << ": " << v1[i-1] << " vs. " << v2_cpu[i-1] << std::endl;
      std::cout << "Error at entry " << i   << ": " << v1[i]   << " vs. " << v2[i]   << std::endl;
      //std::cout << "Neighbor: "      << i+1 << ": " << v1[i+1] << " vs. " << v2_cpu[i+1] << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  NumericT inf_norm = 0;
  for (std::size_t i=0;i<v2_cpu.size(); ++i)
    inf_norm = std::max<NumericT>(inf_norm, std::fabs(v2_cpu[i]));

  return inf_norm;
}

template<typename NumericT>
NumericT diff(std::vector<std::vector<NumericT> > const & A1, cuarma::matrix<NumericT> const & A2)
{
  std::vector<NumericT> host_values(A2.internal_size());
  for (std::size_t i=0; i<A2.size1(); ++i)
    for (std::size_t j=0; j<A2.size2(); ++j)
      host_values[i*A2.internal_size2() + j] = A1[i][j];

  std::vector<NumericT> device_values(A2.internal_size());
  cuarma::fast_copy(A2, &device_values[0]);
  cuarma::vector<NumericT> arma_device_values(A2.internal_size());  // workaround to avoid code duplication
  cuarma::copy(device_values, arma_device_values);

  return diff(host_values, arma_device_values);
}


template<typename HostContainerT, typename DeviceContainerT, typename NumericT>
void check(HostContainerT const & host_container, DeviceContainerT const & device_container,
           std::string current_stage, NumericT epsilon)
{
  current_stage.resize(25, ' ');
  std::cout << "Testing operation: " << current_stage;
  NumericT rel_error = std::fabs(diff(host_container, device_container));

  if (rel_error > epsilon)
  {
    std::cout << std::endl;
    std::cout << "# Error at operation: " << current_stage << std::endl;
    std::cout << "  diff: " << rel_error << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "PASS" << std::endl;
}


struct op_assign
{
  template<typename LHS, typename RHS>
  static void apply(LHS & lhs, RHS const & rhs) { lhs = rhs; }

  static std::string str() { return "="; }
};

struct op_plus_assign
{
  template<typename LHS, typename RHS>
  static void apply(LHS & lhs, RHS const & rhs) { lhs += rhs; }

  static std::string str() { return "+="; }
};

struct op_minus_assign
{
  template<typename LHS, typename RHS>
  static void apply(LHS & lhs, RHS const & rhs) { lhs -= rhs; }

  static std::string str() { return "-="; }
};


// compute C = A * B on host and device and compare results.
// Note that the reference uses three distinct matrices A, B, C,
// whereas C on the device is the same as either A, B, or both.
template<typename OpT, typename NumericT, typename HostMatrixT, typename DeviceMatrixT>
void test_gemm(NumericT epsilon,
                 HostMatrixT &   host_A,   HostMatrixT &   host_B, HostMatrixT & host_C,
               DeviceMatrixT & device_A, std::string name_A,
               DeviceMatrixT & device_B, std::string name_B,
               DeviceMatrixT & device_C, bool copy_from_A,
               bool trans_first, bool trans_second)
{
  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  for (std::size_t i = 0; i<host_A.size(); ++i)
    for (std::size_t j = 0; j<host_A[i].size(); ++j)
    {
      host_A[i][j] = randomNumber();
      host_B[i][j] = randomNumber();
    }

  cuarma::copy(host_A, device_A);
  cuarma::copy(host_B, device_B);
  if (copy_from_A)
    host_C = host_A;
  else
    host_C = host_B;

  for (std::size_t i = 0; i<host_A.size(); ++i)
    for (std::size_t j = 0; j<host_A[i].size(); ++j)
    {
      NumericT tmp = 0;
      for (std::size_t k = 0; k<host_A[i].size(); ++k)
        tmp +=   (trans_first ? host_A[k][i] : host_A[i][k])
               * (trans_second ? host_B[j][k] : host_B[k][j]);
      OpT::apply(host_C[i][j], tmp);
    }

  if (trans_first && trans_second)
  {
    OpT::apply(device_C, cuarma::blas::prod(trans(device_A), trans(device_B)));
    check(host_C, device_C, std::string("A ") + OpT::str() + std::string(" ") + name_A + std::string("^T*") + name_B + std::string("^T"), epsilon);
  }
  else if (trans_first && !trans_second)
  {
    OpT::apply(device_C, cuarma::blas::prod(trans(device_A), device_B));
    check(host_C, device_C, std::string("A ") + OpT::str() + std::string(" ") + name_A + std::string("^T*") + name_B + std::string(""), epsilon);
  }
  else if (!trans_first && trans_second)
  {
    OpT::apply(device_C, cuarma::blas::prod(device_A, trans(device_B)));
    check(host_C, device_C, std::string("A ") + OpT::str() + std::string(" ") + name_A + std::string("*") + name_B + std::string("^T"), epsilon);
  }
  else
  {
    OpT::apply(device_C, cuarma::blas::prod(device_A, device_B));
    check(host_C, device_C, std::string("A ") + OpT::str() + std::string(" ") + name_A + std::string("*") + name_B + std::string(""), epsilon);
  }
}

// dispatch routine for all combinations of transpositions:
// C = A * B, C = A * B^T, C = A^T * B, C = A^T * B^T
template<typename OpT, typename NumericT, typename HostMatrixT, typename DeviceMatrixT>
void test_gemm(NumericT epsilon,
                 HostMatrixT &   host_A,   HostMatrixT &   host_B, HostMatrixT & host_C,
               DeviceMatrixT & device_A, std::string name_A,
               DeviceMatrixT & device_B, std::string name_B,
               DeviceMatrixT & device_C, bool copy_from_A)
{
  test_gemm<OpT>(epsilon, host_A, host_B, host_C, device_A, name_A, device_B, name_B, device_C, copy_from_A, false, false);
  test_gemm<OpT>(epsilon, host_A, host_B, host_C, device_A, name_A, device_B, name_B, device_C, copy_from_A, false, true);
  test_gemm<OpT>(epsilon, host_A, host_B, host_C, device_A, name_A, device_B, name_B, device_C, copy_from_A, true, false);
  test_gemm<OpT>(epsilon, host_A, host_B, host_C, device_A, name_A, device_B, name_B, device_C, copy_from_A, true, true);
}

// The actual testing routine.
// Sets of vectors and matrices using STL types and uses these for reference calculations.
// cuarma operations are carried out as usual and then compared against the reference.
template<typename NumericT>
int test(NumericT epsilon)
{
  std::size_t N = 142; // should be larger than 128 in order to avoid false negatives due to blocking

  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  //
  // Vector setup and test:
  //

  std::vector<NumericT> std_x(N);
  std::vector<NumericT> std_y(N);
  std::vector<NumericT> std_z(N);

  for (std::size_t i=0; i<std_x.size(); ++i)
    std_x[i] = NumericT(i + 1);
  for (std::size_t i=0; i<std_y.size(); ++i)
    std_y[i] = NumericT(i*i + 1);
  for (std::size_t i=0; i<std_z.size(); ++i)
    std_z[i] = NumericT(2 * i + 1);

  cuarma::vector<NumericT> arma_x;
  cuarma::vector<NumericT> arma_y;
  cuarma::vector<NumericT> arma_z;

  cuarma::copy(std_x, arma_x);
  cuarma::copy(std_y, arma_y);
  cuarma::copy(std_z, arma_z);

  // This shouldn't do anything bad:
  arma_x = arma_x;
  check(std_x, arma_x, "x = x", epsilon);

  // This should work, even though we are dealing with the same buffer:
  std_x[0] = std_x[2]; std_x[1] = std_x[3];
  cuarma::project(arma_x, cuarma::range(0, 2)) = cuarma::project(arma_x, cuarma::range(2, 4));
  check(std_x, arma_x, "x = x (range)", epsilon);

  //
  // Matrix-Vector
  //

  std::vector<std::vector<NumericT> > std_A(N, std::vector<NumericT>(N, NumericT(1)));
  std::vector<std::vector<NumericT> > std_B(N, std::vector<NumericT>(N, NumericT(2)));
  std::vector<std::vector<NumericT> > std_C(N, std::vector<NumericT>(N, NumericT(3)));

  cuarma::matrix<NumericT> arma_A;
  cuarma::matrix<NumericT> arma_B;
  cuarma::matrix<NumericT> arma_C;

  cuarma::copy(std_A, arma_A);
  cuarma::copy(std_B, arma_B);
  cuarma::copy(std_C, arma_C);

  // This shouldn't do anything bad:
  arma_A = arma_A;
  check(std_A, arma_A, "A = A", epsilon);

  // This should work, even though we are dealing with the same buffer:
  std_A[0][0] = std_A[0][2]; std_A[0][1] = std_A[0][3];
  cuarma::project(arma_A, cuarma::range(0, 1), cuarma::range(0, 2)) = cuarma::project(arma_A, cuarma::range(0, 1), cuarma::range(2, 4));
  check(std_A, arma_A, "A = A (range)", epsilon);

  // check x <- A * x;
  for (std::size_t i = 0; i<std_y.size(); ++i)
  {
    NumericT val = 0;
    for (std::size_t j = 0; j<std_x.size(); ++j)
      val += std_A[i][j] * std_x[j];
    std_y[i] = val;
  }
  arma_x = cuarma::blas::prod(arma_A, arma_x);
  check(std_y, arma_x, "x = A*x", epsilon);

  typedef unsigned int     KeyType;
  std::vector< std::map<KeyType, NumericT> > std_Asparse(N);

  for (std::size_t i=0; i<std_Asparse.size(); ++i)
  {
    if (i > 0)
      std_Asparse[i][KeyType(i-1)] = randomNumber();
    std_Asparse[i][KeyType(i)] = NumericT(1) + randomNumber();
    if (i < std_Asparse.size() - 1)
      std_Asparse[i][KeyType(i+1)] = randomNumber();
  }

  // Sparse
  cuarma::compressed_matrix<NumericT> arma_A_csr;
  cuarma::coordinate_matrix<NumericT> arma_A_coo;
  cuarma::ell_matrix<NumericT>        arma_A_ell;
  cuarma::sliced_ell_matrix<NumericT> arma_A_sell;
  cuarma::hyb_matrix<NumericT>        arma_A_hyb;

  cuarma::copy(std_Asparse, arma_A_csr);
  cuarma::copy(std_Asparse, arma_A_coo);
  cuarma::copy(std_Asparse, arma_A_ell);
  cuarma::copy(std_Asparse, arma_A_sell);
  cuarma::copy(std_Asparse, arma_A_hyb);

  for (std::size_t i=0; i<std_Asparse.size(); ++i)
  {
    NumericT val = 0;
    for (typename std::map<unsigned int, NumericT>::const_iterator it = std_Asparse[i].begin(); it != std_Asparse[i].end(); ++it)
      val += it->second * std_x[it->first];
    std_y[i] = val;
  }

  cuarma::copy(std_x, arma_x);
  arma_x = cuarma::blas::prod(arma_A_csr, arma_x);
  check(std_y, arma_x, "x = A*x (sparse, csr)", epsilon);

  cuarma::copy(std_x, arma_x);
  arma_x = cuarma::blas::prod(arma_A_coo, arma_x);
  check(std_y, arma_x, "x = A*x (sparse, coo)", epsilon);

  cuarma::copy(std_x, arma_x);
  arma_x = cuarma::blas::prod(arma_A_ell, arma_x);
  check(std_y, arma_x, "x = A*x (sparse, ell)", epsilon);

  cuarma::copy(std_x, arma_x);
  arma_x = cuarma::blas::prod(arma_A_sell, arma_x);
  check(std_y, arma_x, "x = A*x (sparse, sell)", epsilon);

  cuarma::copy(std_x, arma_x);
  arma_x = cuarma::blas::prod(arma_A_hyb, arma_x);
  check(std_y, arma_x, "x = A*x (sparse, hyb)", epsilon);
  std::cout << std::endl;


  //
  // Matrix-Matrix (dense times dense):
  //
  test_gemm<op_assign>(epsilon, std_A, std_B, std_C, arma_A, "A", arma_B, "B", arma_A, true);
  test_gemm<op_assign>(epsilon, std_B, std_A, std_C, arma_B, "B", arma_A, "A", arma_A, false);
  test_gemm<op_assign>(epsilon, std_A, std_A, std_C, arma_A, "A", arma_A, "A", arma_A, true);
  std::cout << std::endl;

  test_gemm<op_plus_assign>(epsilon, std_A, std_B, std_C, arma_A, "A", arma_B, "B", arma_A, true);
  test_gemm<op_plus_assign>(epsilon, std_B, std_A, std_C, arma_B, "B", arma_A, "A", arma_A, false);
  test_gemm<op_plus_assign>(epsilon, std_A, std_A, std_C, arma_A, "A", arma_A, "A", arma_A, true);
  std::cout << std::endl;

  test_gemm<op_minus_assign>(epsilon, std_A, std_B, std_C, arma_A, "A", arma_B, "B", arma_A, true);
  test_gemm<op_minus_assign>(epsilon, std_B, std_A, std_C, arma_B, "B", arma_A, "A", arma_A, false);
  test_gemm<op_minus_assign>(epsilon, std_A, std_A, std_C, arma_A, "A", arma_A, "A", arma_A, true);
  std::cout << std::endl;



  //
  // Matrix-Matrix (sparse times dense)
  //
  // A = sparse * A
  cuarma::copy(std_A, arma_A);
  for (std::size_t i = 0; i<std_A.size(); ++i)
    for (std::size_t j = 0; j<std_A[i].size(); ++j)
    {
      NumericT tmp = 0;
      for (std::size_t k = 0; k<std_A[i].size(); ++k)
        tmp += std_Asparse[i][KeyType(k)] * std_A[k][j];
      std_C[i][j] = tmp;
    }

  cuarma::copy(std_A, arma_A);
  arma_A = cuarma::blas::prod(arma_A_csr, arma_A);
  check(std_C, arma_A, "A = csr*A", epsilon);

  cuarma::copy(std_A, arma_A);
  arma_A = cuarma::blas::prod(arma_A_coo, arma_A);
  check(std_C, arma_A, "A = coo*A", epsilon);

  cuarma::copy(std_A, arma_A);
  arma_A = cuarma::blas::prod(arma_A_ell, arma_A);
  check(std_C, arma_A, "A = ell*A", epsilon);

  cuarma::copy(std_A, arma_A);
  //arma_A = cuarma::blas::prod(arma_A_sell, arma_A);
  //check(std_C, arma_A, "A = sell*A", epsilon);

  cuarma::copy(std_A, arma_A);
  arma_A = cuarma::blas::prod(arma_A_hyb, arma_A);
  check(std_C, arma_A, "A = hyb*A", epsilon);

  // A = sparse * A^T
  cuarma::copy(std_A, arma_A);
  for (std::size_t i = 0; i<std_A.size(); ++i)
    for (std::size_t j = 0; j<std_A[i].size(); ++j)
    {
      NumericT tmp = 0;
      for (std::size_t k = 0; k<std_A[i].size(); ++k)
        tmp += std_Asparse[i][KeyType(k)] * std_A[j][k];
      std_C[i][j] = tmp;
    }

  cuarma::copy(std_A, arma_A);
  arma_A = cuarma::blas::prod(arma_A_csr, trans(arma_A));
  check(std_C, arma_A, "A = csr*A^T", epsilon);

  cuarma::copy(std_A, arma_A);
  arma_A = cuarma::blas::prod(arma_A_coo, trans(arma_A));
  check(std_C, arma_A, "A = coo*A^T", epsilon);

  cuarma::copy(std_A, arma_A);
  arma_A = cuarma::blas::prod(arma_A_ell, trans(arma_A));
  check(std_C, arma_A, "A = ell*A^T", epsilon);

  cuarma::copy(std_A, arma_A);
  //arma_A = cuarma::blas::prod(arma_A_sell, trans(arma_A));
  //check(std_C, arma_A, "A = sell*A^T", epsilon);

  cuarma::copy(std_A, arma_A);
  arma_A = cuarma::blas::prod(arma_A_hyb, trans(arma_A));
  check(std_C, arma_A, "A = hyb*A^T", epsilon);

  return EXIT_SUCCESS;
}


//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Self-Assignment" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    typedef float NumericT;
    NumericT epsilon = static_cast<NumericT>(1E-4);
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    retval = test<NumericT>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  // Note: No need for double precision check, self-assignments are handled in a numeric-type agnostic manner.

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return retval;
}
