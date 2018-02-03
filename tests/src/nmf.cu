/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   nmf.cu
 *   @coding UTF-8
 *   @brief  Tests the nonnegative matrix factorization.
 *   @brief  测试：非负矩阵分解
 */

#include <ctime>
#include <cmath>

#include "head_define.h"

#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/nmf.hpp"

typedef float ScalarType;

const ScalarType EPS = ScalarType(0.03);

template<typename MATRIX>
float matrix_compare(MATRIX & res, cuarma::matrix_base<ScalarType>& ref)
{
  float diff = 0.0;
  float mx = 0.0;

  for (std::size_t i = 0; i < ref.size1(); i++)
  {
    for (std::size_t j = 0; j < ref.size2(); ++j)
    {
      diff = std::max(diff, std::abs(res(i, j) - ref(i, j)));
      ScalarType valRes = (ScalarType) res(i, j);
      mx = std::max(mx, valRes);
    }
  }
  return diff / mx;
}

void fill_random(cuarma::matrix_base<ScalarType>& v);

void fill_random(cuarma::matrix_base<ScalarType>& v)
{
  for (std::size_t i = 0; i < v.size1(); i++)
  {
    for (std::size_t j = 0; j < v.size2(); ++j)
      v(i, j) = static_cast<ScalarType>(rand()) / ScalarType(RAND_MAX);
  }
}

void test_nmf(std::size_t m, std::size_t k, std::size_t n);

void test_nmf(std::size_t m, std::size_t k, std::size_t n)
{
  cuarma::matrix<ScalarType> v_ref(m, n);
  cuarma::matrix<ScalarType> w_ref(m, k);
  cuarma::matrix<ScalarType> h_ref(k, n);

  fill_random(w_ref);
  fill_random(h_ref);

  v_ref = cuarma::blas::prod(w_ref, h_ref);  //reference result

  cuarma::matrix<ScalarType> w_nmf(m, k);
  cuarma::matrix<ScalarType> h_nmf(k, n);

  fill_random(w_nmf);
  fill_random(h_nmf);

  cuarma::blas::nmf_config conf;
  conf.print_relative_error(true);
  conf.max_iterations(3000); //3000 iterations are enough for the test

  cuarma::blas::nmf(v_ref, w_nmf, h_nmf, conf);

  cuarma::matrix<ScalarType> v_nmf = cuarma::blas::prod(w_nmf, h_nmf);

  float diff = matrix_compare(v_ref, v_nmf);
  bool diff_ok = fabs(diff) < EPS;

  long iterations = static_cast<long>(conf.iters());
  printf("%6s [%lux%lux%lu] diff = %.6f (%ld iterations)\n", diff_ok ? "[[OK]]" : "[FAIL]", m, k, n,
      diff, iterations);

  if (!diff_ok)
    exit(EXIT_FAILURE);
}

int main()
{
  //srand(time(NULL));  //let's use deterministic tests, so keep the default srand() initialization
  std::cout << std::endl;
  std::cout << "------- Test NMF --------" << std::endl;
  std::cout << std::endl;

  test_nmf(3, 3, 3);
  test_nmf(5, 4, 5);
  test_nmf(16, 7, 12);
  test_nmf(140, 86, 113);

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
