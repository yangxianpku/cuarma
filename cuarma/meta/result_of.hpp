#pragma once

/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/** @file cuarma/meta/result_of.hpp
 *  @encoding:UTF-8 文档编码
    @brief A collection of compile time type deductions
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"


#ifdef CUARMA_WITH_UBLAS
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#endif

#include <vector>
#include <map>

namespace cuarma
{
namespace result_of
{
//
// Retrieve alignment from vector
//
/** @brief Retrieves the alignment from a vector. Deprecated - will be replaced by a pure runtime facility in the future. */
template<typename T>
struct alignment
{
  typedef typename T::ERROR_ARGUMENT_PROVIDED_IS_NOT_A_VECTOR_OR_A_MATRIX   error_type;
  enum { value = 1 };
};

/** \cond */
template<typename T>
struct alignment<const T>
{
  enum { value = alignment<T>::value };
};

template<typename NumericT, unsigned int AlignmentV>
struct alignment< vector<NumericT, AlignmentV> >
{
  enum { value = AlignmentV };
};

template<typename T>
struct alignment< vector_range<T> >
{
  enum { value = alignment<T>::value };
};

template<typename T>
struct alignment< vector_slice<T> >
{
  enum { value = alignment<T>::value };
};

// support for a*x with scalar a and vector x
template<typename LHS, typename RHS, typename OP>
struct alignment< vector_expression<LHS, RHS, OP> >
{
  enum { value = alignment<LHS>::value };
};


// Matrices
template<typename NumericT, typename F, unsigned int AlignmentV>
struct alignment< matrix<NumericT, F, AlignmentV> >
{
  enum { value = AlignmentV };
};

template<typename T>
struct alignment< matrix_range<T> >
{
  enum { value = alignment<T>::value };
};

template<typename T>
struct alignment< matrix_slice<T> >
{
  enum { value = alignment<T>::value };
};

template<typename LHS, typename RHS>
struct alignment< matrix_expression<LHS, RHS, op_trans> >
{
  enum { value = alignment<LHS>::value };
};
/** \endcond */

//
// Retrieve size_type
//
/** @brief Generic meta-function for retrieving the size_type associated with type T */
template<typename T>
struct size_type
{
  typedef typename T::size_type   type;
};

/** \cond */
template<typename T, typename SizeType>
struct size_type< vector_base<T, SizeType> >
{
  typedef SizeType   type;
};

//
// Retrieve difference_type
//
/** @brief Generic meta-function for retrieving the difference_type associated with type T */
template<typename T>
struct difference_type
{
  typedef typename T::difference_type   type;
};




/** \endcond */

//
// Retrieve value_type:
//
/** @brief Generic helper function for retrieving the value_type associated with type T */
template<typename T>
struct value_type
{
  typedef typename T::value_type    type;
};

/** \cond */



/** \endcond */


//
// Retrieve cpu value_type:
//
/** @brief Helper meta function for retrieving the main RAM-based value type. Particularly important to obtain T from cuarma::scalar<T> in a generic way. */
template<typename T>
struct cpu_value_type
{
  typedef typename T::ERROR_CANNOT_DEDUCE_CPU_SCALAR_TYPE_FOR_T    type;
};

/** \cond */
template<typename T>
struct cpu_value_type<const T>
{
  typedef typename cpu_value_type<T>::type    type;
};

template<>
struct cpu_value_type<char>
{
  typedef char    type;
};

template<>
struct cpu_value_type<unsigned char>
{
  typedef unsigned char    type;
};

template<>
struct cpu_value_type<short>
{
  typedef short    type;
};

template<>
struct cpu_value_type<unsigned short>
{
  typedef unsigned short    type;
};

template<>
struct cpu_value_type<int>
{
  typedef int    type;
};

template<>
struct cpu_value_type<unsigned int>
{
  typedef unsigned int    type;
};

template<>
struct cpu_value_type<long>
{
  typedef int    type;
};

template<>
struct cpu_value_type<unsigned long>
{
  typedef unsigned long    type;
};


template<>
struct cpu_value_type<float>
{
  typedef float    type;
};

template<>
struct cpu_value_type<double>
{
  typedef double    type;
};

template<typename T>
struct cpu_value_type<cuarma::scalar<T> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<cuarma::vector_base<T> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<cuarma::implicit_vector_base<T> >
{
  typedef T    type;
};


template<typename T, unsigned int AlignmentV>
struct cpu_value_type<cuarma::vector<T, AlignmentV> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<cuarma::vector_range<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T>
struct cpu_value_type<cuarma::vector_slice<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T1, typename T2, typename OP>
struct cpu_value_type<cuarma::vector_expression<const T1, const T2, OP> >
{
  typedef typename cpu_value_type<T1>::type    type;
};

template<typename T1, typename T2, typename OP>
struct cpu_value_type<const cuarma::vector_expression<const T1, const T2, OP> >
{
  typedef typename cpu_value_type<T1>::type    type;
};


template<typename T>
struct cpu_value_type<cuarma::matrix_base<T> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<cuarma::implicit_matrix_base<T> >
{
  typedef T    type;
};


template<typename T, typename F, unsigned int AlignmentV>
struct cpu_value_type<cuarma::matrix<T, F, AlignmentV> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<cuarma::matrix_range<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T>
struct cpu_value_type<cuarma::matrix_slice<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<cuarma::compressed_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T>
struct cpu_value_type<cuarma::compressed_compressed_matrix<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<cuarma::coordinate_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<cuarma::ell_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, typename IndexT>
struct cpu_value_type<cuarma::sliced_ell_matrix<T, IndexT> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<cuarma::hyb_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T1, typename T2, typename OP>
struct cpu_value_type<cuarma::matrix_expression<T1, T2, OP> >
{
  typedef typename cpu_value_type<T1>::type    type;
};


//
// Deduce compatible vector type for a matrix type
//

template<typename T>
struct vector_for_matrix
{
  typedef typename T::ERROR_CANNOT_DEDUCE_VECTOR_FOR_MATRIX_TYPE   type;
};

//cuarma
template<typename T, typename F, unsigned int A>
struct vector_for_matrix< cuarma::matrix<T, F, A> >
{
  typedef cuarma::vector<T,A>   type;
};

template<typename T, unsigned int A>
struct vector_for_matrix< cuarma::compressed_matrix<T, A> >
{
  typedef cuarma::vector<T,A>   type;
};

template<typename T, unsigned int A>
struct vector_for_matrix< cuarma::coordinate_matrix<T, A> >
{
  typedef cuarma::vector<T,A>   type;
};

#ifdef CUARMA_WITH_UBLAS
//Boost:
template<typename T, typename F, typename A>
struct vector_for_matrix< boost::numeric::ublas::matrix<T, F, A> >
{
  typedef boost::numeric::ublas::vector<T>   type;
};

template<typename T, typename U, arma_size_t A, typename B, typename C>
struct vector_for_matrix< boost::numeric::ublas::compressed_matrix<T, U, A, B, C> >
{
  typedef boost::numeric::ublas::vector<T>   type;
};

template<typename T, typename U, arma_size_t A, typename B, typename C>
struct vector_for_matrix< boost::numeric::ublas::coordinate_matrix<T, U, A, B, C> >
{
  typedef boost::numeric::ublas::vector<T>   type;
};
#endif

template<typename T>
struct reference_if_nonscalar
{
  typedef T &    type;
};

#define CUARMA_REFERENCE_IF_NONSCALAR_INT(TNAME) \
template<> struct reference_if_nonscalar<TNAME>                { typedef                TNAME  type; }; \
template<> struct reference_if_nonscalar<const TNAME>          { typedef          const TNAME  type; }; \
template<> struct reference_if_nonscalar<unsigned TNAME>       { typedef       unsigned TNAME  type; }; \
template<> struct reference_if_nonscalar<const unsigned TNAME> { typedef const unsigned TNAME  type; };

  CUARMA_REFERENCE_IF_NONSCALAR_INT(char)
  CUARMA_REFERENCE_IF_NONSCALAR_INT(short)
  CUARMA_REFERENCE_IF_NONSCALAR_INT(int)
  CUARMA_REFERENCE_IF_NONSCALAR_INT(long)

#undef CUARMA_REFERENCE_IF_NONSCALAR_INT

template<>
struct reference_if_nonscalar<float>
{
  typedef float    type;
};

template<>
struct reference_if_nonscalar<const float>
{
  typedef const float    type;
};

template<>
struct reference_if_nonscalar<double>
{
  typedef double    type;
};

template<>
struct reference_if_nonscalar<const double>
{
  typedef const double    type;
};

/** \endcond */

} //namespace result_of
} //namespace cuarma