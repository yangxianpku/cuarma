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

/** @file predicate.hpp
 *  @encoding:UTF-8 文档编码
    @brief All the predicates used within cuarma. Checks for expressions to be vectors, etc.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"

namespace cuarma
{

//
// is_cpu_scalar: checks for float or double
//
//template<typename T>
//struct is_cpu_scalar
//{
//  enum { value = false };
//};

/** \cond */
template<> struct is_cpu_scalar<char>           { enum { value = true }; };
template<> struct is_cpu_scalar<unsigned char>  { enum { value = true }; };
template<> struct is_cpu_scalar<short>          { enum { value = true }; };
template<> struct is_cpu_scalar<unsigned short> { enum { value = true }; };
template<> struct is_cpu_scalar<int>            { enum { value = true }; };
template<> struct is_cpu_scalar<unsigned int>   { enum { value = true }; };
template<> struct is_cpu_scalar<long>           { enum { value = true }; };
template<> struct is_cpu_scalar<unsigned long>  { enum { value = true }; };
template<> struct is_cpu_scalar<float>          { enum { value = true }; };
template<> struct is_cpu_scalar<double>         { enum { value = true }; };
/** \endcond */


//
// is_scalar: checks for cuarma::scalar
//
//template<typename T>
//struct is_scalar
//{
//  enum { value = false };
//};

/** \cond */
template<typename T>
struct is_scalar<cuarma::scalar<T> >
{
  enum { value = true };
};
/** \endcond */

//
// is_flip_sign_scalar: checks for cuarma::scalar modified with unary operator-
//
//template<typename T>
//struct is_flip_sign_scalar
//{
//  enum { value = false };
//};

/** \cond */
template<typename T>
struct is_flip_sign_scalar<cuarma::scalar_expression< const scalar<T>,
    const scalar<T>,
    op_flip_sign> >
{
  enum { value = true };
};
/** \endcond */

//
// is_any_scalar: checks for either CPU and GPU scalars, i.e. is_cpu_scalar<>::value || is_scalar<>::value
//
//template<typename T>
//struct is_any_scalar
//{
//  enum { value = (is_scalar<T>::value || is_cpu_scalar<T>::value || is_flip_sign_scalar<T>::value )};
//};

//

/** \cond */
#define CUARMA_MAKE_ANY_VECTOR_TRUE(type) template<> struct is_any_vector< type > { enum { value = 1 }; };
#define CUARMA_MAKE_FOR_ALL_NumericT(type) \
  CUARMA_MAKE_ANY_VECTOR_TRUE(type<float>)\
  CUARMA_MAKE_ANY_VECTOR_TRUE(type<double>)

  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::vector)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::vector_range)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::vector_slice)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::unit_vector)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::zero_vector)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::one_vector)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::scalar_vector)

#undef CUARMA_MAKE_FOR_ALL_NumericT
#undef CUARMA_MAKE_ANY_VECTOR_TRUE
  /** \endcond */


  /** \cond */
#define CUARMA_MAKE_ANY_MATRIX_TRUE(TYPE)\
template<> struct is_any_dense_matrix< TYPE > { enum { value = 1 }; };

#define CUARMA_MAKE_FOR_ALL_NumericT(TYPE) \
  CUARMA_MAKE_ANY_MATRIX_TRUE(TYPE<float>)\
  CUARMA_MAKE_ANY_MATRIX_TRUE(TYPE<double>)

#define CUARMA_COMMA ,
#define CUARMA_MAKE_FOR_ALL_NumericT_LAYOUT(TYPE) \
  CUARMA_MAKE_ANY_MATRIX_TRUE(TYPE<float CUARMA_COMMA cuarma::row_major>)\
  CUARMA_MAKE_ANY_MATRIX_TRUE(TYPE<double CUARMA_COMMA cuarma::row_major>)\
  CUARMA_MAKE_ANY_MATRIX_TRUE(TYPE<float CUARMA_COMMA cuarma::column_major>)\
  CUARMA_MAKE_ANY_MATRIX_TRUE(TYPE<double CUARMA_COMMA cuarma::column_major>)

  CUARMA_MAKE_FOR_ALL_NumericT_LAYOUT(cuarma::matrix)
  //    CUARMA_MAKE_FOR_ALL_NumericT_LAYOUT(cuarma::matrix_range)
  //    CUARMA_MAKE_FOR_ALL_NumericT_LAYOUT(cuarma::matrix_slice)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::identity_matrix)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::zero_matrix)
  CUARMA_MAKE_FOR_ALL_NumericT(cuarma::scalar_matrix)

#undef CUARMA_MAKE_FOR_ALL_NumericT_LAYOUT
#undef CUARMA_MAKE_FOR_ALL_NumericT
#undef CUARMA_MAKE_ANY_MATRIX_TRUE
#undef CUARMA_COMMA
/** \endcond */

//
// is_row_major
//
//template<typename T>
//struct is_row_major
//{
//  enum { value = false };
//};

/** \cond */
template<typename ScalarType>
struct is_row_major<cuarma::matrix<ScalarType, cuarma::row_major> >
{
  enum { value = true };
};

template<>
struct is_row_major< cuarma::row_major >
{
  enum { value = true };
};

template<typename T>
struct is_row_major<cuarma::matrix_expression<T, T, cuarma::op_trans> >
{
  enum { value = is_row_major<T>::value };
};
/** \endcond */

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_compressed_matrix<cuarma::compressed_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_coordinate_matrix
//

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_coordinate_matrix<cuarma::coordinate_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_ell_matrix
//
/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_ell_matrix<cuarma::ell_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_sliced_ell_matrix
//
/** \cond */
template<typename ScalarType, typename IndexT>
struct is_sliced_ell_matrix<cuarma::sliced_ell_matrix<ScalarType, IndexT> >
{
  enum { value = true };
};
/** \endcond */

//
// is_hyb_matrix
//
/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_hyb_matrix<cuarma::hyb_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */


//
// is_any_sparse_matrix
//
//template<typename T>
//struct is_any_sparse_matrix
//{
//  enum { value = false };
//};

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_any_sparse_matrix<cuarma::compressed_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType>
struct is_any_sparse_matrix<cuarma::compressed_compressed_matrix<ScalarType> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_any_sparse_matrix<cuarma::coordinate_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_any_sparse_matrix<cuarma::ell_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType, typename IndexT>
struct is_any_sparse_matrix<cuarma::sliced_ell_matrix<ScalarType, IndexT> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_any_sparse_matrix<cuarma::hyb_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename T>
struct is_any_sparse_matrix<const T>
{
  enum { value = is_any_sparse_matrix<T>::value };
};

/** \endcond */

//////////////// Part 2: Operator predicates ////////////////////

//
// is_addition
//
/** @brief Helper metafunction for checking whether the provided type is cuarma::op_add (for addition) */
template<typename T>
struct is_addition
{
  enum { value = false };
};

/** \cond */
template<>
struct is_addition<cuarma::op_add>
{
  enum { value = true };
};
/** \endcond */

//
// is_subtraction
//
/** @brief Helper metafunction for checking whether the provided type is cuarma::op_sub (for subtraction) */
template<typename T>
struct is_subtraction
{
  enum { value = false };
};

/** \cond */
template<>
struct is_subtraction<cuarma::op_sub>
{
  enum { value = true };
};
/** \endcond */

//
// is_product
//
/** @brief Helper metafunction for checking whether the provided type is cuarma::op_prod (for products/multiplication) */
template<typename T>
struct is_product
{
  enum { value = false };
};

/** \cond */
template<>
struct is_product<cuarma::op_prod>
{
  enum { value = true };
};

template<>
struct is_product<cuarma::op_mult>
{
  enum { value = true };
};

template<>
struct is_product<cuarma::op_element_binary<op_prod> >
{
  enum { value = true };
};
/** \endcond */

//
// is_division
//
/** @brief Helper metafunction for checking whether the provided type is cuarma::op_div (for division) */
template<typename T>
struct is_division
{
  enum { value = false };
};

/** \cond */
template<>
struct is_division<cuarma::op_div>
{
  enum { value = true };
};

template<>
struct is_division<cuarma::op_element_binary<op_div> >
{
  enum { value = true };
};
/** \endcond */

// is_primitive_type
//

/** @brief Helper class for checking whether a type is a primitive type. */
template<class T>
struct is_primitive_type{ enum {value = false}; };

/** \cond */
template<> struct is_primitive_type<float>         { enum { value = true }; };
template<> struct is_primitive_type<double>        { enum { value = true }; };
template<> struct is_primitive_type<unsigned int>  { enum { value = true }; };
template<> struct is_primitive_type<int>           { enum { value = true }; };
template<> struct is_primitive_type<unsigned char> { enum { value = true }; };
template<> struct is_primitive_type<char>          { enum { value = true }; };
template<> struct is_primitive_type<unsigned long> { enum { value = true }; };
template<> struct is_primitive_type<long>          { enum { value = true }; };
template<> struct is_primitive_type<unsigned short>{ enum { value = true }; };
template<> struct is_primitive_type<short>         { enum { value = true }; };
/** \endcond */



} //namespace cuarma