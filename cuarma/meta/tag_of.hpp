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


/** @file tag_of.hpp
    @brief Dispatch facility for distinguishing between ublas, STL and cuarma types
*/

#include <vector>
#include <map>

#include "cuarma/forwards.h"

#ifdef CUARMA_WITH_UBLAS
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#endif

namespace cuarma
{

// ----------------------------------------------------
// TAGS
//
/** @brief A tag class for identifying 'unknown' types. */
struct tag_none     {};

/** @brief A tag class for identifying types from uBLAS. */
struct tag_ublas    {};

/** @brief A tag class for identifying types from the C++ STL. */
struct tag_stl      {};

/** @brief A tag class for identifying types from cuarma. */
struct tag_cuarma {};

namespace traits
{
  // ----------------------------------------------------
  // GENERIC BASE
  //
  /** @brief Generic base for wrapping other linear algebra packages
  *
  *  Maps types to tags, e.g. cuarma::vector to tag_cuarma, ublas::vector to tag_ublas
  *  if the matrix type is unknown, tag_none is returned
  *
  *  This is an internal function only, there is no need for a library user of cuarma to care about it any further
  *
  * @tparam T   The type to be inspected
  */
  template< typename T, typename Active = void >
  struct tag_of;

  /** \cond */
  template< typename Sequence, typename Active >
  struct tag_of
  {
    typedef cuarma::tag_none  type;
  };


#ifdef CUARMA_WITH_UBLAS
  // ----------------------------------------------------
  // UBLAS
  //
  template< typename T >
  struct tag_of< boost::numeric::ublas::vector<T> >
  {
    typedef cuarma::tag_ublas  type;
  };

  template< typename T >
  struct tag_of< boost::numeric::ublas::matrix<T> >
  {
    typedef cuarma::tag_ublas  type;
  };

  template< typename T1, typename T2 >
  struct tag_of< boost::numeric::ublas::matrix_unary2<T1,T2> >
  {
    typedef cuarma::tag_ublas  type;
  };

  template< typename T1, typename T2 >
  struct tag_of< boost::numeric::ublas::compressed_matrix<T1,T2> >
  {
    typedef cuarma::tag_ublas  type;
  };

#endif

  // ----------------------------------------------------
  // STL types
  //

  //vector
  template< typename T, typename A >
  struct tag_of< std::vector<T, A> >
  {
    typedef cuarma::tag_stl  type;
  };

  //dense matrix
  template< typename T, typename A >
  struct tag_of< std::vector<std::vector<T, A>, A> >
  {
    typedef cuarma::tag_stl  type;
  };

  //sparse matrix (vector of maps)
  template< typename KEY, typename DATA, typename COMPARE, typename AMAP, typename AVEC>
  struct tag_of< std::vector<std::map<KEY, DATA, COMPARE, AMAP>, AVEC> >
  {
    typedef cuarma::tag_stl  type;
  };


  // ----------------------------------------------------
  // CUARMA
  //
  template< typename T, unsigned int alignment >
  struct tag_of< cuarma::vector<T, alignment> >
  {
    typedef cuarma::tag_cuarma  type;
  };

  template< typename T, typename F, unsigned int alignment >
  struct tag_of< cuarma::matrix<T, F, alignment> >
  {
    typedef cuarma::tag_cuarma  type;
  };

  template< typename T1, typename T2, typename OP >
  struct tag_of< cuarma::matrix_expression<T1,T2,OP> >
  {
    typedef cuarma::tag_cuarma  type;
  };

  template< typename T >
  struct tag_of< cuarma::matrix_range<T> >
  {
    typedef cuarma::tag_cuarma  type;

  };

  template< typename T, unsigned int I>
  struct tag_of< cuarma::compressed_matrix<T,I> >
  {
    typedef cuarma::tag_cuarma  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< cuarma::coordinate_matrix<T,I> >
  {
    typedef cuarma::tag_cuarma  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< cuarma::ell_matrix<T,I> >
  {
    typedef cuarma::tag_cuarma  type;
  };

  template< typename T, typename I>
  struct tag_of< cuarma::sliced_ell_matrix<T,I> >
  {
    typedef cuarma::tag_cuarma  type;
  };


  template< typename T, unsigned int I>
  struct tag_of< cuarma::hyb_matrix<T,I> >
  {
    typedef cuarma::tag_cuarma  type;
  };

  // ----------------------------------------------------
} // end namespace traits


/** @brief Meta function which checks whether a tag is tag_ublas
*
*  This is an internal function only, there is no need for a library user of cuarma to care about it any further
*/
template<typename Tag>
struct is_ublas
{
  enum { value = false };
};

/** \cond */
template<>
struct is_ublas< cuarma::tag_ublas >
{
  enum { value = true };
};
/** \endcond */

/** @brief Meta function which checks whether a tag is tag_ublas
*
*  This is an internal function only, there is no need for a library user of cuarma to care about it any further
*/
template<typename Tag>
struct is_stl
{
  enum { value = false };
};

/** \cond */
template<>
struct is_stl< cuarma::tag_stl >
{
  enum { value = true };
};
/** \endcond */


/** @brief Meta function which checks whether a tag is tag_cuarma
*
*  This is an internal function only, there is no need for a library user of cuarma to care about it any further
*/
template<typename Tag>
struct is_cuarma
{
  enum { value = false };
};

/** \cond */
template<>
struct is_cuarma< cuarma::tag_cuarma >
{
  enum { value = true };
};
/** \endcond */

} // end namespace cuarma