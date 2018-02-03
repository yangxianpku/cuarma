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

/** @file cuarma/traits/handle.hpp
 *  @encoding:UTF-8 文档编码
    @brief Extracts the underlying handle from a vector, a matrix, an expression etc.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"

#include "cuarma/backend/mem_handle.hpp"

namespace cuarma
{
namespace traits
{
//
// Generic memory handle
//
/** @brief Returns the generic memory handle of an object. Non-const version. */
template<typename T>
cuarma::backend::mem_handle & handle(T & obj)
{
  return obj.handle();
}

/** @brief Returns the generic memory handle of an object. Const-version. */
template<typename T>
cuarma::backend::mem_handle const & handle(T const & obj)
{
  return obj.handle();
}

/** \cond */
inline char   handle(char val)   { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline short  handle(short val)  { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline int    handle(int val)    { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline long   handle(long val)   { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline float  handle(float val)  { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline double handle(double val) { return val; }  //for unification purposes when passing CPU-scalars to kernels

template<typename LHS, typename RHS, typename OP>
cuarma::backend::mem_handle       & handle(cuarma::scalar_expression< const LHS, const RHS, OP> & obj)
{
  return handle(obj.lhs());
}

template<typename LHS, typename RHS, typename OP>
cuarma::backend::mem_handle const & handle(cuarma::matrix_expression<LHS, RHS, OP> const & obj);

template<typename LHS, typename RHS, typename OP>
cuarma::backend::mem_handle const & handle(cuarma::vector_expression<LHS, RHS, OP> const & obj);

template<typename LHS, typename RHS, typename OP>
cuarma::backend::mem_handle const & handle(cuarma::scalar_expression< const LHS, const RHS, OP> const & obj)
{
  return handle(obj.lhs());
}

// proxy objects require extra care (at the moment)
template<typename T>
cuarma::backend::mem_handle       & handle(cuarma::vector_base<T>       & obj)
{
  return obj.handle();
}

template<typename T>
cuarma::backend::mem_handle const & handle(cuarma::vector_base<T> const & obj)
{
  return obj.handle();
}



template<typename T>
cuarma::backend::mem_handle       & handle(cuarma::matrix_range<T>       & obj)
{
  return obj.get().handle();
}

template<typename T>
cuarma::backend::mem_handle const & handle(cuarma::matrix_range<T> const & obj)
{
  return obj.get().handle();
}


template<typename T>
cuarma::backend::mem_handle       & handle(cuarma::matrix_slice<T>      & obj)
{
  return obj.get().handle();
}

template<typename T>
cuarma::backend::mem_handle const & handle(cuarma::matrix_slice<T> const & obj)
{
  return obj.get().handle();
}

template<typename LHS, typename RHS, typename OP>
cuarma::backend::mem_handle const & handle(cuarma::vector_expression<LHS, RHS, OP> const & obj)
{
  return handle(obj.lhs());
}

template<typename LHS, typename RHS, typename OP>
cuarma::backend::mem_handle const & handle(cuarma::matrix_expression<LHS, RHS, OP> const & obj)
{
  return handle(obj.lhs());
}

/** \endcond */

//
// RAM handle extraction
//
/** @brief Generic helper routine for extracting the RAM handle of a cuarma object. Non-const version. */
template<typename T>
typename cuarma::backend::mem_handle::ram_handle_type & ram_handle(T & obj)
{
  return cuarma::traits::handle(obj).ram_handle();
}

/** @brief Generic helper routine for extracting the RAM handle of a cuarma object. Const version. */
template<typename T>
typename cuarma::backend::mem_handle::ram_handle_type const & ram_handle(T const & obj)
{
  return cuarma::traits::handle(obj).ram_handle();
}

/** \cond */
inline cuarma::backend::mem_handle::ram_handle_type & ram_handle(cuarma::backend::mem_handle & h)
{
  return h.ram_handle();
}

inline cuarma::backend::mem_handle::ram_handle_type const & ram_handle(cuarma::backend::mem_handle const & h)
{
  return h.ram_handle();
}
/** \endcond */

//
// OpenCL handle extraction
//


//
// OpenCL context extraction
//




//
// Active handle ID
//
/** @brief Returns an ID for the currently active memory domain of an object */
template<typename T>
cuarma::memory_types active_handle_id(T const & obj)
{
  return handle(obj).get_active_handle_id();
}

template<typename LHS, typename RHS, typename OP>
cuarma::memory_types active_handle_id(cuarma::vector_expression<LHS, RHS, OP> const &);

template<typename LHS, typename RHS, typename OP>
cuarma::memory_types active_handle_id(cuarma::scalar_expression<LHS, RHS, OP> const & obj)
{
  return active_handle_id(obj.lhs());
}

template<typename LHS, typename RHS, typename OP>
cuarma::memory_types active_handle_id(cuarma::vector_expression<LHS, RHS, OP> const & obj)
{
  return active_handle_id(obj.lhs());
}

template<typename LHS, typename RHS, typename OP>
cuarma::memory_types active_handle_id(cuarma::matrix_expression<LHS, RHS, OP> const & obj)
{
  return active_handle_id(obj.lhs());
}

// for user-provided matrix-vector routines:
template<typename LHS, typename NumericT>
cuarma::memory_types active_handle_id(cuarma::vector_expression<LHS, const vector_base<NumericT>, op_prod> const & obj)
{
  return active_handle_id(obj.rhs());
}

/** \endcond */

} //namespace traits
} //namespace cuarma