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

/** @file cuarma/scalar.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Implementation of the cuarma scalar class
*/

#include <iostream>

#include "cuarma/forwards.h"
#include "cuarma/backend/memory.hpp"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/blas/scalar_operations.hpp"
#include "cuarma/traits/handle.hpp"



namespace cuarma
{
/** @brief A proxy for scalar expressions (e.g. from inner vector products)
  *
  * assumption: dim(LHS) >= dim(RHS), where dim(scalar) = 0, dim(vector) = 1 and dim(matrix = 2)
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  * @tparam OP    The operation tag
  */
template<typename LHS, typename RHS, typename OP>
class scalar_expression
{
  typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<DummyType>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    cuarma::scalar<ScalarType> temp;
    temp = *this;
    return temp;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};


/** @brief Specialization of a scalar expression for inner products. Allows for a final reduction on the CPU
  *
  * assumption: dim(LHS) >= dim(RHS), where dim(scalar) = 0, dim(vector) = 1 and dim(matrix = 2)
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  * @tparam OP    The operation tag
  */
template<typename LHS, typename RHS>
class scalar_expression<LHS, RHS, op_inner_prod>
{
  //typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<LHS>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    ScalarType result;
    cuarma::blas::inner_prod_cpu(lhs_, rhs_, result);
    return result;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};


/** @brief Specialization of a scalar expression for norm_1. Allows for a final reduction on the CPU
  *
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  */
template<typename LHS, typename RHS>
class scalar_expression<LHS, RHS, op_norm_1>
{
  //typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<LHS>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    ScalarType result;
    cuarma::blas::norm_1_cpu(lhs_, result);
    return result;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};

/** @brief Specialization of a scalar expression for norm_2. Allows for a final reduction on the CPU
  *
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  */
template<typename LHS, typename RHS>
class scalar_expression<LHS, RHS, op_norm_2>
{
  //typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<LHS>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    ScalarType result;
    cuarma::blas::norm_2_cpu(lhs_, result);
    return result;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};


/** @brief Specialization of a scalar expression for norm_inf. Allows for a final reduction on the CPU
  *
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  */
template<typename LHS, typename RHS>
class scalar_expression<LHS, RHS, op_norm_inf>
{
  //typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<LHS>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    ScalarType result;
    cuarma::blas::norm_inf_cpu(lhs_, result);
    return result;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};

/** @brief Specialization of a scalar expression for max(). Allows for a final reduction on the CPU
  *
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  */
template<typename LHS, typename RHS>
class scalar_expression<LHS, RHS, op_max>
{
  //typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<LHS>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    ScalarType result;
    cuarma::blas::max_cpu(lhs_, result);
    return result;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};


/** @brief Specialization of a scalar expression for norm_inf. Allows for a final reduction on the CPU
  *
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  */
template<typename LHS, typename RHS>
class scalar_expression<LHS, RHS, op_min>
{
  //typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<LHS>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    ScalarType result;
    cuarma::blas::min_cpu(lhs_, result);
    return result;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};

/** @brief Specialization of a scalar expression for norm_inf. Allows for a final reduction on the CPU
  *
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  */
template<typename LHS, typename RHS>
class scalar_expression<LHS, RHS, op_sum>
{
  //typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<LHS>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    ScalarType result;
    cuarma::blas::sum_cpu(lhs_, result);
    return result;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};


/** @brief Specialization of a scalar expression for norm_frobenius. Allows for a final reduction on the CPU
  *
  * @tparam LHS   The left hand side operand
  * @tparam RHS   The right hand side operand
  */
template<typename LHS, typename RHS>
class scalar_expression<LHS, RHS, op_norm_frobenius>
{
  //typedef typename LHS::value_type          DummyType; //Visual C++ 2005 does not allow to write LHS::value_type::value_type
public:
  typedef typename cuarma::result_of::cpu_value_type<LHS>::type    ScalarType;

  scalar_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Returns the left hand side operand */
  LHS & lhs() const { return lhs_; }
  /** @brief Returns the left hand side operand */
  RHS & rhs() const { return rhs_; }

  /** @brief Conversion operator to a cuarma scalar */
  operator ScalarType () const
  {
    ScalarType result;
    cuarma::blas::norm_frobenius_cpu(lhs_, result);
    return result;
  }

private:
  LHS & lhs_;
  RHS & rhs_;
};




/** @brief This class represents a single scalar value on the GPU and behaves mostly like a built-in scalar type like float or double.
  *
  * Since every read and write operation requires a CPU->GPU or GPU->CPU transfer, this type should be used with care.
  * The advantage of this type is that the GPU command queue can be filled without blocking read operations.
  *
  * @tparam NumericT  Either float or double. Checked at compile time.
  */
template<class NumericT>
class scalar
{
  typedef scalar<NumericT>         self_type;
public:
  typedef cuarma::backend::mem_handle                     handle_type;
  typedef arma_size_t                                     size_type;

  /** @brief Returns the underlying host scalar type. */
  typedef NumericT   value_type;

  /** @brief Creates the scalar object, but does not yet allocate memory. Thus, scalar<> can also be a global variable (if really necessary). */
  scalar() {}

  /** @brief Allocates the memory for the scalar and sets it to the supplied value. */
  scalar(NumericT val, cuarma::context ctx = cuarma::context())
  {
    cuarma::backend::memory_create(val_, sizeof(NumericT), ctx, &val);
  }



  /** @brief Allocates memory for the scalar and sets it to the result of supplied expression. */
  template<typename T1, typename T2, typename OP>
  scalar(scalar_expression<T1, T2, OP> const & proxy)
  {
    val_.switch_active_handle_id(cuarma::traits::handle(proxy.lhs()).get_active_handle_id());
    cuarma::backend::memory_create(val_, sizeof(NumericT), cuarma::traits::context(proxy));
    *this = proxy;
  }

  //copy constructor
  /** @brief Copy constructor. Allocates new memory for the scalar and copies the value of the supplied scalar */
  scalar(const scalar & other)
  {
    if (other.handle().get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED)
    {
      //copy value:
      val_.switch_active_handle_id(other.handle().get_active_handle_id());
      cuarma::backend::memory_create(val_, sizeof(NumericT), cuarma::traits::context(other));
      cuarma::backend::memory_copy(other.handle(), val_, 0, 0, sizeof(NumericT));
    }
  }

  /** @brief Reads the value of the scalar from the GPU and returns the float or double value. */
  operator NumericT() const
  {
    // make sure the scalar contains reasonable data:
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized, cannot read!"));

    NumericT tmp;
    cuarma::backend::memory_read(val_, 0, sizeof(NumericT), &tmp);
    return tmp;
  }

  /** @brief Assigns a vector entry. */
  self_type & operator= (entry_proxy<NumericT> const & other)
  {
    init_if_necessary(cuarma::traits::context(other));
    cuarma::backend::memory_copy(other.handle(), val_, other.index() * sizeof(NumericT), 0, sizeof(NumericT));
    return *this;
  }

  /** @brief Assigns the value from another scalar. */
  self_type & operator= (scalar<NumericT> const & other)
  {
    init_if_necessary(cuarma::traits::context(other));
    cuarma::backend::memory_copy(other.handle(), val_, 0, 0, sizeof(NumericT));
    return *this;
  }

  self_type & operator= (float cpu_other)
  {
    init_if_necessary(cuarma::context());

    //copy value:
    NumericT value = static_cast<NumericT>(cpu_other);
    cuarma::backend::memory_write(val_, 0, sizeof(NumericT), &value);
    return *this;
  }

  self_type & operator= (double cpu_other)
  {
    init_if_necessary(cuarma::context());

    NumericT value = static_cast<NumericT>(cpu_other);
    cuarma::backend::memory_write(val_, 0, sizeof(NumericT), &value);
    return *this;
  }

  self_type & operator= (long cpu_other)
  {
    init_if_necessary(cuarma::context());

    NumericT value = static_cast<NumericT>(cpu_other);
    cuarma::backend::memory_write(val_, 0, sizeof(NumericT), &value);
    return *this;
  }

  self_type & operator= (unsigned long cpu_other)
  {
    init_if_necessary(cuarma::context());

    NumericT value = static_cast<NumericT>(cpu_other);
    cuarma::backend::memory_write(val_, 0, sizeof(NumericT), &value);
    return *this;
  }

  self_type & operator= (int cpu_other)
  {
    init_if_necessary(cuarma::context());

    NumericT value = static_cast<NumericT>(cpu_other);
    cuarma::backend::memory_write(val_, 0, sizeof(NumericT), &value);
    return *this;
  }

  self_type & operator= (unsigned int cpu_other)
  {
    init_if_necessary(cuarma::context());

    NumericT value = static_cast<NumericT>(cpu_other);
    cuarma::backend::memory_write(val_, 0, sizeof(NumericT), &value);
    return *this;
  }

  /** @brief Sets the scalar to the result of supplied inner product expression. */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_inner_prod> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::inner_prod_impl(proxy.lhs(), proxy.rhs(), *this);
    return *this;
  }

  /** @brief Sets the scalar to the result of supplied norm_1 expression. */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_norm_1> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::norm_1_impl(proxy.lhs(), *this);
    return *this;
  }

  /** @brief Sets the scalar to the result of supplied norm_2 expression. */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_norm_2> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::norm_2_impl(proxy.lhs(), *this);
    return *this;
  }

  /** @brief Sets the scalar to the result of supplied norm_inf expression. */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_norm_inf> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::norm_inf_impl(proxy.lhs(), *this);
    return *this;
  }

  /** @brief Sets the scalar to the result of supplied max expression. */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_max> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::max_impl(proxy.lhs(), *this);
    return *this;
  }

  /** @brief Sets the scalar to the result of supplied min expression. */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_min> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::min_impl(proxy.lhs(), *this);
    return *this;
  }

  /** @brief Sets the scalar to the result of supplied sum expression. */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_sum> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::sum_impl(proxy.lhs(), *this);
    return *this;
  }


  /** @brief Sets the scalar to the result of supplied norm_frobenius expression. */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_norm_frobenius> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::norm_frobenius_impl(proxy.lhs(), *this);
    return *this;
  }

  /** @brief Sets the scalar to the inverse with respect to addition of the supplied sub-expression */
  template<typename T1, typename T2>
  self_type & operator= (scalar_expression<T1, T2, op_flip_sign> const & proxy)
  {
    init_if_necessary(cuarma::traits::context(proxy));

    cuarma::blas::as(*this, proxy.lhs(), NumericT(-1.0), 1, false, true);
    return *this;
  }


  /** @brief Inplace addition of a cuarma scalar */
  self_type & operator += (scalar<NumericT> const & other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    cuarma::blas::asbs(*this,                                       // s1 =
                           *this, NumericT(1.0), 1, false, false,     //       s1 * 1.0
                           other, NumericT(1.0), 1, false, false);    //     + s2 * 1.0
    return *this;
  }
  /** @brief Inplace addition of a host scalar (float or double) */
  self_type & operator += (NumericT other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    cuarma::blas::asbs(*this,                                       // s1 =
                           *this, NumericT(1.0), 1, false, false,     //       s1 * 1.0
                           other, NumericT(1.0), 1, false, false);    //     + s2 * 1.0
    return *this;
  }


  /** @brief Inplace subtraction of a cuarma scalar */
  self_type & operator -= (scalar<NumericT> const & other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    cuarma::blas::asbs(*this,                                       // s1 =
                           *this, NumericT(1.0), 1, false, false,     //       s1 * 1.0
                           other, NumericT(-1.0), 1, false, false);   //     + s2 * (-1.0)
    return *this;
  }
  /** @brief Inplace subtraction of a host scalar (float or double) */
  self_type & operator -= (NumericT other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    cuarma::blas::asbs(*this,                                       // s1 =
                           *this, NumericT(1.0), 1, false, false,     //       s1 * 1.0
                           other, NumericT(-1.0), 1, false, false);   //     + s2 * (-1.0)
    return *this;
  }


  /** @brief Inplace multiplication with a cuarma scalar */
  self_type & operator *= (scalar<NumericT> const & other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    cuarma::blas::as(*this,                                       // s1 =
                         *this, other, 1, false, false);              //      s1 * s2
    return *this;
  }
  /** @brief Inplace  multiplication with a host scalar (float or double) */
  self_type & operator *= (NumericT other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    cuarma::blas::as(*this,                                       // s1 =
                         *this, other, 1, false, false);              //      s1 * s2
    return *this;
  }


  //////////////// operator /=    ////////////////////////////
  /** @brief Inplace division with a cuarma scalar */
  self_type & operator /= (scalar<NumericT> const & other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    cuarma::blas::as(*this,                                       // s1 =
                         *this, other, 1, true, false);              //      s1 / s2
    return *this;
  }
  /** @brief Inplace division with a host scalar (float or double) */
  self_type & operator /= (NumericT other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    cuarma::blas::as(*this,                                       // s1 =
                         *this, other, 1, true, false);              //      s1 / s2
    return *this;
  }


  //////////////// operator + ////////////////////////////
  /** @brief Addition of two cuarma scalars */
  self_type operator + (scalar<NumericT> const & other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = 0;

    cuarma::blas::asbs(result,                                       // result =
                           *this, NumericT(1.0), 1, false, false,      //            *this * 1.0
                           other, NumericT(1.0), 1, false, false);     //          + other * 1.0

    return result;
  }
  /** @brief Addition of a cuarma scalar with a scalar expression */
  template<typename T1, typename T2, typename OP>
  self_type operator + (scalar_expression<T1, T2, OP> const & proxy) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = proxy;

    cuarma::blas::asbs(result,                                       // result =
                           *this, NumericT(1.0), 1, false, false,      //            *this * 1.0
                           result, NumericT(1.0), 1, false, false);     //        + result * 1.0

    return result;
  }
  /** @brief Addition of a cuarma scalar with a host scalar (float, double) */
  self_type operator + (NumericT other)
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = 0;

    cuarma::blas::asbs(result,                                       // result =
                           *this, NumericT(1.0), 1, false, false,      //            *this * 1.0
                           other, NumericT(1.0), 1, false, false);     //          + other * 1.0

    return result;
  }


  //////////////// operator - ////////////////////////////

  /** @brief Sign flip of the scalar. Does not evaluate immediately, but instead returns an expression template object */
  scalar_expression<const self_type, const self_type, op_flip_sign> operator-() const
  {
    return scalar_expression<const self_type, const self_type, op_flip_sign>(*this, *this);
  }


  /** @brief Subtraction of two cuarma scalars */
  self_type operator - (scalar<NumericT> const & other) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = 0;

    cuarma::blas::asbs(result,                                       // result =
                           *this, NumericT(1.0), 1, false, false,      //            *this * 1.0
                           other, NumericT(-1.0), 1, false, false);    //          + other * (-1.0)

    return result;
  }
  /** @brief Subtraction of a cuarma scalar from a scalar expression */
  template<typename T1, typename T2, typename OP>
  self_type operator - (scalar_expression<T1, T2, OP> const & proxy) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = proxy;

    cuarma::blas::asbs(result,                                       // result =
                           *this, NumericT(1.0), 1 , false, false,    //            *this * 1.0
                           result, NumericT(-1.0), 1, false, false);  //          + result * (-1.0)

    return result;
  }
  /** @brief Subtraction of a host scalar (float, double) from a cuarma scalar */
  scalar<NumericT> operator - (NumericT other) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = 0;

    cuarma::blas::asbs(result,                                       // result =
                           *this, NumericT(1.0), 1, false, false,      //            *this * 1.0
                           other, NumericT(-1.0), 1, false, false);    //          + other * (-1.0)

    return result;
  }

  //////////////// operator * ////////////////////////////
  /** @brief Multiplication of two cuarma scalars */
  self_type operator * (scalar<NumericT> const & other) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    scalar<NumericT> result = 0;

    cuarma::blas::as(result,                                     // result =
                         *this, other, 1, false, false);              //          *this * other

    return result;
  }
  /** @brief Multiplication of a cuarma scalar with a scalar expression */
  template<typename T1, typename T2, typename OP>
  self_type operator * (scalar_expression<T1, T2, OP> const & proxy) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = proxy;

    cuarma::blas::as(result,                                       // result =
                         *this, result, 1, false, false);              //            *this * proxy

    return result;
  }
  /** @brief Multiplication of a host scalar (float, double) with a cuarma scalar */
  self_type operator * (NumericT other) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    scalar<NumericT> result = 0;

    cuarma::blas::as(result,                                     // result =
                         *this, other, 1, false, false);              //          *this * other

    return result;
  }

  //////////////// operator /    ////////////////////////////
  /** @brief Division of two cuarma scalars */
  self_type operator / (scalar<NumericT> const & other) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = 0;

    cuarma::blas::as(result,                                     // result =
                         *this, other, 1, true, false);              //           *this / other

    return result;
  }
  /** @brief Division of a cuarma scalar by a scalar expression */
  template<typename T1, typename T2, typename OP>
  self_type operator / (scalar_expression<T1, T2, OP> const & proxy) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = proxy;

    cuarma::blas::as(result,                                    // result =
                         *this, result, 1, true, false);        //           *this / proxy

    return result;
  }
  /** @brief Division of a cuarma scalar by a host scalar (float, double)*/
  self_type operator / (NumericT other) const
  {
    assert( val_.get_active_handle_id() != cuarma::MEMORY_NOT_INITIALIZED && bool("Scalar not initialized!"));

    self_type result = 0;

    cuarma::blas::as(result,                                     // result =
                         *this, other, 1, true, false);          //           *this / other

    return result;
  }

  /** @brief Returns the memory handle, non-const version */
  handle_type & handle() { return val_; }

  /** @brief Returns the memory handle, const version */
  const handle_type & handle() const { return val_; }

private:

  void init_if_necessary(cuarma::context ctx)
  {
    if (val_.get_active_handle_id() == cuarma::MEMORY_NOT_INITIALIZED)
    {
      cuarma::backend::memory_create(val_, sizeof(NumericT), ctx);
    }
  }

  handle_type val_;
};


//stream operators:
/** @brief Allows to directly print the value of a scalar to an output stream */
template<class NumericT>
std::ostream & operator<<(std::ostream & s, const scalar<NumericT> & val)
{
  NumericT temp = val;
  s << temp;
  return s;
}

/** @brief Allows to directly read a value of a scalar from an input stream */
template<class NumericT>
std::istream & operator>>(std::istream & s, const scalar<NumericT> & val)
{
  NumericT temp;
  s >> temp;
  val = temp;
  return s;
}

} //namespace cuarma
