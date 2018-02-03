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

/** @file cuarma/scheduler/execute_elementwise.hpp
 *  @encoding:UTF-8 文档编码
    @brief Deals with the execution of unary and binary element-wise operations
*/

#include "cuarma/forwards.h"
#include "cuarma/scheduler/forwards.h"
#include "cuarma/scheduler/execute_util.hpp"
#include "cuarma/blas/vector_operations.hpp"
#include "cuarma/blas/matrix_operations.hpp"

namespace cuarma
{
namespace scheduler
{
namespace detail
{
  // result = element_op(x,y) for vectors or matrices x, y
  inline void element_op(lhs_rhs_element result,
                         lhs_rhs_element const & x,
                         operation_node_type  op_type)
  {
    assert( result.numeric_type == x.numeric_type && bool("Numeric type not the same!"));
    assert( result.type_family == x.type_family && bool("Subtype not the same!"));

    if (x.subtype == DENSE_VECTOR_TYPE)
    {
      assert( result.subtype == x.subtype && bool("result not of vector type for unary elementwise operation"));
      if (x.numeric_type == FLOAT_TYPE)
      {
        switch (op_type)
        {
#define CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPNAME, NumericT, OPTAG) \
        case OPNAME:  cuarma::blas::element_op(*result.vector_##NumericT, \
        cuarma::vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, \
        op_element_unary<OPTAG> >(*x.vector_##NumericT, *x.vector_##NumericT)); break;

        CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ABS_TYPE,   float, op_abs)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ACOS_TYPE,  float, op_acos)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ASIN_TYPE,  float, op_asin)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ATAN_TYPE,  float, op_atan)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_CEIL_TYPE,  float, op_ceil)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_COS_TYPE,   float, op_cos)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_COSH_TYPE,  float, op_cosh)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_EXP_TYPE,   float, op_exp)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_FABS_TYPE,  float, op_fabs)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_FLOOR_TYPE, float, op_floor)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_LOG_TYPE,   float, op_log)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_LOG10_TYPE, float, op_log10)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SIN_TYPE,   float, op_sin)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SINH_TYPE,  float, op_sinh)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SQRT_TYPE,  float, op_sqrt)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_TAN_TYPE,   float, op_tan)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_TANH_TYPE,  float, op_tanh)
            default:
          throw statement_not_supported_exception("Invalid op_type in unary elementwise operations");
        }
      }
      else if (x.numeric_type == DOUBLE_TYPE)
      {
        switch (op_type)
        {
        CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ABS_TYPE,   double, op_abs)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ACOS_TYPE,  double, op_acos)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ASIN_TYPE,  double, op_asin)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ATAN_TYPE,  double, op_atan)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_CEIL_TYPE,  double, op_ceil)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_COS_TYPE,   double, op_cos)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_COSH_TYPE,  double, op_cosh)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_EXP_TYPE,   double, op_exp)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_FABS_TYPE,  double, op_fabs)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_FLOOR_TYPE, double, op_floor)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_LOG_TYPE,   double, op_log)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_LOG10_TYPE, double, op_log10)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SIN_TYPE,   double, op_sin)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SINH_TYPE,  double, op_sinh)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SQRT_TYPE,  double, op_sqrt)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_TAN_TYPE,   double, op_tan)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_TANH_TYPE,  double, op_tanh)

#undef CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP
            default:
          throw statement_not_supported_exception("Invalid op_type in unary elementwise operations");
        }
      }
      else
        throw statement_not_supported_exception("Invalid numeric type in unary elementwise operator");
    }
    else if (x.subtype == DENSE_MATRIX_TYPE)
    {
      if (x.numeric_type == FLOAT_TYPE)
      {
        switch (op_type)
        {
#define CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPNAME, NumericT, OPTAG) \
        case OPNAME:  cuarma::blas::element_op(*result.matrix_##NumericT, \
        cuarma::matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, \
        op_element_unary<OPTAG> >(*x.matrix_##NumericT, *x.matrix_##NumericT)); break;

        CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ABS_TYPE,   float, op_abs)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ACOS_TYPE,  float, op_acos)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ASIN_TYPE,  float, op_asin)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ATAN_TYPE,  float, op_atan)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_CEIL_TYPE,  float, op_ceil)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_COS_TYPE,   float, op_cos)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_COSH_TYPE,  float, op_cosh)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_EXP_TYPE,   float, op_exp)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_FABS_TYPE,  float, op_fabs)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_FLOOR_TYPE, float, op_floor)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_LOG_TYPE,   float, op_log)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_LOG10_TYPE, float, op_log10)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SIN_TYPE,   float, op_sin)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SINH_TYPE,  float, op_sinh)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SQRT_TYPE,  float, op_sqrt)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_TAN_TYPE,   float, op_tan)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_TANH_TYPE,  float, op_tanh)
            default:
          throw statement_not_supported_exception("Invalid op_type in unary elementwise operations");
        }

      }
      else if (x.numeric_type == DOUBLE_TYPE)
      {
        switch (op_type)
        {
        CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ABS_TYPE,   double, op_abs)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ACOS_TYPE,  double, op_acos)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ASIN_TYPE,  double, op_asin)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_ATAN_TYPE,  double, op_atan)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_CEIL_TYPE,  double, op_ceil)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_COS_TYPE,   double, op_cos)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_COSH_TYPE,  double, op_cosh)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_EXP_TYPE,   double, op_exp)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_FABS_TYPE,  double, op_fabs)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_FLOOR_TYPE, double, op_floor)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_LOG_TYPE,   double, op_log)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_LOG10_TYPE, double, op_log10)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SIN_TYPE,   double, op_sin)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SINH_TYPE,  double, op_sinh)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_SQRT_TYPE,  double, op_sqrt)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_TAN_TYPE,   double, op_tan)
            CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP(OPERATION_UNARY_TANH_TYPE,  double, op_tanh)
            default:
          throw statement_not_supported_exception("Invalid op_type in unary elementwise operations");
        }
      }
      else
        throw statement_not_supported_exception("Invalid numeric type in unary elementwise operator");

#undef CUARMA_SCHEDULER_GENERATE_UNARY_ELEMENT_OP

    }
  }

  // result = element_op(x,y) for vectors or matrices x, y
  inline void element_op(lhs_rhs_element result,
                         lhs_rhs_element const & x,
                         lhs_rhs_element const & y,
                         operation_node_type  op_type)
  {
    assert(      x.numeric_type == y.numeric_type && bool("Numeric type not the same!"));
    assert( result.numeric_type == y.numeric_type && bool("Numeric type not the same!"));

    assert(      x.type_family == y.type_family && bool("Subtype not the same!"));
    assert( result.type_family == y.type_family && bool("Subtype not the same!"));

    switch (op_type)
    {

    case OPERATION_BINARY_ELEMENT_DIV_TYPE:
      if (x.subtype == DENSE_VECTOR_TYPE)
      {
        switch (x.numeric_type)
        {
        case FLOAT_TYPE:
          cuarma::blas::element_op(*result.vector_float,
                                       vector_expression<const vector_base<float>,
                                       const vector_base<float>,
                                       op_element_binary<op_div> >(*x.vector_float, *y.vector_float));
          break;
        case DOUBLE_TYPE:
          cuarma::blas::element_op(*result.vector_double,
                                       vector_expression<const vector_base<double>,
                                       const vector_base<double>,
                                       op_element_binary<op_div> >(*x.vector_double, *y.vector_double));
          break;
        default:
          throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
        }
      }
      else if (x.subtype == DENSE_MATRIX_TYPE)
      {
        switch (x.numeric_type)
        {
        case FLOAT_TYPE:
          cuarma::blas::element_op(*result.matrix_float,
                                       matrix_expression< const matrix_base<float>,
                                       const matrix_base<float>,
                                       op_element_binary<op_div> >(*x.matrix_float, *y.matrix_float));
          break;
        case DOUBLE_TYPE:
          cuarma::blas::element_op(*result.matrix_double,
                                       matrix_expression< const matrix_base<double>,
                                       const matrix_base<double>,
                                       op_element_binary<op_div> >(*x.matrix_double, *y.matrix_double));
          break;
        default:
          throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
        }
      }
      else
        throw statement_not_supported_exception("Invalid operand type for binary elementwise division");
      break;


    case OPERATION_BINARY_ELEMENT_PROD_TYPE:
      if (x.subtype == DENSE_VECTOR_TYPE)
      {
        switch (x.numeric_type)
        {
        case FLOAT_TYPE:
          cuarma::blas::element_op(*result.vector_float,
                                       vector_expression<const vector_base<float>,
                                       const vector_base<float>,
                                       op_element_binary<op_prod> >(*x.vector_float, *y.vector_float));
          break;
        case DOUBLE_TYPE:
          cuarma::blas::element_op(*result.vector_double,
                                       vector_expression<const vector_base<double>,
                                       const vector_base<double>,
                                       op_element_binary<op_prod> >(*x.vector_double, *y.vector_double));
          break;
        default:
          throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
        }
      }
      else if (x.subtype == DENSE_MATRIX_TYPE)
      {
        switch (x.numeric_type)
        {
        case FLOAT_TYPE:
          cuarma::blas::element_op(*result.matrix_float,
                                       matrix_expression< const matrix_base<float>,
                                       const matrix_base<float>,
                                       op_element_binary<op_prod> >(*x.matrix_float, *y.matrix_float));
          break;
        case DOUBLE_TYPE:
          cuarma::blas::element_op(*result.matrix_double,
                                       matrix_expression< const matrix_base<double>,
                                       const matrix_base<double>,
                                       op_element_binary<op_prod> >(*x.matrix_double, *y.matrix_double));
          break;
        default:
          throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
        }
      }
      else
        throw statement_not_supported_exception("Invalid operand type for binary elementwise division");
      break;


    case OPERATION_BINARY_ELEMENT_POW_TYPE:
      if (x.subtype == DENSE_VECTOR_TYPE)
      {
        switch (x.numeric_type)
        {
        case FLOAT_TYPE:
          cuarma::blas::element_op(*result.vector_float,
                                       vector_expression<const vector_base<float>,
                                       const vector_base<float>,
                                       op_element_binary<op_pow> >(*x.vector_float, *y.vector_float));
          break;
        case DOUBLE_TYPE:
          cuarma::blas::element_op(*result.vector_double,
                                       vector_expression<const vector_base<double>,
                                       const vector_base<double>,
                                       op_element_binary<op_pow> >(*x.vector_double, *y.vector_double));
          break;
        default:
          throw statement_not_supported_exception("Invalid numeric type for binary elementwise division");
        }
      }
      else if (x.subtype == DENSE_MATRIX_TYPE)
      {
        switch (x.numeric_type)
        {
        case FLOAT_TYPE:
          cuarma::blas::element_op(*result.matrix_float,
                                       matrix_expression< const matrix_base<float>,
                                       const matrix_base<float>,
                                       op_element_binary<op_pow> >(*x.matrix_float, *y.matrix_float));
          break;
        case DOUBLE_TYPE:
          cuarma::blas::element_op(*result.matrix_double,
                                       matrix_expression< const matrix_base<double>,
                                       const matrix_base<double>,
                                       op_element_binary<op_pow> >(*x.matrix_double, *y.matrix_double));
          break;
        default:
          throw statement_not_supported_exception("Invalid numeric type for binary elementwise power");
        }
      }
      else
        throw statement_not_supported_exception("Invalid operand type for binary elementwise power");
      break;

    default:
      throw statement_not_supported_exception("Invalid operation type for binary elementwise operations");
    }
  }
}

/** @brief Deals with x = RHS where RHS is a vector expression */
inline void execute_element_composite(statement const & s, statement_node const & root_node)
{
  statement_node const & leaf = s.array()[root_node.rhs.node_index];
  cuarma::context ctx = detail::extract_context(root_node);

  statement_node new_root_lhs;
  statement_node new_root_rhs;

  // check for temporary on lhs:
  if (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY)
  {
    detail::new_element(new_root_lhs.lhs, root_node.lhs, ctx);

    new_root_lhs.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
    new_root_lhs.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

    new_root_lhs.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
    new_root_lhs.rhs.subtype      = INVALID_SUBTYPE;
    new_root_lhs.rhs.numeric_type = INVALID_NUMERIC_TYPE;
    new_root_lhs.rhs.node_index   = leaf.lhs.node_index;

    // work on subexpression:
    // TODO: Catch exception, free temporary, then rethrow
    detail::execute_composite(s, new_root_lhs);
  }

  if (leaf.op.type == OPERATION_BINARY_ELEMENT_PROD_TYPE || leaf.op.type == OPERATION_BINARY_ELEMENT_DIV_TYPE || leaf.op.type == OPERATION_BINARY_ELEMENT_POW_TYPE)
  {
    // check for temporary on rhs:
    if (leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY)
    {
      detail::new_element(new_root_rhs.lhs, root_node.lhs, ctx);

      new_root_rhs.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
      new_root_rhs.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

      new_root_rhs.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
      new_root_rhs.rhs.subtype      = INVALID_SUBTYPE;
      new_root_rhs.rhs.numeric_type = INVALID_NUMERIC_TYPE;
      new_root_rhs.rhs.node_index   = leaf.rhs.node_index;

      // work on subexpression:
      // TODO: Catch exception, free temporary, then rethrow
      detail::execute_composite(s, new_root_rhs);
    }

    lhs_rhs_element x = (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY) ? new_root_lhs.lhs : leaf.lhs;
    lhs_rhs_element y = (leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) ? new_root_rhs.lhs : leaf.rhs;

    // compute element-wise operation:
    detail::element_op(root_node.lhs, x, y, leaf.op.type);

    if (leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY)
      detail::delete_element(new_root_rhs.lhs);
  }
  else if (leaf.op.type_family  == OPERATION_UNARY_TYPE_FAMILY)
  {
    lhs_rhs_element x = (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY) ? new_root_lhs.lhs : leaf.lhs;

    // compute element-wise operation:
    detail::element_op(root_node.lhs, x, leaf.op.type);
  }
  else
    throw statement_not_supported_exception("Unsupported elementwise operation.");

  // clean up:
  if (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY)
    detail::delete_element(new_root_lhs.lhs);

}


} // namespace scheduler
} // namespace cuarma