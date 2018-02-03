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

/*  @encoding:UTF-8 文档编码 */

#include "cuarma/device_specific/forwards.h"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/scheduler/forwards.h"

namespace cuarma
{
namespace scheduler
{
namespace preset
{

template<typename NumericT>
statement mat_mat_prod(NumericT alpha, cuarma::matrix_base<NumericT> const * A, bool A_trans,
                       cuarma::matrix_base<NumericT> const * B, bool B_trans,
                       NumericT beta, cuarma::matrix_base<NumericT> const * C)
{
  arma_size_t dummy = 0;
  statement::container_type array(7);

  scheduler::statement::add_element(dummy, array[0].lhs, *C);
  array[0].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
  array[0].op.type = OPERATION_BINARY_ASSIGN_TYPE;
  array[0].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
  array[0].rhs.node_index = 1;

  array[1].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
  array[1].lhs.node_index = 2;
  array[1].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
  array[1].op.type = OPERATION_BINARY_ADD_TYPE;
  array[1].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
  array[1].rhs.node_index = 6;

  array[2].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
  array[2].lhs.node_index = 3;
  array[2].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
  array[2].op.type = OPERATION_BINARY_MULT_TYPE;
  scheduler::statement::add_element(dummy, array[2].rhs, alpha);


  if (A_trans)
  {
    array[3].lhs.type_family = COMPOSITE_OPERATION_FAMILY;
    array[3].lhs.node_index = 4;

    statement::add_element(dummy, array[4].lhs, *A);
    array[4].op.type_family = OPERATION_UNARY_TYPE_FAMILY;
    array[4].op.type = OPERATION_UNARY_TRANS_TYPE;
  }
  else
  {
    statement::add_element(dummy, array[3].lhs, *A);
  }

  array[3].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
  array[3].op.type = OPERATION_BINARY_MAT_MAT_PROD_TYPE;

  if (B_trans)
  {
    array[3].rhs.type_family = COMPOSITE_OPERATION_FAMILY;
    array[3].rhs.node_index = 5;

    statement::add_element(dummy, array[5].lhs, *B);
    array[5].op.type_family = OPERATION_UNARY_TYPE_FAMILY;
    array[5].op.type = OPERATION_UNARY_TRANS_TYPE;
  }
  else
  {
    statement::add_element(dummy, array[3].rhs, *B);
  }

  scheduler::statement::add_element(dummy, array[6].rhs, *C);
  array[6].op.type_family = OPERATION_BINARY_TYPE_FAMILY;
  array[6].op.type = OPERATION_BINARY_MULT_TYPE;
  scheduler::statement::add_element(dummy, array[6].rhs, beta);



  return statement(array);
}

}
}
}