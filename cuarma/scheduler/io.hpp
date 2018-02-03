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


/** @file cuarma/scheduler/io.hpp
 *  @encoding:UTF-8 文档编码
    @brief Some helper routines for reading/writing/printing scheduler expressions
*/

#include <iostream>
#include <sstream>
#include "cuarma/forwards.h"
#include "cuarma/scheduler/forwards.h"


namespace cuarma
{
namespace scheduler
{
namespace detail
{
#define CUARMA_TRANSLATE_OP_TO_STRING(NAME)   case NAME: return #NAME;

  /** @brief Helper routine for converting the operation enums to string */
  inline std::string to_string(cuarma::scheduler::op_element op_elem)
  {
    if (op_elem.type_family == OPERATION_UNARY_TYPE_FAMILY)
    {
      switch (op_elem.type)
      {
      CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_ABS_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_ACOS_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_ASIN_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_ATAN_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_CEIL_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_COS_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_COSH_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_EXP_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_FABS_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_FLOOR_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_LOG_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_LOG10_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_SIN_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_SINH_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_SQRT_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_TAN_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_TANH_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_TRANS_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_NORM_1_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_NORM_2_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_NORM_INF_TYPE)

          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_UNARY_MINUS_TYPE)

          default: throw statement_not_supported_exception("Cannot convert unary operation to string");
      }
    }
    else if (op_elem.type_family == OPERATION_BINARY_TYPE_FAMILY)
    {
      switch (op_elem.type)
      {
      CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_ASSIGN_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_INPLACE_ADD_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_INPLACE_SUB_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_ADD_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_SUB_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_MAT_VEC_PROD_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_MAT_MAT_PROD_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_MULT_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_DIV_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_ELEMENT_PROD_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_ELEMENT_DIV_TYPE)
          CUARMA_TRANSLATE_OP_TO_STRING(OPERATION_BINARY_INNER_PROD_TYPE)

          default: throw statement_not_supported_exception("Cannot convert unary operation to string");
      }
    }
    else if (op_elem.type_family == OPERATION_INVALID_TYPE_FAMILY)
    {
      if (op_elem.type == OPERATION_INVALID_TYPE)
        return "OPERATION_INVALID_TYPE";
      else
        throw statement_not_supported_exception("Unknown invalid operation type when converting to string");
    }
    else
      throw statement_not_supported_exception("Unknown operation family when converting to string");
  }
#undef CUARMA_TRANSLATE_OP_TO_STRING

#define CUARMA_TRANSLATE_ELEMENT_TO_STRING(NAME, ELEMENT)   case NAME: ss << "(" << element.ELEMENT << ")"; return #NAME + ss.str();

  /** @brief Helper routine converting the enum and union values inside a statement node to a string */
  inline std::string to_string(cuarma::scheduler::lhs_rhs_element element)
  {
    std::stringstream ss;

    if (element.type_family == COMPOSITE_OPERATION_FAMILY)
    {
      ss << "(" << element.node_index << ")";
      return "COMPOSITE_OPERATION_FAMILY" + ss.str();
    }
    else if (element.type_family == SCALAR_TYPE_FAMILY)
    {
      if (element.subtype == HOST_SCALAR_TYPE)
      {
        ss << ", HOST_SCALAR_TYPE ";
        switch (element.numeric_type)
        {
        CUARMA_TRANSLATE_ELEMENT_TO_STRING(CHAR_TYPE,   host_char)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(UCHAR_TYPE,  host_uchar)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(SHORT_TYPE,  host_short)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(USHORT_TYPE, host_ushort)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(INT_TYPE,    host_int)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(UINT_TYPE,   host_uint)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(LONG_TYPE,   host_long)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(ULONG_TYPE,  host_ulong)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(FLOAT_TYPE,  host_float)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(DOUBLE_TYPE, host_double)

            default: throw statement_not_supported_exception("Cannot convert host scalar type to string");
        }
      }
      else
      {
        ss << ", DEVICE_SCALAR_TYPE";
        switch (element.numeric_type)
        {
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(CHAR_TYPE,   scalar_char)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(UCHAR_TYPE,  scalar_uchar)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(SHORT_TYPE,  scalar_short)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(USHORT_TYPE, scalar_ushort)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(INT_TYPE,    scalar_int)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(UINT_TYPE,   scalar_uint)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(LONG_TYPE,   scalar_long)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(ULONG_TYPE,  scalar_ulong)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(HALF_TYPE,   scalar_half)
        CUARMA_TRANSLATE_ELEMENT_TO_STRING(FLOAT_TYPE,  scalar_float)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(DOUBLE_TYPE, scalar_double)
            default: throw statement_not_supported_exception("Cannot convert scalar type to string");
        }
      }
    }
    else if (element.type_family == VECTOR_TYPE_FAMILY)
    {
      ss << ", DENSE_VECTOR_TYPE ";
      switch (element.numeric_type)
      {
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(CHAR_TYPE,   vector_char)
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(UCHAR_TYPE,  vector_uchar)
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(SHORT_TYPE,  vector_short)
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(USHORT_TYPE, vector_ushort)
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(INT_TYPE,    vector_int)
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(UINT_TYPE,   vector_uint)
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(LONG_TYPE,   vector_long)
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(ULONG_TYPE,  vector_ulong)
      //CUARMA_TRANSLATE_ELEMENT_TO_STRING(HALF_TYPE,   vector_half)
      CUARMA_TRANSLATE_ELEMENT_TO_STRING(FLOAT_TYPE,  vector_float)
          CUARMA_TRANSLATE_ELEMENT_TO_STRING(DOUBLE_TYPE, vector_double)

          default: throw statement_not_supported_exception("Cannot convert vector type to string");
      }
    }
    else if (element.type_family == MATRIX_TYPE_FAMILY)
    {
      if (element.subtype == DENSE_MATRIX_TYPE)
      {
        ss << ", DENSE_MATRIX_TYPE ";
        switch (element.numeric_type)
        {
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(CHAR_TYPE,   matrix_char)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(UCHAR_TYPE,  matrix_uchar)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(SHORT_TYPE,  matrix_short)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(USHORT_TYPE, matrix_ushort)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(INT_TYPE,    matrix_int)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(UINT_TYPE,   matrix_uint)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(LONG_TYPE,   matrix_long)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(ULONG_TYPE,  matrix_ulong)
        //CUARMA_TRANSLATE_ELEMENT_TO_STRING(HALF_TYPE,   matrix_half)
        CUARMA_TRANSLATE_ELEMENT_TO_STRING(FLOAT_TYPE,  matrix_float)
            CUARMA_TRANSLATE_ELEMENT_TO_STRING(DOUBLE_TYPE, matrix_double)

            default: throw statement_not_supported_exception("Cannot convert dense matrix type to string");
        }
      }
      else
        throw statement_not_supported_exception("Cannot convert matrix sub-type to string");
    }
    else if (element.type_family == INVALID_TYPE_FAMILY)
    {
      return "INVALID_TYPE_FAMILY";
    }
    else
      throw statement_not_supported_exception("Unknown operation family when converting to string");
  }

#undef CUARMA_TRANSLATE_ELEMENT_TO_STRING

} // namespace detail


/** @brief Print a single statement_node. Non-recursive */
inline std::ostream & operator<<(std::ostream & os, cuarma::scheduler::statement_node const & s_node)
{
  os << "LHS: " << detail::to_string(s_node.lhs) << ", "
     << "OP: "  << detail::to_string(s_node.op) << ", "
     << "RHS: " << detail::to_string(s_node.rhs);

  return os;
}

namespace detail
{
  /** @brief Recursive worker routine for printing a whole statement */
  inline void print_node(std::ostream & os, cuarma::scheduler::statement const & s, arma_size_t node_index, arma_size_t indent = 0)
  {
    typedef cuarma::scheduler::statement::container_type   StatementNodeContainer;
    typedef cuarma::scheduler::statement::value_type       StatementNode;

    StatementNodeContainer const & nodes = s.array();
    StatementNode const & current_node = nodes[node_index];

    for (arma_size_t i=0; i<indent; ++i)
      os << " ";

    os << "Node " << node_index << ": " << current_node << std::endl;

    if (current_node.lhs.type_family == COMPOSITE_OPERATION_FAMILY)
      print_node(os, s, current_node.lhs.node_index, indent+1);

    if (current_node.rhs.type_family == COMPOSITE_OPERATION_FAMILY)
      print_node(os, s, current_node.rhs.node_index, indent+1);
  }
}

/** @brief Writes a string identifying the scheduler statement to an output stream.
  *
  * Typically used for debugging
  * @param os    The output stream
  * @param s     The statement object
  */
inline std::ostream & operator<<(std::ostream & os, cuarma::scheduler::statement const & s)
{
  detail::print_node(os, s, s.root());
  return os;
}

}
} //namespace cuarma