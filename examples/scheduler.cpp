/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   scheduler.cu
 *   @coding UTF-8
 *   @brief  This tutorial show how to use the low-level scheduler to generate efficient custom kernels at run time.
 *           The purpose of the scheduler is to provide a low-level interface for interfacing cuarma from languages 
 *           other than C++, yet providing the user the ability to specify complex operations. Typical consumers are
 *           scripting languages such as Python, but the facility should be used in the future to also fuse compute 
 *           kernels on the fly.
 *
 *   warning: The scheduler is experimental and only intended for expert users.
 *   @brief  scheduler测试
 */

#include <iostream>
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"

#include "cuarma/scheduler/execute.hpp"
#include "cuarma/scheduler/io.hpp"


/**
*  This tutorial sets up three vectors and finally assigns the sum of two to the third.
*  Although this can be achieved with only a few lines of code using the standard cuarma C++ API,
*  we go through the low-level interface for demonstration purposes.
**/
int main()
{
  typedef float       ScalarType;  // do not change without adjusting the code for the low-level interface below

  /**
  * Create three vectors, initialize two of them with ascending/descending integers:
  **/
  cuarma::vector<ScalarType> arma_vec1(10);
  cuarma::vector<ScalarType> arma_vec2(10);
  cuarma::vector<ScalarType> arma_vec3(10);

  for (unsigned int i = 0; i < 10; ++i)
  {
    arma_vec1[i] = ScalarType(i);
    arma_vec2[i] = ScalarType(10 - i);
  }

  /**
  * Build expression graph for the operation arma_vec3 = arma_vec1 + arma_vec2
  *
  * This requires the following expression graph:
  * \code
  *             ( = )
  *            /      |
  *    arma_vec3      ( + )
  *                 /     |
  *           arma_vec1    arma_vec2
  * \endcode
  * One expression node consists of two leaves and the operation connecting the two.
  * Here we thus need two nodes: One for {arma_vec3, = , link}, where 'link' points to the second node
  * {arma_vec1, +, arma_vec2}.
  *
  * The following is the lowest level on which one could build the expression tree.
  * Even for a C API one would introduce some additional convenience layer such as add_vector_float_to_lhs(...); etc.
  **/
  typedef cuarma::scheduler::statement::container_type   NodeContainerType;   // this is just std::vector<cuarma::scheduler::statement_node>
  NodeContainerType expression_nodes(2);                                        //container with two nodes

  /**
  * <h2>First Node (Assignment)</h2>
  **/

  // specify LHS of first node, i.e. arma_vec3:
  expression_nodes[0].lhs.type_family  = cuarma::scheduler::VECTOR_TYPE_FAMILY;   // family of vectors
  expression_nodes[0].lhs.subtype      = cuarma::scheduler::DENSE_VECTOR_TYPE;    // a dense vector
  expression_nodes[0].lhs.numeric_type = cuarma::scheduler::FLOAT_TYPE;           // vector consisting of floats
  expression_nodes[0].lhs.vector_float = &arma_vec3;                                 // provide pointer to arma_vec3;

  // specify assignment operation for this node:
  expression_nodes[0].op.type_family   = cuarma::scheduler::OPERATION_BINARY_TYPE_FAMILY; // this is a binary operation, so both LHS and RHS operands are important
  expression_nodes[0].op.type          = cuarma::scheduler::OPERATION_BINARY_ASSIGN_TYPE; // assignment operation: '='

  // specify RHS: Just refer to the second node:
  expression_nodes[0].rhs.type_family  = cuarma::scheduler::COMPOSITE_OPERATION_FAMILY; // this links to another node (no need to set .subtype and .numeric_type)
  expression_nodes[0].rhs.node_index   = 1;                                               // index of the other node

  /**
  * <h2>Second Node (Addition)</h2>
  **/

  // LHS
  expression_nodes[1].lhs.type_family  = cuarma::scheduler::VECTOR_TYPE_FAMILY;   // family of vectors
  expression_nodes[1].lhs.subtype      = cuarma::scheduler::DENSE_VECTOR_TYPE;    // a dense vector
  expression_nodes[1].lhs.numeric_type = cuarma::scheduler::FLOAT_TYPE;           // vector consisting of floats
  expression_nodes[1].lhs.vector_float = &arma_vec1;                                 // provide pointer to arma_vec1

  // OP
  expression_nodes[1].op.type_family   = cuarma::scheduler::OPERATION_BINARY_TYPE_FAMILY; // this is a binary operation, so both LHS and RHS operands are important
  expression_nodes[1].op.type          = cuarma::scheduler::OPERATION_BINARY_ADD_TYPE;    // addition operation: '+'

  // RHS
  expression_nodes[1].rhs.type_family  = cuarma::scheduler::VECTOR_TYPE_FAMILY;  // family of vectors
  expression_nodes[1].rhs.subtype      = cuarma::scheduler::DENSE_VECTOR_TYPE;   // a dense vector
  expression_nodes[1].rhs.numeric_type = cuarma::scheduler::FLOAT_TYPE;          // vector consisting of floats
  expression_nodes[1].rhs.vector_float = &arma_vec2;                                // provide pointer to arma_vec2


  /**
  *  Create the full statement (aka. single line of code such as arma_vec3 = arma_vec1 + arma_vec2):
  **/
  cuarma::scheduler::statement vec_addition(expression_nodes);

  /**
  *  Print the expression. Resembles the tree outlined in comments above.
  **/
  std::cout << vec_addition << std::endl;

  /**
  *  Execute the operation
  **/
  cuarma::scheduler::execute(vec_addition);

  /**
  *  Print vectors in order to check the result:
  **/
  std::cout << "arma_vec1: " << arma_vec1 << std::endl;
  std::cout << "arma_vec2: " << arma_vec2 << std::endl;
  std::cout << "arma_vec3: " << arma_vec3 << std::endl;

  /**
  *   That's it! Print success message and exit.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

