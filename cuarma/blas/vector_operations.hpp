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
/** @file cuarma/blas/vector_operations.hpp
    @brief Implementations of vector operations.
*/

#include "cuarma/forwards.h"
#include "cuarma/range.hpp"
#include "cuarma/scalar.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/traits/handle.hpp"
#include "cuarma/traits/stride.hpp"
#include "cuarma/blas/detail/op_executor.hpp"
#include "cuarma/blas/host_based/vector_operations.hpp"



#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/vector_operations.hpp"
#endif

namespace cuarma
{
  namespace blas
  {
    template<typename DestNumericT, typename SrcNumericT>
    void convert(vector_base<DestNumericT> & dest, vector_base<SrcNumericT> const & src)
    {
      assert(cuarma::traits::size(dest) == cuarma::traits::size(src) && bool("Incompatible vector sizes in v1 = v2 (convert): size(v1) != size(v2)"));

      switch (cuarma::traits::handle(dest).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::convert(dest, src);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::convert(dest, src);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    template<typename T, typename ScalarType1>
    void av(vector_base<T> & vec1,
            vector_base<T> const & vec2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
    {
      assert(cuarma::traits::size(vec1) == cuarma::traits::size(vec2) && bool("Incompatible vector sizes in v1 = v2 @ alpha: size(v1) != size(v2)"));

      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::av(vec1, vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::av(vec1, vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename T, typename ScalarType1, typename ScalarType2>
    void avbv(vector_base<T> & vec1,
              vector_base<T> const & vec2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
              vector_base<T> const & vec3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      assert(cuarma::traits::size(vec1) == cuarma::traits::size(vec2) && bool("Incompatible vector sizes in v1 = v2 @ alpha + v3 @ beta: size(v1) != size(v2)"));
      assert(cuarma::traits::size(vec2) == cuarma::traits::size(vec3) && bool("Incompatible vector sizes in v1 = v2 @ alpha + v3 @ beta: size(v2) != size(v3)"));

      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::avbv(vec1,
                                                  vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                  vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::avbv(vec1,
                                       vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                       vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename T, typename ScalarType1, typename ScalarType2>
    void avbv_v(vector_base<T> & vec1,
                vector_base<T> const & vec2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                vector_base<T> const & vec3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      assert(cuarma::traits::size(vec1) == cuarma::traits::size(vec2) && bool("Incompatible vector sizes in v1 += v2 @ alpha + v3 @ beta: size(v1) != size(v2)"));
      assert(cuarma::traits::size(vec2) == cuarma::traits::size(vec3) && bool("Incompatible vector sizes in v1 += v2 @ alpha + v3 @ beta: size(v2) != size(v3)"));

      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::avbv_v(vec1,
                                                    vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                    vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::avbv_v(vec1,
                                         vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Assign a constant value to a vector (-range/-slice)
    *
    * @param vec1   The vector to which the value should be assigned
    * @param alpha  The value to be assigned
    * @param up_to_internal_size    Whether 'alpha' should be written to padded memory as well. This is used for setting all entries to zero, including padded memory.
    */
    template<typename T>
    void vector_assign(vector_base<T> & vec1, const T & alpha, bool up_to_internal_size = false)
    {
      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::vector_assign(vec1, alpha, up_to_internal_size);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::vector_assign(vec1, alpha, up_to_internal_size);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Swaps the contents of two vectors, data is copied
    *
    * @param vec1   The first vector (or -range, or -slice)
    * @param vec2   The second vector (or -range, or -slice)
    */
    template<typename T>
    void vector_swap(vector_base<T> & vec1, vector_base<T> & vec2)
    {
      assert(cuarma::traits::size(vec1) == cuarma::traits::size(vec2) && bool("Incompatible vector sizes in vector_swap()"));

      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::vector_swap(vec1, vec2);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::vector_swap(vec1, vec2);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    ///////////////////////// Elementwise operations /////////////



    /** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
    *
    * @param vec1   The result vector (or -range, or -slice)
    * @param proxy  The proxy object holding v2, v3 and the operation
    */
    template<typename T, typename OP>
    void element_op(vector_base<T> & vec1,
                    vector_expression<const vector_base<T>, const vector_base<T>, OP> const & proxy)
    {
      assert(cuarma::traits::size(vec1) == cuarma::traits::size(proxy) && bool("Incompatible vector sizes in element_op()"));

      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::element_op(vec1, proxy);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::element_op(vec1, proxy);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** \cond */

// Helper macro for generating binary element-wise operations such as element_prod(), element_div(), element_pow() without unnecessary code duplication */
#define CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(OPNAME) \
    template<typename T> \
    cuarma::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<op_##OPNAME> > \
    element_##OPNAME(vector_base<T> const & v1, vector_base<T> const & v2) \
    { \
      return cuarma::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<op_##OPNAME> >(v1, v2); \
    } \
\
    template<typename V1, typename V2, typename OP, typename T> \
    cuarma::vector_expression<const vector_expression<const V1, const V2, OP>, const vector_base<T>, op_element_binary<op_##OPNAME> > \
    element_##OPNAME(vector_expression<const V1, const V2, OP> const & proxy, vector_base<T> const & v2) \
    { \
      return cuarma::vector_expression<const vector_expression<const V1, const V2, OP>, const vector_base<T>, op_element_binary<op_##OPNAME> >(proxy, v2); \
    } \
\
    template<typename T, typename V2, typename V3, typename OP> \
    cuarma::vector_expression<const vector_base<T>, const vector_expression<const V2, const V3, OP>, op_element_binary<op_##OPNAME> > \
    element_##OPNAME(vector_base<T> const & v1, vector_expression<const V2, const V3, OP> const & proxy) \
    { \
      return cuarma::vector_expression<const vector_base<T>, const vector_expression<const V2, const V3, OP>, op_element_binary<op_##OPNAME> >(v1, proxy); \
    } \
\
    template<typename V1, typename V2, typename OP1, \
              typename V3, typename V4, typename OP2> \
    cuarma::vector_expression<const vector_expression<const V1, const V2, OP1>, \
                                const vector_expression<const V3, const V4, OP2>, \
                                op_element_binary<op_##OPNAME> > \
    element_##OPNAME(vector_expression<const V1, const V2, OP1> const & proxy1, \
                     vector_expression<const V3, const V4, OP2> const & proxy2) \
    {\
      return cuarma::vector_expression<const vector_expression<const V1, const V2, OP1>, \
                                         const vector_expression<const V3, const V4, OP2>, \
                                         op_element_binary<op_##OPNAME> >(proxy1, proxy2); \
    }

    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(prod)  //for element_prod()
    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(div)   //for element_div()
    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(pow)   //for element_pow()

    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(eq)
    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(neq)
    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(greater)
    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(less)
    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(geq)
    CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(leq)

#undef CUARMA_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS

// Helper macro for generating unary element-wise operations such as element_exp(), element_sin(), etc. without unnecessary code duplication */
#define CUARMA_MAKE_UNARY_ELEMENT_OP(funcname) \
    template<typename T> \
    cuarma::vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<op_##funcname> > \
    element_##funcname(vector_base<T> const & v) \
    { \
      return cuarma::vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<op_##funcname> >(v, v); \
    } \
    template<typename LHS, typename RHS, typename OP> \
    cuarma::vector_expression<const vector_expression<const LHS, const RHS, OP>, \
                                const vector_expression<const LHS, const RHS, OP>, \
                                op_element_unary<op_##funcname> > \
    element_##funcname(vector_expression<const LHS, const RHS, OP> const & proxy) \
    { \
      return cuarma::vector_expression<const vector_expression<const LHS, const RHS, OP>, \
                                         const vector_expression<const LHS, const RHS, OP>, \
                                         op_element_unary<op_##funcname> >(proxy, proxy); \
    } \

    CUARMA_MAKE_UNARY_ELEMENT_OP(abs)
    CUARMA_MAKE_UNARY_ELEMENT_OP(acos)
    CUARMA_MAKE_UNARY_ELEMENT_OP(asin)
    CUARMA_MAKE_UNARY_ELEMENT_OP(atan)
    CUARMA_MAKE_UNARY_ELEMENT_OP(ceil)
    CUARMA_MAKE_UNARY_ELEMENT_OP(cos)
    CUARMA_MAKE_UNARY_ELEMENT_OP(cosh)
    CUARMA_MAKE_UNARY_ELEMENT_OP(exp)
    CUARMA_MAKE_UNARY_ELEMENT_OP(fabs)
    CUARMA_MAKE_UNARY_ELEMENT_OP(floor)
    CUARMA_MAKE_UNARY_ELEMENT_OP(log)
    CUARMA_MAKE_UNARY_ELEMENT_OP(log10)
    CUARMA_MAKE_UNARY_ELEMENT_OP(sin)
    CUARMA_MAKE_UNARY_ELEMENT_OP(sinh)
    CUARMA_MAKE_UNARY_ELEMENT_OP(sqrt)
    CUARMA_MAKE_UNARY_ELEMENT_OP(tan)
    CUARMA_MAKE_UNARY_ELEMENT_OP(tanh)

#undef CUARMA_MAKE_UNARY_ELEMENT_OP

    /** \endcond */

    ///////////////////////// Norms and inner product ///////////////////


    //implementation of inner product:
    //namespace {

    /** @brief Computes the inner product of two vectors - dispatcher interface
     *
     * @param vec1 The first vector
     * @param vec2 The second vector
     * @param result The result scalar (on the gpu)
     */
    template<typename T>
    void inner_prod_impl(vector_base<T> const & vec1,
                         vector_base<T> const & vec2,
                         scalar<T> & result)
    {
      assert( vec1.size() == vec2.size() && bool("Size mismatch") );

      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::inner_prod_impl(vec1, vec2, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::inner_prod_impl(vec1, vec2, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    // vector expression on lhs
    template<typename LHS, typename RHS, typename OP, typename T>
    void inner_prod_impl(cuarma::vector_expression<LHS, RHS, OP> const & vec1,
                         vector_base<T> const & vec2,
                         scalar<T> & result)
    {
      cuarma::vector<T> temp = vec1;
      inner_prod_impl(temp, vec2, result);
    }


    // vector expression on rhs
    template<typename T, typename LHS, typename RHS, typename OP>
    void inner_prod_impl(vector_base<T> const & vec1,
                         cuarma::vector_expression<LHS, RHS, OP> const & vec2,
                         scalar<T> & result)
    {
      cuarma::vector<T> temp = vec2;
      inner_prod_impl(vec1, temp, result);
    }


    // vector expression on lhs and rhs
    template<typename LHS1, typename RHS1, typename OP1,
              typename LHS2, typename RHS2, typename OP2, typename T>
    void inner_prod_impl(cuarma::vector_expression<LHS1, RHS1, OP1> const & vec1,
                         cuarma::vector_expression<LHS2, RHS2, OP2> const & vec2,
                         scalar<T> & result)
    {
      cuarma::vector<T> temp1 = vec1;
      cuarma::vector<T> temp2 = vec2;
      inner_prod_impl(temp1, temp2, result);
    }




    /** @brief Computes the inner product of two vectors with the final reduction step on the CPU - dispatcher interface
     *
     * @param vec1 The first vector
     * @param vec2 The second vector
     * @param result The result scalar (on the gpu)
     */
    template<typename T>
    void inner_prod_cpu(vector_base<T> const & vec1,
                        vector_base<T> const & vec2,
                        T & result)
    {
      assert( vec1.size() == vec2.size() && bool("Size mismatch") );

      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::inner_prod_impl(vec1, vec2, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::inner_prod_cpu(vec1, vec2, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    // vector expression on lhs
    template<typename LHS, typename RHS, typename OP, typename T>
    void inner_prod_cpu(cuarma::vector_expression<LHS, RHS, OP> const & vec1,
                        vector_base<T> const & vec2,
                        T & result)
    {
      cuarma::vector<T> temp = vec1;
      inner_prod_cpu(temp, vec2, result);
    }


    // vector expression on rhs
    template<typename T, typename LHS, typename RHS, typename OP>
    void inner_prod_cpu(vector_base<T> const & vec1,
                        cuarma::vector_expression<LHS, RHS, OP> const & vec2,
                        T & result)
    {
      cuarma::vector<T> temp = vec2;
      inner_prod_cpu(vec1, temp, result);
    }


    // vector expression on lhs and rhs
    template<typename LHS1, typename RHS1, typename OP1,
              typename LHS2, typename RHS2, typename OP2, typename S3>
    void inner_prod_cpu(cuarma::vector_expression<LHS1, RHS1, OP1> const & vec1,
                        cuarma::vector_expression<LHS2, RHS2, OP2> const & vec2,
                        S3 & result)
    {
      cuarma::vector<S3> temp1 = vec1;
      cuarma::vector<S3> temp2 = vec2;
      inner_prod_cpu(temp1, temp2, result);
    }



    /** @brief Computes the inner products <x, y1>, <x, y2>, ..., <x, y_N> and writes the result to a (sub-)vector
     *
     * @param x       The common vector
     * @param y_tuple A collection of vector, all of the same size.
     * @param result  The result scalar (on the gpu). Needs to match the number of elements in y_tuple
     */
    template<typename T>
    void inner_prod_impl(vector_base<T> const & x,
                         vector_tuple<T> const & y_tuple,
                         vector_base<T> & result)
    {
      assert( x.size() == y_tuple.const_at(0).size() && bool("Size mismatch") );
      assert( result.size() == y_tuple.const_size() && bool("Number of elements does not match result size") );

      switch (cuarma::traits::handle(x).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::inner_prod_impl(x, y_tuple, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::inner_prod_impl(x, y_tuple, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Computes the l^1-norm of a vector - dispatcher interface
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_1_impl(vector_base<T> const & vec,
                     scalar<T> & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::norm_1_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::norm_1_impl(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Computes the l^1-norm of a vector - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_1_impl(cuarma::vector_expression<LHS, RHS, OP> const & vec,
                     S2 & result)
    {
      cuarma::vector<typename cuarma::result_of::cpu_value_type<S2>::type> temp = vec;
      norm_1_impl(temp, result);
    }



    /** @brief Computes the l^1-norm of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_1_cpu(vector_base<T> const & vec,
                    T & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::norm_1_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::norm_1_cpu(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the l^1-norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_1_cpu(cuarma::vector_expression<LHS, RHS, OP> const & vec,
                    S2 & result)
    {
      cuarma::vector<typename cuarma::result_of::cpu_value_type<LHS>::type> temp = vec;
      norm_1_cpu(temp, result);
    }




    /** @brief Computes the l^2-norm of a vector - dispatcher interface
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_2_impl(vector_base<T> const & vec,
                     scalar<T> & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::norm_2_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::norm_2_impl(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the l^2-norm of a vector - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename T>
    void norm_2_impl(cuarma::vector_expression<LHS, RHS, OP> const & vec,
                     scalar<T> & result)
    {
      cuarma::vector<T> temp = vec;
      norm_2_impl(temp, result);
    }


    /** @brief Computes the l^2-norm of a vector with final reduction on the CPU - dispatcher interface
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_2_cpu(vector_base<T> const & vec,
                    T & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::norm_2_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::norm_2_cpu(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the l^2-norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_2_cpu(cuarma::vector_expression<LHS, RHS, OP> const & vec,
                    S2 & result)
    {
      cuarma::vector<typename cuarma::result_of::cpu_value_type<LHS>::type> temp = vec;
      norm_2_cpu(temp, result);
    }




    /** @brief Computes the supremum-norm of a vector
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_inf_impl(vector_base<T> const & vec,
                       scalar<T> & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::norm_inf_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::norm_inf_impl(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename T>
    void norm_inf_impl(cuarma::vector_expression<LHS, RHS, OP> const & vec,
                       scalar<T> & result)
    {
      cuarma::vector<T> temp = vec;
      norm_inf_impl(temp, result);
    }


    /** @brief Computes the supremum-norm of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_inf_cpu(vector_base<T> const & vec,
                      T & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::norm_inf_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::norm_inf_cpu(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_inf_cpu(cuarma::vector_expression<LHS, RHS, OP> const & vec,
                      S2 & result)
    {
      cuarma::vector<typename cuarma::result_of::cpu_value_type<LHS>::type> temp = vec;
      norm_inf_cpu(temp, result);
    }


    //This function should return a CPU scalar, otherwise statements like
    // arma_rhs[index_norm_inf(arma_rhs)]
    // are ambiguous
    /** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
    *
    * @param vec The vector
    * @return The result. Note that the result must be a CPU scalar
    */
    template<typename T>
    arma_size_t index_norm_inf(vector_base<T> const & vec)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          return cuarma::blas::host_based::index_norm_inf(vec);

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          return cuarma::blas::cuda::index_norm_inf(vec);
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    */
    template<typename LHS, typename RHS, typename OP>
    arma_size_t index_norm_inf(cuarma::vector_expression<LHS, RHS, OP> const & vec)
    {
      cuarma::vector<typename cuarma::result_of::cpu_value_type<LHS>::type> temp = vec;
      return index_norm_inf(temp);
    }

///////////////////

    /** @brief Computes the maximum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename NumericT>
    void max_impl(vector_base<NumericT> const & vec, cuarma::scalar<NumericT> & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::max_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::max_impl(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename NumericT>
    void max_impl(cuarma::vector_expression<LHS, RHS, OP> const & vec, cuarma::scalar<NumericT> & result)
    {
      cuarma::vector<NumericT> temp = vec;
      max_impl(temp, result);
    }


    /** @brief Computes the maximum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void max_cpu(vector_base<T> const & vec, T & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::max_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::max_cpu(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void max_cpu(cuarma::vector_expression<LHS, RHS, OP> const & vec, S2 & result)
    {
      cuarma::vector<typename cuarma::result_of::cpu_value_type<LHS>::type> temp = vec;
      max_cpu(temp, result);
    }

///////////////////

    /** @brief Computes the minimum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename NumericT>
    void min_impl(vector_base<NumericT> const & vec, cuarma::scalar<NumericT> & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::min_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::min_impl(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename NumericT>
    void min_impl(cuarma::vector_expression<LHS, RHS, OP> const & vec, cuarma::scalar<NumericT> & result)
    {
      cuarma::vector<NumericT> temp = vec;
      min_impl(temp, result);
    }


    /** @brief Computes the minimum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void min_cpu(vector_base<T> const & vec, T & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::min_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::min_cpu(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void min_cpu(cuarma::vector_expression<LHS, RHS, OP> const & vec, S2 & result)
    {
      cuarma::vector<typename cuarma::result_of::cpu_value_type<LHS>::type> temp = vec;
      min_cpu(temp, result);
    }

///////////////////

    /** @brief Computes the sum of a vector with final reduction on the device (GPU, etc.)
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename NumericT>
    void sum_impl(vector_base<NumericT> const & vec, cuarma::scalar<NumericT> & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::sum_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::sum_impl(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the sum of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename NumericT>
    void sum_impl(cuarma::vector_expression<LHS, RHS, OP> const & vec, cuarma::scalar<NumericT> & result)
    {
      cuarma::vector<NumericT> temp = vec;
      sum_impl(temp, result);
    }


    /** @brief Computes the sum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void sum_cpu(vector_base<T> const & vec, T & result)
    {
      switch (cuarma::traits::handle(vec).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::sum_impl(vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::sum_cpu(vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the sum of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void sum_cpu(cuarma::vector_expression<LHS, RHS, OP> const & vec, S2 & result)
    {
      cuarma::vector<typename cuarma::result_of::cpu_value_type<LHS>::type> temp = vec;
      sum_cpu(temp, result);
    }





    /** @brief Computes a plane rotation of two vectors.
    *
    * Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
    *
    * @param vec1   The first vector
    * @param vec2   The second vector
    * @param alpha  The first transformation coefficient (CPU scalar)
    * @param beta   The second transformation coefficient (CPU scalar)
    */
    template<typename T>
    void plane_rotation(vector_base<T> & vec1,
                        vector_base<T> & vec2,
                        T alpha, T beta)
    {
      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::plane_rotation(vec1, vec2, alpha, beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::plane_rotation(vec1, vec2, alpha, beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief This function implements an inclusive scan.
    *
    * Given an element vector (x_0, x_1, ..., x_{n-1}),
    * this routine computes (x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-1})
    *
    * The two vectors either need to be the same (in-place), or reside in distinct memory regions.
    * Partial overlaps of vec1 and vec2 are not allowed.
    *
    * @param vec1       Input vector.
    * @param vec2       The output vector.
    */
    template<typename NumericT>
    void inclusive_scan(vector_base<NumericT> & vec1,
                        vector_base<NumericT> & vec2)
    {
      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::inclusive_scan(vec1, vec2);
          break;
  

  #ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::inclusive_scan(vec1, vec2);
          break;
  #endif

        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Implements an in-place inclusive scan of a vector.
    *
    * Given an input element vector (x_0, x_1, ..., x_{n-1}),
    * this routine overwrites the vector with (x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-1})
    */
    template<typename NumericT>
    void inclusive_scan(vector_base<NumericT> & vec)
    {
      inclusive_scan(vec, vec);
    }

    /** @brief This function implements an exclusive scan.
    *
    * Given an element vector (x_0, x_1, ..., x_{n-1}),
    * this routine computes (0, x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-2})
    *
    * The two vectors either need to be the same (in-place), or reside in distinct memory regions.
    * Partial overlaps of vec1 and vec2 are not allowed.
    *
    * @param vec1       Input vector.
    * @param vec2       The output vector.
    */
    template<typename NumericT>
    void exclusive_scan(vector_base<NumericT> & vec1,
                        vector_base<NumericT> & vec2)
    {
      switch (cuarma::traits::handle(vec1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::exclusive_scan(vec1, vec2);
          break;
  

  #ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::exclusive_scan(vec1, vec2);
          break;
  #endif

        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Inplace exclusive scan of a vector
    *
    * Given an element vector (x_0, x_1, ..., x_{n-1}),
    * this routine overwrites the input vector with (0, x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-2})
    */
    template<typename NumericT>
    void exclusive_scan(vector_base<NumericT> & vec)
    {
      exclusive_scan(vec, vec);
    }
  } //namespace blas

  template<typename T, typename LHS, typename RHS, typename OP>
  vector_base<T> & operator += (vector_base<T> & v1, const vector_expression<const LHS, const RHS, OP> & proxy)
  {
    assert( (cuarma::traits::size(proxy) == v1.size()) && bool("Incompatible vector sizes!"));
    assert( (v1.size() > 0) && bool("Vector not yet initialized!") );

    blas::detail::op_executor<vector_base<T>, op_inplace_add, vector_expression<const LHS, const RHS, OP> >::apply(v1, proxy);

    return v1;
  }

  template<typename T, typename LHS, typename RHS, typename OP>
  vector_base<T> & operator -= (vector_base<T> & v1, const vector_expression<const LHS, const RHS, OP> & proxy)
  {
    assert( (cuarma::traits::size(proxy) == v1.size()) && bool("Incompatible vector sizes!"));
    assert( (v1.size() > 0) && bool("Vector not yet initialized!") );

    blas::detail::op_executor<vector_base<T>, op_inplace_sub, vector_expression<const LHS, const RHS, OP> >::apply(v1, proxy);

    return v1;
  }

} //namespace cuarma
