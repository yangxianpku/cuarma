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

/** @file bicgstab.hpp
 *  @encoding:UTF-8 文档编码
    @brief The stabilized bi-conjugate gradient method is implemented here
*/

#include <vector>
#include <cmath>
#include <numeric>

#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/inner_prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/traits/clear.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/context.hpp"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/blas/iterative_operations.hpp"

namespace cuarma
{
namespace blas
{

/** @brief A tag for the stabilized Bi-conjugate gradient solver. Used for supplying solver parameters and for dispatching the solve() function
*/
class bicgstab_tag
{
public:
  /** @brief The constructor
  *
  * @param tol              Relative tolerance for the residual (solver quits if ||r|| < tol * ||r_initial||)
  * @param max_iters        The maximum number of iterations
  * @param max_iters_before_restart   The maximum number of iterations before BiCGStab is reinitialized (to avoid accumulation of round-off errors)
  */
  bicgstab_tag(double tol = 1e-8, arma_size_t max_iters = 400, arma_size_t max_iters_before_restart = 200)
    : tol_(tol), abs_tol_(0), iterations_(max_iters), iterations_before_restart_(max_iters_before_restart) {}

  /** @brief Returns the relative tolerance */
  double tolerance() const { return tol_; }

  /** @brief Returns the absolute tolerance */
  double abs_tolerance() const { return abs_tol_; }
  /** @brief Sets the absolute tolerance */
  void abs_tolerance(double new_tol) { if (new_tol >= 0) abs_tol_ = new_tol; }

  /** @brief Returns the maximum number of iterations */
  arma_size_t max_iterations() const { return iterations_; }
  /** @brief Returns the maximum number of iterations before a restart*/
  arma_size_t max_iterations_before_restart() const { return iterations_before_restart_; }

  /** @brief Return the number of solver iterations: */
  arma_size_t iters() const { return iters_taken_; }
  void iters(arma_size_t i) const { iters_taken_ = i; }

  /** @brief Returns the estimated relative error at the end of the solver run */
  double error() const { return last_error_; }
  /** @brief Sets the estimated relative error at the end of the solver run */
  void error(double e) const { last_error_ = e; }

private:
  double tol_;
  double abs_tol_;
  arma_size_t iterations_;
  arma_size_t iterations_before_restart_;

  //return values from solver
  mutable arma_size_t iters_taken_;
  mutable double last_error_;
};



namespace detail
{
  /** @brief Implementation of a pipelined stabilized Bi-conjugate gradient solver */
  template<typename MatrixT, typename NumericT>
  cuarma::vector<NumericT> pipelined_solve(MatrixT const & A, //MatrixType const & A,
                                             cuarma::vector_base<NumericT> const & rhs,
                                             bicgstab_tag const & tag,
                                             cuarma::blas::no_precond,
                                             bool (*monitor)(cuarma::vector<NumericT> const &, NumericT, void*) = NULL,
                                             void *monitor_data = NULL)
  {
    cuarma::vector<NumericT> result = cuarma::zero_vector<NumericT>(rhs.size(), cuarma::traits::context(rhs));

    cuarma::vector<NumericT> residual = rhs;
    cuarma::vector<NumericT> p = rhs;
    cuarma::vector<NumericT> r0star = rhs;
    cuarma::vector<NumericT> Ap = rhs;
    cuarma::vector<NumericT> s  = rhs;
    cuarma::vector<NumericT> As = rhs;

    // Layout of temporary buffer:
    //  chunk 0: <residual, r_0^*>
    //  chunk 1: <As, As>
    //  chunk 2: <As, s>
    //  chunk 3: <Ap, r_0^*>
    //  chunk 4: <As, r_0^*>
    //  chunk 5: <s, s>
    arma_size_t buffer_size_per_vector = 256;
    arma_size_t num_buffer_chunks = 6;
    cuarma::vector<NumericT> inner_prod_buffer = cuarma::zero_vector<NumericT>(num_buffer_chunks*buffer_size_per_vector, cuarma::traits::context(rhs)); // temporary buffer
    std::vector<NumericT>      host_inner_prod_buffer(inner_prod_buffer.size());

    NumericT norm_rhs_host = cuarma::blas::norm_2(residual);
    NumericT beta;
    NumericT alpha;
    NumericT omega;
    NumericT residual_norm = norm_rhs_host;
    inner_prod_buffer[0] = norm_rhs_host * norm_rhs_host;

    NumericT  r_dot_r0 = 0;
    NumericT As_dot_As = 0;
    NumericT As_dot_s  = 0;
    NumericT Ap_dot_r0 = 0;
    NumericT As_dot_r0 = 0;
    NumericT  s_dot_s  = 0;

    if (norm_rhs_host <= tag.abs_tolerance()) //solution is zero if RHS norm is zero
      return result;

    for (arma_size_t i = 0; i < tag.max_iterations(); ++i)
    {
      tag.iters(i+1);
      // Ap = A*p_j
      // Ap_dot_r0 = <Ap, r_0^*>
      cuarma::blas::pipelined_bicgstab_prod(A, p, Ap, r0star, inner_prod_buffer, buffer_size_per_vector, 3*buffer_size_per_vector);

      //////// first (weak) synchronization point ////

      ///// method 1: compute alpha on host:
      //
      //// we only need the second chunk of the buffer for computing Ap_dot_r0:
      //cuarma::fast_copy(inner_prod_buffer.begin(), inner_prod_buffer.end(), host_inner_prod_buffer.begin());
      //Ap_dot_r0 = std::accumulate(host_inner_prod_buffer.begin() +     buffer_size_per_vector, host_inner_prod_buffer.begin() + 2 * buffer_size_per_vector, ScalarType(0));

      //alpha = residual_dot_r0 / Ap_dot_r0;

      //// s_j = r_j - alpha_j q_j
      //s = residual - alpha * Ap;

      ///// method 2: compute alpha on device:
      // s = r - alpha * Ap
      // <s, s> first stage
      // dump alpha at end of inner_prod_buffer
      cuarma::blas::pipelined_bicgstab_update_s(s, residual, Ap, inner_prod_buffer, buffer_size_per_vector, 5*buffer_size_per_vector);

      // As = A*s_j
      // As_dot_As = <As, As>
      // As_dot_s  = <As, s>
      // As_dot_r0 = <As, r_0^*>
      cuarma::blas::pipelined_bicgstab_prod(A, s, As, r0star, inner_prod_buffer, buffer_size_per_vector, 4*buffer_size_per_vector);

      //////// second (strong) synchronization point ////

      cuarma::fast_copy(inner_prod_buffer.begin(), inner_prod_buffer.end(), host_inner_prod_buffer.begin());

      typedef typename std::vector<NumericT>::difference_type       difference_type;

       r_dot_r0 = std::accumulate(host_inner_prod_buffer.begin(),                                               host_inner_prod_buffer.begin() + difference_type(    buffer_size_per_vector), NumericT(0));
      As_dot_As = std::accumulate(host_inner_prod_buffer.begin() + difference_type(    buffer_size_per_vector), host_inner_prod_buffer.begin() + difference_type(2 * buffer_size_per_vector), NumericT(0));
      As_dot_s  = std::accumulate(host_inner_prod_buffer.begin() + difference_type(2 * buffer_size_per_vector), host_inner_prod_buffer.begin() + difference_type(3 * buffer_size_per_vector), NumericT(0));
      Ap_dot_r0 = std::accumulate(host_inner_prod_buffer.begin() + difference_type(3 * buffer_size_per_vector), host_inner_prod_buffer.begin() + difference_type(4 * buffer_size_per_vector), NumericT(0));
      As_dot_r0 = std::accumulate(host_inner_prod_buffer.begin() + difference_type(4 * buffer_size_per_vector), host_inner_prod_buffer.begin() + difference_type(5 * buffer_size_per_vector), NumericT(0));
       s_dot_s  = std::accumulate(host_inner_prod_buffer.begin() + difference_type(5 * buffer_size_per_vector), host_inner_prod_buffer.begin() + difference_type(6 * buffer_size_per_vector), NumericT(0));

      alpha =   r_dot_r0 / Ap_dot_r0;
      beta  = - As_dot_r0 / Ap_dot_r0;
      omega =   As_dot_s  / As_dot_As;

      residual_norm = std::sqrt(s_dot_s - NumericT(2.0) * omega * As_dot_s + omega * omega *  As_dot_As);
      if (monitor && monitor(result, std::fabs(residual_norm / norm_rhs_host), monitor_data))
        break;
      if (std::fabs(residual_norm / norm_rhs_host) < tag.tolerance() || residual_norm < tag.abs_tolerance())
        break;

      // x_{j+1} = x_j + alpha * p_j + omega * s_j
      // r_{j+1} = s_j - omega * t_j
      // p_{j+1} = r_{j+1} + beta * (p_j - omega * q_j)
      // and compute first stage of r_dot_r0 = <r_{j+1}, r_o^*> for use in next iteration
       cuarma::blas::pipelined_bicgstab_vector_update(result, alpha, p, omega, s,
                                                          residual, As, beta, Ap, r0star, inner_prod_buffer, buffer_size_per_vector);
    }

    //store last error estimate:
    tag.error(residual_norm / norm_rhs_host);

    return result;
  }

  /** @brief Overload for the pipelined CG implementation for the cuarma sparse matrix types */
  template<typename NumericT>
  cuarma::vector<NumericT> solve_impl(cuarma::compressed_matrix<NumericT> const & A,
                                        cuarma::vector<NumericT> const & rhs,bicgstab_tag const & tag,cuarma::blas::no_precond,
                                        bool (*monitor)(cuarma::vector<NumericT> const &, NumericT, void*) = NULL, void *monitor_data = NULL)
  {
    return pipelined_solve(A, rhs, tag, cuarma::blas::no_precond(), monitor, monitor_data);
  }


  /** @brief Overload for the pipelined CG implementation for the cuarma sparse matrix types */
  template<typename NumericT>
  cuarma::vector<NumericT> solve_impl(cuarma::coordinate_matrix<NumericT> const & A,
                                        cuarma::vector<NumericT> const & rhs, bicgstab_tag const & tag,cuarma::blas::no_precond,
                                        bool (*monitor)(cuarma::vector<NumericT> const &, NumericT, void*) = NULL, void *monitor_data = NULL)
  {
    return detail::pipelined_solve(A, rhs, tag, cuarma::blas::no_precond(), monitor, monitor_data);
  }



  /** @brief Overload for the pipelined CG implementation for the cuarma sparse matrix types */
  template<typename NumericT>
  cuarma::vector<NumericT> solve_impl(cuarma::ell_matrix<NumericT> const & A,
                                        cuarma::vector<NumericT> const & rhs,
                                        bicgstab_tag const & tag,
                                        cuarma::blas::no_precond,
                                        bool (*monitor)(cuarma::vector<NumericT> const &, NumericT, void*) = NULL,
                                        void *monitor_data = NULL)
  {
    return detail::pipelined_solve(A, rhs, tag, cuarma::blas::no_precond(), monitor, monitor_data);
  }


  /** @brief Overload for the pipelined CG implementation for the cuarma sparse matrix types */
  template<typename NumericT>
  cuarma::vector<NumericT> solve_impl(cuarma::sliced_ell_matrix<NumericT> const & A,
                                        cuarma::vector<NumericT> const & rhs,
                                        bicgstab_tag const & tag,
                                        cuarma::blas::no_precond,
                                        bool (*monitor)(cuarma::vector<NumericT> const &, NumericT, void*) = NULL,
                                        void *monitor_data = NULL)
  {
    return detail::pipelined_solve(A, rhs, tag, cuarma::blas::no_precond(), monitor, monitor_data);
  }


  /** @brief Overload for the pipelined CG implementation for the cuarma sparse matrix types */
  template<typename NumericT>
  cuarma::vector<NumericT> solve_impl(cuarma::hyb_matrix<NumericT> const & A,
                                        cuarma::vector<NumericT> const & rhs,
                                        bicgstab_tag const & tag,
                                        cuarma::blas::no_precond,
                                        bool (*monitor)(cuarma::vector<NumericT> const &, NumericT, void*) = NULL,
                                        void *monitor_data = NULL)
  {
    return detail::pipelined_solve(A, rhs, tag, cuarma::blas::no_precond(), monitor, monitor_data);
  }


  /** @brief Implementation of the unpreconditioned stabilized Bi-conjugate gradient solver
  *
  * Following the description in "Iterative Methods for Sparse Linear Systems" by Y. Saad
  *
  * @param matrix       The system matrix
  * @param rhs          The load vector
  * @param tag          Solver configuration tag
  * @param monitor      A callback routine which is called at each GMRES restart
  * @param monitor_data Data pointer to be passed to the callback routine to pass on user-specific data
  * @return The result vector
  */
  template<typename MatrixT, typename VectorT>
  VectorT solve_impl(MatrixT const & matrix,
                     VectorT const & rhs,
                     bicgstab_tag const & tag,
                     cuarma::blas::no_precond,
                     bool (*monitor)(VectorT const &, typename cuarma::result_of::cpu_value_type<typename cuarma::result_of::value_type<VectorT>::type>::type, void*) = NULL,
                     void *monitor_data = NULL)
  {
    typedef typename cuarma::result_of::value_type<VectorT>::type            NumericType;
    typedef typename cuarma::result_of::cpu_value_type<NumericType>::type    CPU_NumericType;
    VectorT result = rhs;
    cuarma::traits::clear(result);

    VectorT residual = rhs;
    VectorT p = rhs;
    VectorT r0star = rhs;
    VectorT tmp0 = rhs;
    VectorT tmp1 = rhs;
    VectorT s = rhs;

    CPU_NumericType norm_rhs_host = cuarma::blas::norm_2(residual);
    CPU_NumericType ip_rr0star = norm_rhs_host * norm_rhs_host;
    CPU_NumericType beta;
    CPU_NumericType alpha;
    CPU_NumericType omega;
    //ScalarType inner_prod_temp; //temporary variable for inner product computation
    CPU_NumericType new_ip_rr0star = 0;
    CPU_NumericType residual_norm = norm_rhs_host;

    if (norm_rhs_host <= tag.abs_tolerance()) //solution is zero if RHS norm is zero
      return result;

    bool restart_flag = true;
    arma_size_t last_restart = 0;
    for (arma_size_t i = 0; i < tag.max_iterations(); ++i)
    {
      if (restart_flag)
      {
        residual = cuarma::blas::prod(matrix, result);
        residual = rhs - residual;
        p = residual;
        r0star = residual;
        ip_rr0star = cuarma::blas::norm_2(residual);
        ip_rr0star *= ip_rr0star;
        restart_flag = false;
        last_restart = i;
      }

      tag.iters(i+1);
      tmp0 = cuarma::blas::prod(matrix, p);
      alpha = ip_rr0star / cuarma::blas::inner_prod(tmp0, r0star);

      s = residual - alpha*tmp0;

      tmp1 = cuarma::blas::prod(matrix, s);
      CPU_NumericType norm_tmp1 = cuarma::blas::norm_2(tmp1);
      omega = cuarma::blas::inner_prod(tmp1, s) / (norm_tmp1 * norm_tmp1);

      result += alpha * p + omega * s;
      residual = s - omega * tmp1;

      new_ip_rr0star = cuarma::blas::inner_prod(residual, r0star);
      residual_norm = cuarma::blas::norm_2(residual);
      if (monitor && monitor(result, std::fabs(residual_norm / norm_rhs_host), monitor_data))
        break;
      if (std::fabs(residual_norm / norm_rhs_host) < tag.tolerance() || residual_norm < tag.abs_tolerance())
        break;

      beta = new_ip_rr0star / ip_rr0star * alpha/omega;
      ip_rr0star = new_ip_rr0star;

      if (    (ip_rr0star <= 0 && ip_rr0star >= 0)
           || (omega <= 0 && omega >= 0)
           || (i - last_restart > tag.max_iterations_before_restart())
         ) //search direction degenerate. A restart might help
        restart_flag = true;

      // Execution of
      //  p = residual + beta * (p - omega*tmp0);
      // without introducing temporary vectors:
      p -= omega * tmp0;
      p = residual + beta * p;
    }

    //store last error estimate:
    tag.error(residual_norm / norm_rhs_host);

    return result;
  }


  /** @brief Implementation of the preconditioned stabilized Bi-conjugate gradient solver
  *
  * Following the description of the unpreconditioned case in "Iterative Methods for Sparse Linear Systems" by Y. Saad
  *
  * @param matrix       The system matrix
  * @param rhs          The load vector
  * @param tag          Solver configuration tag
  * @param precond      A preconditioner. Precondition operation is done via member function apply()
  * @param monitor      A callback routine which is called at each GMRES restart
  * @param monitor_data Data pointer to be passed to the callback routine to pass on user-specific data
  * @return The result vector
  */
  template<typename MatrixT, typename VectorT, typename PreconditionerT>
  VectorT solve_impl(MatrixT const & matrix,
                     VectorT const & rhs,
                     bicgstab_tag const & tag,
                     PreconditionerT const & precond,
                     bool (*monitor)(VectorT const &, typename cuarma::result_of::cpu_value_type<typename cuarma::result_of::value_type<VectorT>::type>::type, void*) = NULL,
                     void *monitor_data = NULL)
  {
    typedef typename cuarma::result_of::value_type<VectorT>::type            NumericType;
    typedef typename cuarma::result_of::cpu_value_type<NumericType>::type    CPU_NumericType;
    VectorT result = rhs;
    cuarma::traits::clear(result);

    VectorT residual = rhs;
    VectorT r0star = residual;  //can be chosen arbitrarily in fact
    VectorT tmp0 = rhs;
    VectorT tmp1 = rhs;
    VectorT s = rhs;

    VectorT p = residual;

    CPU_NumericType ip_rr0star = cuarma::blas::norm_2(residual);
    CPU_NumericType norm_rhs_host = cuarma::blas::norm_2(residual);
    CPU_NumericType beta;
    CPU_NumericType alpha;
    CPU_NumericType omega;
    CPU_NumericType new_ip_rr0star = 0;
    CPU_NumericType residual_norm = norm_rhs_host;

    if (norm_rhs_host <= tag.abs_tolerance()) //solution is zero if RHS norm is zero
      return result;

    bool restart_flag = true;
    arma_size_t last_restart = 0;
    for (unsigned int i = 0; i < tag.max_iterations(); ++i)
    {
      if (restart_flag)
      {
        residual = cuarma::blas::prod(matrix, result);
        residual = rhs - residual;
        precond.apply(residual);
        p = residual;
        r0star = residual;
        ip_rr0star = cuarma::blas::norm_2(residual);
        ip_rr0star *= ip_rr0star;
        restart_flag = false;
        last_restart = i;
      }

      tag.iters(i+1);
      tmp0 = cuarma::blas::prod(matrix, p);
      precond.apply(tmp0);
      alpha = ip_rr0star / cuarma::blas::inner_prod(tmp0, r0star);

      s = residual - alpha*tmp0;

      tmp1 = cuarma::blas::prod(matrix, s);
      precond.apply(tmp1);
      CPU_NumericType norm_tmp1 = cuarma::blas::norm_2(tmp1);
      omega = cuarma::blas::inner_prod(tmp1, s) / (norm_tmp1 * norm_tmp1);

      result += alpha * p + omega * s;
      residual = s - omega * tmp1;

      residual_norm = cuarma::blas::norm_2(residual);
      if (monitor && monitor(result, std::fabs(residual_norm / norm_rhs_host), monitor_data))
        break;
      if (residual_norm / norm_rhs_host < tag.tolerance() || residual_norm < tag.abs_tolerance())
        break;

      new_ip_rr0star = cuarma::blas::inner_prod(residual, r0star);

      beta = new_ip_rr0star / ip_rr0star * alpha/omega;
      ip_rr0star = new_ip_rr0star;

      if ( (ip_rr0star >= 0 && ip_rr0star <= 0) || (omega >=0 && omega <= 0) || i - last_restart > tag.max_iterations_before_restart()) //search direction degenerate. A restart might help
        restart_flag = true;

      // Execution of
      //  p = residual + beta * (p - omega*tmp0);
      // without introducing temporary vectors:
      p -= omega * tmp0;
      p = residual + beta * p;

      //std::cout << "Rel. Residual in current step: " << std::sqrt(std::fabs(cuarma::blas::inner_prod(residual, residual) / norm_rhs_host)) << std::endl;
    }

    //store last error estimate:
    tag.error(residual_norm / norm_rhs_host);

    return result;
  }

}



template<typename MatrixT, typename VectorT, typename PreconditionerT>
VectorT solve(MatrixT const & matrix, VectorT const & rhs, bicgstab_tag const & tag, PreconditionerT const & precond)
{
  return detail::solve_impl(matrix, rhs, tag, precond);
}


/** @brief Convenience overload for calling the preconditioned BiCGStab solver using types from the C++ STL.
  *
  * A std::vector<std::map<T, U> > matrix is convenient for e.g. finite element assembly.
  * It is not the fastest option for setting up a system, but often it is fast enough - particularly for just trying things out.
  */
template<typename IndexT, typename NumericT, typename PreconditionerT>
std::vector<NumericT> solve(std::vector< std::map<IndexT, NumericT> > const & A, std::vector<NumericT> const & rhs, bicgstab_tag const & tag, PreconditionerT const & precond)
{
  cuarma::compressed_matrix<NumericT> arma_A;
  cuarma::copy(A, arma_A);

  cuarma::vector<NumericT> arma_rhs(rhs.size());
  cuarma::copy(rhs, arma_rhs);

  cuarma::vector<NumericT> arma_result = solve(arma_A, arma_rhs, tag, precond);

  std::vector<NumericT> result(arma_result.size());
  cuarma::copy(arma_result, result);
  return result;
}

/** @brief Entry point for the unpreconditioned BiCGStab method.
 *
 *  @param matrix    The system matrix
 *  @param rhs       Right hand side vector (load vector)
 *  @param tag       A BiCGStab tag providing relative tolerances, etc.
 */
template<typename MatrixT, typename VectorT>
VectorT solve(MatrixT const & matrix, VectorT const & rhs, bicgstab_tag const & tag)
{
  return solve(matrix, rhs, tag, cuarma::blas::no_precond());
}



template<typename VectorT>
class bicgstab_solver
{
public:
  typedef typename cuarma::result_of::cpu_value_type<VectorT>::type   numeric_type;

  bicgstab_solver(bicgstab_tag const & tag) : tag_(tag), monitor_callback_(NULL), user_data_(NULL) {}

  template<typename MatrixT, typename PreconditionerT>
  VectorT operator()(MatrixT const & A, VectorT const & b, PreconditionerT const & precond) const
  {
    if (cuarma::traits::size(init_guess_) > 0) // take initial guess into account
    {
      VectorT mod_rhs = cuarma::blas::prod(A, init_guess_);
      mod_rhs = b - mod_rhs;
      VectorT y = detail::solve_impl(A, mod_rhs, tag_, precond, monitor_callback_, user_data_);
      return init_guess_ + y;
    }
    return detail::solve_impl(A, b, tag_, precond, monitor_callback_, user_data_);
  }


  template<typename MatrixT>
  VectorT operator()(MatrixT const & A, VectorT const & b) const
  {
    return operator()(A, b, cuarma::blas::no_precond());
  }

  /** @brief Specifies an initial guess for the iterative solver.
    *
    * An iterative solver for Ax = b with initial guess x_0 is equivalent to an iterative solver for Ay = b' := b - Ax_0, where x = x_0 + y.
    */
  void set_initial_guess(VectorT const & x) { init_guess_ = x; }

  /** @brief Sets a monitor function pointer to be called in each iteration. Set to NULL to run without monitor.
   *
   *  The monitor function is called with the current guess for the result as first argument and the current relative residual estimate as second argument.
   *  The third argument is a pointer to user-defined data, through which additional information can be passed.
   *  This pointer needs to be set with set_monitor_data. If not set, NULL is passed.
   *  If the montior function returns true, the solver terminates (either convergence or divergence).
   */
  void set_monitor(bool (*monitor_fun)(VectorT const &, numeric_type, void *), void *user_data)
  {
    monitor_callback_ = monitor_fun;
    user_data_ = user_data;
  }

  /** @brief Returns the solver tag containing basic configuration such as tolerances, etc. */
  bicgstab_tag const & tag() const { return tag_; }

private:
  bicgstab_tag  tag_;
  VectorT       init_guess_;
  bool          (*monitor_callback_)(VectorT const &, numeric_type, void *);
  void          *user_data_;
};


}
}

