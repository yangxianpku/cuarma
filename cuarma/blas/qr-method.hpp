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

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/qr-method-common.hpp"
#include "cuarma/blas/tql2.hpp"
#include "cuarma/blas/prod.hpp"

/** @file cuarma/blas/qr-method.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementation of the QR method for eigenvalue computations. Experimental.
*/

namespace cuarma
{
namespace blas
{
namespace detail
{
template <typename SCALARTYPE>
void final_iter_update_gpu(matrix_base<SCALARTYPE> & A, int n, int last_n, SCALARTYPE q, SCALARTYPE p )
{
  (void)A; (void)n; (void)last_n; (void)q; (void)p;

}

template <typename SCALARTYPE, typename VectorType>
void update_float_QR_column_gpu(matrix_base<SCALARTYPE> & A, const VectorType& buf,
                    cuarma::vector<SCALARTYPE>& buf_arma, int m, int n, int last_n,bool /*is_triangular*/)
{
  (void)A; (void)buf; (void)buf_arma; (void)m; (void)n; (void)last_n;

}

template<typename SCALARTYPE, typename MatrixT>
void final_iter_update(MatrixT& A, int n, int last_n, SCALARTYPE q, SCALARTYPE p)
{
    for (int i = 0; i < last_n; i++)
    {
        SCALARTYPE v_in = A(i, n);
        SCALARTYPE z = A(i, n - 1);
        A(i, n - 1) = q * z + p * v_in;
        A(i, n) = q * v_in - p * z;
    }
}

template<typename SCALARTYPE, typename MatrixT>
void update_float_QR_column(MatrixT& A, const std::vector<SCALARTYPE>& buf, int m,int n, int last_i, bool is_triangular)
{
    for (int i = 0; i < last_i; i++)
    {
        int start_k = is_triangular?std::max(i + 1, m):m;

        SCALARTYPE* a_row = A.row(i);

        SCALARTYPE a_ik   = a_row[start_k];
        SCALARTYPE a_ik_1 = 0;
        SCALARTYPE a_ik_2 = 0;

        if (start_k < n)
            a_ik_1 = a_row[start_k + 1];

        for (int k = start_k; k < n; k++)
        {
            bool notlast = (k != n - 1);

            SCALARTYPE p = buf[5 * static_cast<arma_size_t>(k)] * a_ik + buf[5 * static_cast<arma_size_t>(k) + 1] * a_ik_1;

            if (notlast)
            {
                a_ik_2 = a_row[k + 2];
                p = p + buf[5 * static_cast<arma_size_t>(k) + 2] * a_ik_2;
                a_ik_2 = a_ik_2 - p * buf[5 * static_cast<arma_size_t>(k) + 4];
            }

            a_row[k] = a_ik - p;
            a_ik_1 = a_ik_1 - p * buf[5 * static_cast<arma_size_t>(k) + 3];

            a_ik = a_ik_1;
            a_ik_1 = a_ik_2;
        }

        if (start_k < n)
            a_row[n] = a_ik;
    }
}

    /** @brief Internal helper class representing a row-major dense matrix used for the QR method for the purpose of computing eigenvalues. */
    template<typename SCALARTYPE>
    class FastMatrix
    {
    public:
        FastMatrix()
        {
            size_ = 0;
        }

        FastMatrix(arma_size_t sz, arma_size_t internal_size) : size_(sz), internal_size_(internal_size)
        {
            data.resize(internal_size * internal_size);
        }

        SCALARTYPE& operator()(int i, int j)
        {
            return data[static_cast<arma_size_t>(i) * internal_size_ + static_cast<arma_size_t>(j)];
        }

        SCALARTYPE* row(int i)
        {
            return &data[static_cast<arma_size_t>(i) * internal_size_];
        }

        SCALARTYPE* begin()
        {
            return &data[0];
        }

        SCALARTYPE* end()
        {
            return &data[0] + data.size();
        }

        std::vector<SCALARTYPE> data;
    private:
        arma_size_t size_;
        arma_size_t internal_size_;
    };

    // Nonsymmetric reduction from Hessenberg to real Schur form.
    // This is derived from the Algol procedure hqr2, by Martin and Wilkinson, Handbook for Auto. Comp.,
    // Vol.ii-Linear Algebra, and the corresponding  Fortran subroutine in EISPACK.
    template <typename SCALARTYPE, typename VectorType>
    void hqr2(cuarma::matrix<SCALARTYPE>& arma_H, cuarma::matrix<SCALARTYPE>& V,VectorType & d,  VectorType & e)
    {
        transpose(V);

        int nn = static_cast<int>(arma_H.size1());

        FastMatrix<SCALARTYPE> H(arma_size_t(nn), arma_H.internal_size2());//, V(nn);

        std::vector<SCALARTYPE>  buf(5 * arma_size_t(nn));
        cuarma::vector<SCALARTYPE> buf_arma(5 * arma_size_t(nn));

        cuarma::fast_copy(arma_H, H.begin());


        int n = nn - 1;

        SCALARTYPE eps = 2 * static_cast<SCALARTYPE>(EPS);
        SCALARTYPE exshift = 0;
        SCALARTYPE p = 0;
        SCALARTYPE q = 0;
        SCALARTYPE r = 0;
        SCALARTYPE s = 0;
        SCALARTYPE z = 0;
        SCALARTYPE t;
        SCALARTYPE w;
        SCALARTYPE x;
        SCALARTYPE y;

        SCALARTYPE out1, out2;

        // compute matrix norm
        SCALARTYPE norm = 0;
        for (int i = 0; i < nn; i++)
        {
            for (int j = std::max(i - 1, 0); j < nn; j++)
                norm = norm + std::fabs(H(i, j));
        }

        // Outer loop over eigenvalue index
        int iter = 0;
        while (n >= 0)
        {
            // Look for single small sub-diagonal element
            int l = n;
            while (l > 0)
            {
                s = std::fabs(H(l - 1, l - 1)) + std::fabs(H(l, l));
                if (s <= 0)
                  s = norm;
                if (std::fabs(H(l, l - 1)) < eps * s)
                  break;

                l--;
            }

            // Check for convergence
            if (l == n)
            {
                // One root found
                H(n, n) = H(n, n) + exshift;
                d[arma_size_t(n)] = H(n, n);
                e[arma_size_t(n)] = 0;
                n--;
                iter = 0;
            }
            else if (l == n - 1)
            {
                // Two roots found
                w = H(n, n - 1) * H(n - 1, n);
                p = (H(n - 1, n - 1) - H(n, n)) / 2;
                q = p * p + w;
                z = static_cast<SCALARTYPE>(std::sqrt(std::fabs(q)));
                H(n, n) = H(n, n) + exshift;
                H(n - 1, n - 1) = H(n - 1, n - 1) + exshift;
                x = H(n, n);

                if (q >= 0)
                {
                    // Real pair
                    z = (p >= 0) ? (p + z) : (p - z);
                    d[arma_size_t(n) - 1] = x + z;
                    d[arma_size_t(n)] = d[arma_size_t(n) - 1];
                    if (z <= 0 && z >= 0) // z == 0 without compiler complaints
                      d[arma_size_t(n)] = x - w / z;
                    e[arma_size_t(n) - 1] = 0;
                    e[arma_size_t(n)] = 0;
                    x = H(n, n - 1);
                    s = std::fabs(x) + std::fabs(z);
                    p = x / s;
                    q = z / s;
                    r = static_cast<SCALARTYPE>(std::sqrt(p * p + q * q));
                    p = p / r;
                    q = q / r;

                    // Row modification
                    for (int j = n - 1; j < nn; j++)
                    {
                        SCALARTYPE h_nj = H(n, j);
                        z = H(n - 1, j);
                        H(n - 1, j) = q * z + p * h_nj;
                        H(n, j) = q * h_nj - p * z;
                    }

                    final_iter_update(H, n, n + 1, q, p);
                    final_iter_update_gpu(V, n, nn, q, p);
                }
                else
                {
                    // Complex pair
                    d[arma_size_t(n) - 1] = x + p;
                    d[arma_size_t(n)] = x + p;
                    e[arma_size_t(n) - 1] = z;
                    e[arma_size_t(n)] = -z;
                }

                n = n - 2;
                iter = 0;
            }
            else
            {
                // No convergence yet

                // Form shift
                x = H(n, n);
                y = 0;
                w = 0;
                if (l < n)
                {
                    y = H(n - 1, n - 1);
                    w = H(n, n - 1) * H(n - 1, n);
                }

                // Wilkinson's original ad hoc shift
                if (iter == 10)
                {
                    exshift += x;
                    for (int i = 0; i <= n; i++)
                        H(i, i) -= x;

                    s = std::fabs(H(n, n - 1)) + std::fabs(H(n - 1, n - 2));
                    x = y = SCALARTYPE(0.75) * s;
                    w = SCALARTYPE(-0.4375) * s * s;
                }

                // MATLAB's new ad hoc shift
                if (iter == 30)
                {
                    s = (y - x) / 2;
                    s = s * s + w;
                    if (s > 0)
                    {
                        s = static_cast<SCALARTYPE>(std::sqrt(s));
                        if (y < x) s = -s;
                        s = x - w / ((y - x) / 2 + s);
                        for (int i = 0; i <= n; i++)
                            H(i, i) -= s;
                        exshift += s;
                        x = y = w = SCALARTYPE(0.964);
                    }
                }

                iter = iter + 1;

                // Look for two consecutive small sub-diagonal elements
                int m = n - 2;
                while (m >= l)
                {
                    SCALARTYPE h_m1_m1 = H(m + 1, m + 1);
                    z = H(m, m);
                    r = x - z;
                    s = y - z;
                    p = (r * s - w) / H(m + 1, m) + H(m, m + 1);
                    q = h_m1_m1 - z - r - s;
                    r = H(m + 2, m + 1);
                    s = std::fabs(p) + std::fabs(q) + std::fabs(r);
                    p = p / s;
                    q = q / s;
                    r = r / s;
                    if (m == l)
                        break;
                    if (std::fabs(H(m, m - 1)) * (std::fabs(q) + std::fabs(r)) < eps * (std::fabs(p) * (std::fabs(H(m - 1, m - 1)) + std::fabs(z) + std::fabs(h_m1_m1))))
                        break;
                    m--;
                }

                for (int i = m + 2; i <= n; i++)
                {
                    H(i, i - 2) = 0;
                    if (i > m + 2)
                        H(i, i - 3) = 0;
                }

                // float QR step involving rows l:n and columns m:n
                for (int k = m; k < n; k++)
                {
                    bool notlast = (k != n - 1);
                    if (k != m)
                    {
                        p = H(k, k - 1);
                        q = H(k + 1, k - 1);
                        r = (notlast ? H(k + 2, k - 1) : 0);
                        x = std::fabs(p) + std::fabs(q) + std::fabs(r);
                        if (x > 0)
                        {
                            p = p / x;
                            q = q / x;
                            r = r / x;
                        }
                    }

                    if (x <= 0 && x >= 0) break;  // x == 0 without compiler complaints

                    s = static_cast<SCALARTYPE>(std::sqrt(p * p + q * q + r * r));
                    if (p < 0) s = -s;

                    if (s < 0 || s > 0)
                    {
                        if (k != m)
                            H(k, k - 1) = -s * x;
                        else
                            if (l != m)
                                H(k, k - 1) = -H(k, k - 1);

                        p = p + s;
                        y = q / s;
                        z = r / s;
                        x = p / s;
                        q = q / p;
                        r = r / p;

                        buf[5 * arma_size_t(k)] = x;
                        buf[5 * arma_size_t(k) + 1] = y;
                        buf[5 * arma_size_t(k) + 2] = z;
                        buf[5 * arma_size_t(k) + 3] = q;
                        buf[5 * arma_size_t(k) + 4] = r;


                        SCALARTYPE* a_row_k = H.row(k);
                        SCALARTYPE* a_row_k_1 = H.row(k + 1);
                        SCALARTYPE* a_row_k_2 = H.row(k + 2);
                        // Row modification
                        for (int j = k; j < nn; j++)
                        {
                            SCALARTYPE h_kj = a_row_k[j];
                            SCALARTYPE h_k1_j = a_row_k_1[j];

                            p = h_kj + q * h_k1_j;
                            if (notlast)
                            {
                                SCALARTYPE h_k2_j = a_row_k_2[j];
                                p = p + r * h_k2_j;
                                a_row_k_2[j] = h_k2_j - p * z;
                            }

                            a_row_k[j] = h_kj - p * x;
                            a_row_k_1[j] = h_k1_j - p * y;
                        }

                        //H(k + 1, nn - 1) = h_kj;


                        // Column modification
                        for (int i = k; i < std::min(nn, k + 4); i++)
                        {
                            p = x * H(i, k) + y * H(i, k + 1);
                            if (notlast)
                            {
                                p = p + z * H(i, k + 2);
                                H(i, k + 2) = H(i, k + 2) - p * r;
                            }

                            H(i, k) = H(i, k) - p;
                            H(i, k + 1) = H(i, k + 1) - p * q;
                        }
                    }
                    else
                    {
                        buf[5 * arma_size_t(k)] = 0;
                        buf[5 * arma_size_t(k) + 1] = 0;
                        buf[5 * arma_size_t(k) + 2] = 0;
                        buf[5 * arma_size_t(k) + 3] = 0;
                        buf[5 * arma_size_t(k) + 4] = 0;
                    }
                }

                // Timer timer;
                // timer.start();

                update_float_QR_column<SCALARTYPE>(H, buf, m, n, n, true);
                update_float_QR_column_gpu(V, buf, buf_arma, m, n, nn, false);

                // std::cout << timer.get() << "\n";
            }
        }

        // Backsubstitute to find vectors of upper triangular form
        if (norm <= 0)
        {
            return;
        }

        for (n = nn - 1; n >= 0; n--)
        {
            p = d[arma_size_t(n)];
            q = e[arma_size_t(n)];

            // Real vector
            if (q <= 0 && q >= 0)
            {
                int l = n;
                H(n, n) = 1;
                for (int i = n - 1; i >= 0; i--)
                {
                    w = H(i, i) - p;
                    r = 0;
                    for (int j = l; j <= n; j++)
                        r = r + H(i, j) * H(j, n);

                    if (e[arma_size_t(i)] < 0)
                    {
                        z = w;
                        s = r;
                    }
                    else
                    {
                        l = i;
                        if (e[arma_size_t(i)] <= 0) // e[i] == 0 with previous if
                        {
                            H(i, n) = (w > 0 || w < 0) ? (-r / w) : (-r / (eps * norm));
                        }
                        else
                        {
                            // Solve real equations
                            x = H(i, i + 1);
                            y = H(i + 1, i);
                            q = (d[arma_size_t(i)] - p) * (d[arma_size_t(i)] - p) + e[arma_size_t(i)] * e[arma_size_t(i)];
                            t = (x * s - z * r) / q;
                            H(i, n) = t;
                            H(i + 1, n) = (std::fabs(x) > std::fabs(z)) ? ((-r - w * t) / x) : ((-s - y * t) / z);
                        }

                        // Overflow control
                        t = std::fabs(H(i, n));
                        if ((eps * t) * t > 1)
                            for (int j = i; j <= n; j++)
                                H(j, n) /= t;
                    }
                }
            }
            else if (q < 0)
            {
                // Complex vector
                int l = n - 1;

                // Last vector component imaginary so matrix is triangular
                if (std::fabs(H(n, n - 1)) > std::fabs(H(n - 1, n)))
                {
                    H(n - 1, n - 1) = q / H(n, n - 1);
                    H(n - 1, n) = -(H(n, n) - p) / H(n, n - 1);
                }
                else
                {
                    cdiv<SCALARTYPE>(0, -H(n - 1, n), H(n - 1, n - 1) - p, q, out1, out2);

                    H(n - 1, n - 1) = out1;
                    H(n - 1, n) = out2;
                }

                H(n, n - 1) = 0;
                H(n, n) = 1;
                for (int i = n - 2; i >= 0; i--)
                {
                    SCALARTYPE ra, sa, vr, vi;
                    ra = 0;
                    sa = 0;
                    for (int j = l; j <= n; j++)
                    {
                        SCALARTYPE h_ij = H(i, j);
                        ra = ra + h_ij * H(j, n - 1);
                        sa = sa + h_ij * H(j, n);
                    }

                    w = H(i, i) - p;

                    if (e[arma_size_t(i)] < 0)
                    {
                        z = w;
                        r = ra;
                        s = sa;
                    }
                    else
                    {
                        l = i;
                        if (e[arma_size_t(i)] <= 0) // e[i] == 0 with previous if
                        {
                            cdiv<SCALARTYPE>(-ra, -sa, w, q, out1, out2);
                            H(i, n - 1) = out1;
                            H(i, n) = out2;
                        }
                        else
                        {
                            // Solve complex equations
                            x = H(i, i + 1);
                            y = H(i + 1, i);
                            vr = (d[arma_size_t(i)] - p) * (d[arma_size_t(i)] - p) + e[arma_size_t(i)] * e[arma_size_t(i)] - q * q;
                            vi = (d[arma_size_t(i)] - p) * 2 * q;
                            if ( (vr <= 0 && vr >= 0) && (vi <= 0 && vi >= 0) )
                                vr = eps * norm * (std::fabs(w) + std::fabs(q) + std::fabs(x) + std::fabs(y) + std::fabs(z));

                            cdiv<SCALARTYPE>(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi, out1, out2);

                            H(i, n - 1) = out1;
                            H(i, n) = out2;


                            if (std::fabs(x) > (std::fabs(z) + std::fabs(q)))
                            {
                                H(i + 1, n - 1) = (-ra - w * H(i, n - 1) + q * H(i, n)) / x;
                                H(i + 1, n) = (-sa - w * H(i, n) - q * H(i, n - 1)) / x;
                            }
                            else
                            {
                                cdiv<SCALARTYPE>(-r - y * H(i, n - 1), -s - y * H(i, n), z, q, out1, out2);

                                H(i + 1, n - 1) = out1;
                                H(i + 1, n) = out2;
                            }
                        }

                        // Overflow control
                        t = std::max(std::fabs(H(i, n - 1)), std::fabs(H(i, n)));
                        if ((eps * t) * t > 1)
                        {
                            for (int j = i; j <= n; j++)
                            {
                                H(j, n - 1) /= t;
                                H(j, n) /= t;
                            }
                        }
                    }
                }
            }
        }

        cuarma::fast_copy(H.begin(), H.end(),  arma_H);
        // cuarma::fast_copy(V.begin(), V.end(),  arma_V);

        cuarma::matrix<SCALARTYPE> tmp = V;

        V = cuarma::blas::prod(trans(tmp), arma_H);
    }


    template <typename SCALARTYPE>
    bool householder_twoside(matrix_base<SCALARTYPE>& A, matrix_base<SCALARTYPE>& Q, vector_base<SCALARTYPE>& D, arma_size_t start)
    {
        arma_size_t A_size1 = static_cast<arma_size_t>(cuarma::traits::size1(A));
        if(start + 2 >= A_size1)
            return false;

        prepare_householder_vector(A, D, A_size1, start + 1, start, start + 1, true);
        cuarma::blas::house_update_A_left(A, D, start);
        cuarma::blas::house_update_A_right(A, D);
        cuarma::blas::house_update_QL(Q, D, A_size1);

        return true;
    }

    template <typename SCALARTYPE>
    void tridiagonal_reduction(matrix_base<SCALARTYPE>& A,  matrix_base<SCALARTYPE>& Q)
    {
        arma_size_t sz = A.size1();
        cuarma::vector<SCALARTYPE> hh_vector(sz);

        for(arma_size_t i = 0; i < sz; i++)
        {
            householder_twoside(A, Q, hh_vector, i);
        }

    }

#ifdef CUARMA_WITH_UBLAS
    template <typename SCALARTYPE>
    void qr_method(cuarma::matrix<SCALARTYPE> & A,  cuarma::matrix<SCALARTYPE> & Q, std::vector<SCALARTYPE> & D, std::vector<SCALARTYPE> & E, bool is_symmetric = true)
    {
        assert(A.size1() == A.size2() && bool("Input matrix must be square for QR method!"));
    /*    if (!cuarma::is_row_major<F>::value && !is_symmetric)
        {
          std::cout << "qr_method for non-symmetric column-major matrices not implemented yet!" << std::endl;
          exit(EXIT_FAILURE);
        }

        */
        arma_size_t mat_size = A.size1();
        D.resize(A.size1());
        E.resize(A.size1());

        cuarma::vector<SCALARTYPE> arma_D(mat_size), arma_E(mat_size);
        //std::vector<SCALARTYPE> std_D(mat_size), std_E(mat_size);

        Q = cuarma::identity_matrix<SCALARTYPE>(Q.size1());

        // reduce to tridiagonal form
        detail::tridiagonal_reduction(A, Q);

        // pack diagonal and super-diagonal
        cuarma::blas::bidiag_pack(A, arma_D, arma_E);
        copy(arma_D, D);
        copy(arma_E, E);

        // find eigenvalues of symmetric tridiagonal matrix
        if(is_symmetric)
        {
          cuarma::blas::tql2(Q, D, E);

        }
        else
        {
              detail::hqr2(A, Q, D, E);
        }

        boost::numeric::ublas::matrix<SCALARTYPE> eigen_values(A.size1(), A.size1());
        eigen_values.clear();

        for (arma_size_t i = 0; i < A.size1(); i++)
        {
            if(std::fabs(E[i]) < EPS)
            {
                eigen_values(i, i) = D[i];
            }
            else
            {
                eigen_values(i, i) = D[i];
                eigen_values(i, i + 1) = E[i];
                eigen_values(i + 1, i) = -E[i];
                eigen_values(i + 1, i + 1) = D[i];
                i++;
            }
        }
        copy(eigen_values, A);
    }
#endif
}

#ifdef CUARMA_WITH_UBLAS
template <typename SCALARTYPE>
void qr_method_nsm(cuarma::matrix<SCALARTYPE>& A, cuarma::matrix<SCALARTYPE>& Q, std::vector<SCALARTYPE>& D, std::vector<SCALARTYPE>& E )
{
    detail::qr_method(A, Q, D, E, false);
}

template <typename SCALARTYPE>
void qr_method_sym(cuarma::matrix<SCALARTYPE>& A, cuarma::matrix<SCALARTYPE>& Q, std::vector<SCALARTYPE>& D)
{
    std::vector<SCALARTYPE> E(A.size1());
    detail::qr_method(A, Q, D, E, true);
}

template <typename SCALARTYPE>
void qr_method_sym(cuarma::matrix<SCALARTYPE>& A, cuarma::matrix<SCALARTYPE>& Q, cuarma::vector_base<SCALARTYPE>& D )
{
    std::vector<SCALARTYPE> std_D(D.size());
    std::vector<SCALARTYPE> E(A.size1());

    cuarma::copy(D, std_D);
    detail::qr_method(A, Q, std_D, E, true);
    cuarma::copy(std_D, D);
}
#endif

}
}
