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

#include <stdlib.h>
// Extra export declarations when building with Visual Studio:
#if defined(_MSC_VER)
  #if defined(CUARMA_EXPORTS)
    #define  CUARMA_EXPORTED_FUNCTION __declspec(dllexport)
  #else
    #define  CUARMA_EXPORTED_FUNCTION __declspec(dllimport)
  #endif /* cuarma_EXPORTS */
#else /* defined (_MSC_VER) */
 #define CUARMA_EXPORTED_FUNCTION
#endif


#ifdef __cplusplus
extern "C" {
#endif

typedef int cuarmaInt;


/************** Enums ***************/

typedef enum
{
  cuarmaInvalidBackend, // for catching uninitialized and invalid values
  cuarmaCUDA,
  cuarmaHost
} cuarmaBackendTypes;

typedef enum
{
  cuarmaInvalidOrder,  // for catching uninitialized and invalid values
  cuarmaRowMajor,
  cuarmaColumnMajor
} cuarmaOrder;

typedef enum
{
  cuarmaInvalidTranspose, // for catching uninitialized and invalid values
  cuarmaNoTrans,
  cuarmaTrans
} cuarmaTranspose;

typedef enum
{
  cuarmaInvalidUplo, // for catching uninitialized and invalid values
  cuarmaUpper,
  cuarmaLower
} cuarmaUplo;

typedef enum
{
  cuarmaInvalidDiag, // for catching uninitialized and invalid values
  cuarmaUnit,
  cuarmaNonUnit
} cuarmaDiag;

typedef enum
{
  cuarmaInvalidPrecision,  // for catching uninitialized and invalid values
  cuarmaFloat,
  cuarmaDouble
} cuarmaPrecision;

// Error codes:
typedef enum
{
  cuarmaSuccess = 0,
  cuarmaGenericFailure
} cuarmaStatus;


/************* Backend Management ******************/

/** @brief Generic backend for CUDA, host-based stuff */
struct cuarmaBackend_impl;
typedef cuarmaBackend_impl*   cuarmaBackend;

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaBackendCreate(cuarmaBackend * backend);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaBackendDestroy(cuarmaBackend * backend);

/******** User Types **********/

struct cuarmaHostScalar_impl;
typedef cuarmaHostScalar_impl*    cuarmaHostScalar;

struct cuarmaScalar_impl;
typedef cuarmaScalar_impl*        cuarmaScalar;

struct cuarmaVector_impl;
typedef cuarmaVector_impl*        cuarmaVector;

struct cuarmaMatrix_impl;
typedef cuarmaMatrix_impl*        cuarmaMatrix;


/******************** BLAS Level 1 ***********************/

// IxASUM

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaiamax(cuarmaInt *alpha, cuarmaVector x);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDAiSamax(cuarmaBackend backend, cuarmaInt n,
                                                             cuarmaInt *alpha,
                                                             float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDAiDamax(cuarmaBackend backend, cuarmaInt n,
                                                             cuarmaInt *alpha,
                                                             double *x, cuarmaInt offx, cuarmaInt incx);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostiSamax(cuarmaBackend backend, cuarmaInt n,
                                                             cuarmaInt *alpha,
                                                             float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostiDamax(cuarmaBackend backend, cuarmaInt n,
                                                             cuarmaInt *alpha,
                                                             double *x, cuarmaInt offx, cuarmaInt incx);


// xASUM

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaasum(cuarmaHostScalar *alpha, cuarmaVector x);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASasum(cuarmaBackend backend, cuarmaInt n,
                                                            float *alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADasum(cuarmaBackend backend, cuarmaInt n,
                                                            double *alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSasum(cuarmaBackend backend, cuarmaInt n,
                                                            float *alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDasum(cuarmaBackend backend, cuarmaInt n,
                                                            double *alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx);



// xAXPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaaxpy(cuarmaHostScalar alpha, cuarmaVector x, cuarmaVector y);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASaxpy(cuarmaBackend backend, cuarmaInt n,
                                                            float alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADaxpy(cuarmaBackend backend, cuarmaInt n,
                                                            double alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSaxpy(cuarmaBackend backend, cuarmaInt n,
                                                            float alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDaxpy(cuarmaBackend backend, cuarmaInt n,
                                                            double alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy);


// xCOPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmacopy(cuarmaVector x, cuarmaVector y);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDAScopy(cuarmaBackend backend, cuarmaInt n,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADcopy(cuarmaBackend backend, cuarmaInt n,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostScopy(cuarmaBackend backend, cuarmaInt n,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDcopy(cuarmaBackend backend, cuarmaInt n,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy);

// xDOT

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmadot(cuarmaHostScalar *alpha, cuarmaVector x, cuarmaVector y);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASdot(cuarmaBackend backend, cuarmaInt n,
                                                           float *alpha,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADdot(cuarmaBackend backend, cuarmaInt n,
                                                           double *alpha,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSdot(cuarmaBackend backend, cuarmaInt n,
                                                           float *alpha,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDdot(cuarmaBackend backend, cuarmaInt n,
                                                           double *alpha,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy);

// xNRM2

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmanrm2(cuarmaHostScalar *alpha, cuarmaVector x);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASnrm2(cuarmaBackend backend, cuarmaInt n,
                                                            float *alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADnrm2(cuarmaBackend backend, cuarmaInt n,
                                                            double *alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSnrm2(cuarmaBackend backend, cuarmaInt n,
                                                            float *alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDnrm2(cuarmaBackend backend, cuarmaInt n,
                                                            double *alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx);


// xROT

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmarot(cuarmaVector     x,     cuarmaVector y,
                                                      cuarmaHostScalar c, cuarmaHostScalar s);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASrot(cuarmaBackend backend, cuarmaInt n,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy,
                                                           float c, float s);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADrot(cuarmaBackend backend, cuarmaInt n,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy,
                                                           double c, double s);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSrot(cuarmaBackend backend, cuarmaInt n,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy,
                                                           float c, float s);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDrot(cuarmaBackend backend, cuarmaInt n,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy,
                                                           double c, double s);



// xSCAL

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmascal(cuarmaHostScalar alpha, cuarmaVector x);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASscal(cuarmaBackend backend, cuarmaInt n,
                                                            float alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADscal(cuarmaBackend backend, cuarmaInt n,
                                                            double alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSscal(cuarmaBackend backend, cuarmaInt n,
                                                            float alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDscal(cuarmaBackend backend, cuarmaInt n,
                                                            double alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx);


// xSWAP

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaswap(cuarmaVector x, cuarmaVector y);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASswap(cuarmaBackend backend, cuarmaInt n,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADswap(cuarmaBackend backend, cuarmaInt n,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSswap(cuarmaBackend backend, cuarmaInt n,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDswap(cuarmaBackend backend, cuarmaInt n,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy);



/******************** BLAS Level 2 ***********************/

// xGEMV: y <- alpha * Ax + beta * y

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmagemv(cuarmaHostScalar alpha, cuarmaMatrix A, cuarmaVector x, cuarmaHostScalar beta, cuarmaVector y);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASgemv(cuarmaBackend backend,
                                                            cuarmaOrder order, cuarmaTranspose transA,
                                                            cuarmaInt m, cuarmaInt n, float alpha, float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float beta,
                                                            float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADgemv(cuarmaBackend backend,
                                                            cuarmaOrder order, cuarmaTranspose transA,
                                                            cuarmaInt m, cuarmaInt n, double alpha, double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double beta,
                                                            double *y, cuarmaInt offy, cuarmaInt incy);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSgemv(cuarmaBackend backend,
                                                            cuarmaOrder order, cuarmaTranspose transA,
                                                            cuarmaInt m, cuarmaInt n, float alpha, float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float beta,
                                                            float *y, cuarmaInt offy, cuarmaInt incy);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDgemv(cuarmaBackend backend,
                                                            cuarmaOrder order, cuarmaTranspose transA,
                                                            cuarmaInt m, cuarmaInt n, double alpha, double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double beta,
                                                            double *y, cuarmaInt offy, cuarmaInt incy);

// xTRSV: Ax <- x

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmatrsv(cuarmaMatrix A, cuarmaVector x, cuarmaUplo uplo);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDAStrsv(cuarmaBackend backend,
                                                            cuarmaUplo uplo, cuarmaOrder order, cuarmaTranspose transA, cuarmaDiag diag,
                                                            cuarmaInt n, float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADtrsv(cuarmaBackend backend,
                                                            cuarmaUplo uplo, cuarmaOrder order, cuarmaTranspose transA, cuarmaDiag diag,
                                                            cuarmaInt n, double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *x, cuarmaInt offx, cuarmaInt incx);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostStrsv(cuarmaBackend backend,
                                                            cuarmaUplo uplo, cuarmaOrder order, cuarmaTranspose transA, cuarmaDiag diag,
                                                            cuarmaInt n, float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *x, cuarmaInt offx, cuarmaInt incx);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDtrsv(cuarmaBackend backend,
                                                            cuarmaUplo uplo, cuarmaOrder order, cuarmaTranspose transA, cuarmaDiag diag,
                                                            cuarmaInt n, double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *x, cuarmaInt offx, cuarmaInt incx);


// xGER: A <- alpha * x * y + A

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmager(cuarmaHostScalar alpha, cuarmaVector x, cuarmaVector y, cuarmaMatrix A);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASger(cuarmaBackend backend,
                                                           cuarmaOrder order,
                                                           cuarmaInt m, cuarmaInt n,
                                                           float alpha,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy,
                                                           float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADger(cuarmaBackend backend,
                                                           cuarmaOrder order,
                                                           cuarmaInt m,  cuarmaInt n,
                                                           double alpha,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy,
                                                           double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSger(cuarmaBackend backend,
                                                           cuarmaOrder order,
                                                           cuarmaInt m, cuarmaInt n,
                                                           float alpha,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy,
                                                           float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDger(cuarmaBackend backend,
                                                           cuarmaOrder order,
                                                           cuarmaInt m, cuarmaInt n,
                                                           double alpha,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy,
                                                           double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda);



/******************** BLAS Level 3 ***********************/

// xGEMM: C <- alpha * AB + beta * C

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmagemm(cuarmaHostScalar alpha, cuarmaMatrix A, cuarmaMatrix B, cuarmaHostScalar beta, cuarmaMatrix C);

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASgemm(cuarmaBackend backend,
                                                            cuarmaOrder orderA, cuarmaTranspose transA,
                                                            cuarmaOrder orderB, cuarmaTranspose transB,
                                                            cuarmaOrder orderC,
                                                            cuarmaInt m, cuarmaInt n, cuarmaInt k,
                                                            float alpha,
                                                            float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *B, cuarmaInt offB_row, cuarmaInt offB_col, cuarmaInt incB_row, cuarmaInt incB_col, cuarmaInt ldb,
                                                            float beta,
                                                            float *C, cuarmaInt offC_row, cuarmaInt offC_col, cuarmaInt incC_row, cuarmaInt incC_col, cuarmaInt ldc);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADgemm(cuarmaBackend backend,
                                                            cuarmaOrder orderA, cuarmaTranspose transA,
                                                            cuarmaOrder orderB, cuarmaTranspose transB,
                                                            cuarmaOrder orderC,
                                                            cuarmaInt m, cuarmaInt n, cuarmaInt k,
                                                            double alpha,
                                                            double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *B, cuarmaInt offB_row, cuarmaInt offB_col, cuarmaInt incB_row, cuarmaInt incB_col, cuarmaInt ldb,
                                                            double beta,
                                                            double *C, cuarmaInt offC_row, cuarmaInt offC_col, cuarmaInt incC_row, cuarmaInt incC_col, cuarmaInt ldc);



CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSgemm(cuarmaBackend backend,
                                                            cuarmaOrder orderA, cuarmaTranspose transA,
                                                            cuarmaOrder orderB, cuarmaTranspose transB,
                                                            cuarmaOrder orderC,
                                                            cuarmaInt m, cuarmaInt n, cuarmaInt k,
                                                            float alpha,
                                                            float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *B, cuarmaInt offB_row, cuarmaInt offB_col, cuarmaInt incB_row, cuarmaInt incB_col, cuarmaInt ldb,
                                                            float beta,
                                                            float *C, cuarmaInt offC_row, cuarmaInt offC_col, cuarmaInt incC_row, cuarmaInt incC_col, cuarmaInt ldc);
CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDgemm(cuarmaBackend backend,
                                                            cuarmaOrder orderA, cuarmaTranspose transA,
                                                            cuarmaOrder orderB, cuarmaTranspose transB,
                                                            cuarmaOrder orderC,
                                                            cuarmaInt m, cuarmaInt n, cuarmaInt k,
                                                            double alpha,
                                                            double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *B, cuarmaInt offB_row, cuarmaInt offB_col, cuarmaInt incB_row, cuarmaInt incB_col, cuarmaInt ldb,
                                                            double beta,
                                                            double *C, cuarmaInt offC_row, cuarmaInt offC_col, cuarmaInt incC_row, cuarmaInt incC_col, cuarmaInt ldc);

// xTRSM: Triangular solves with multiple right hand sides

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmatrsm(cuarmaMatrix A, cuarmaUplo uplo, cuarmaDiag diag, cuarmaMatrix B);

#ifdef __cplusplus
}
#endif