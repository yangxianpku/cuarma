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

// Connector for Mex files to use cuarma for calculation
// encoding:UTF-8
// Version 0.1

#include "cuarma/cuarma"
#include <mex.h>
#include <mat.h>
#include <cstring>
#include "gpu/mxGPUArray.h"
#include <algorithm>

///////////////////////////////////////////////////////////////////////////
/****************************** 定义错误信息  ****************************/
///////////////////////////////////////////////////////////////////////////

char const * const errId = "cuarma::cuda::cuarmaMex::Error!";

char const * const notGPUArrayErr     = "Input is not valid GPU data!";
char const * const notCPUArrayErr     = "Input is not valid CPU data!";
char const * const notDoubleErr       = "Input is not mxDOUBLE_CLASS!";
char const * const notFloatErr        = "Input is not mxSINGLE_CLASS!";
char const * const notMatrixErr       = "Number of dimensions must be 2.";
char const * const notVectorErr       = "Number of dimensions must be 1.";
char const * const noDataAvailableErr = "No data available.";

char const * const GPUDeviceNotAvailableErr = "CUARMA_WITH_CUDA not defined or CUDA device not available.";

char const * const matFileNotAvailableErr = "specified mat file is not available.";
char const * const matFileCantOpen        = "Could not open MAT file.";

///////////////////////////////////////////////////////////////////////////
/************************** 从matlab中获取值  ****************************/
///////////////////////////////////////////////////////////////////////////

/** 从Matlab获取标量 **/
template<typename ScalarType>
inline cuarma::scalar<ScalarType> cuarmaGetScalar(const mxArray *matlabScalar)
{
	cuarma::scalar<ScalarType> cuamraScalar = ScalarType(0);
	if (mxGetData(matlabScalar) != NULL)
	{
		cuamraScalar = (ScalarType)mxGetScalar(matlabScalar);
	}
	else
	{
		mexErrMsgIdAndTxt(errId, noDataAvailableErr);
	}
	return cuamraScalar;
}

/************************************
// MethodName:    cuarmaGetDouble
// Description:   获取cuarma double标量
// Author:        YangXian
// Date:          2017/10/06 16:35
// Parameter:     const mxArray * matlabScalar
// Returns:       cuarma::scalar<double>
**************************************/
inline cuarma::scalar<double> cuarmaGetDouble(const mxArray *matlabScalar)
{
	return cuarmaGetScalar<double>(matlabScalar);
}

inline cuarma::scalar<float> cuarmaGetFloat(const mxArray *matlabScalar)
{
	return cuarmaGetScalar<float>(matlabScalar);
}

/************************************
// MethodName:    cuarmaGetData
// Description:   从Matlab中获取非双精度实数矩阵，主程序中使用mxGetClassID查看精度
// Author:        YangXian
// Date:          2017/10/07 17:41
// Parameter:     const mxArray * matlabMatrix 实数矩阵
// Returns:
**************************************/
template<class ScalarType>
inline cuarma::matrix<ScalarType, cuarma::column_major> cuarmaGetData(const mxArray *matlabMatrix)
{
	if (mxGetData(matlabMatrix) != NULL)
	{
#ifdef CUARMA_WITH_CUDA
		mxGPUArray * gpu_data = mxGPUCopyFromMxArray(matlabMatrix);
		mwSize n_dim = mxGPUGetNumberOfDimensions(gpu_data);

		if (n_dim != 2)
		{
			mexErrMsgIdAndTxt(errId, notMatrixErr);
			return cuarma::matrix<ScalarType, cuarma::column_major>();
		}
		const mwSize * n_size = mxGPUGetDimensions(gpu_data);
		ScalarType * gpu_data_addr = (ScalarType*)(mxGPUGetData(gpu_data));
		cuarma::matrix<ScalarType, cuarma::column_major> cuarmaMat = cuarma::matrix<ScalarType, cuarma::column_major>(gpu_data_addr, cuarma::CUDA_MEMORY, n_size[0], n_size[2]);
		mxGPUDestroyGPUArray(gpu_data);
		return cuarmaMat;
#else
		mexErrMsgIdAndTxt(errId, GPUDeviceNotAvailableErr);
		return cuarma::matrix<ScalarType, cuarma::column_major>();
#endif
	}
	mexErrMsgIdAndTxt(errId, noDataAvailableErr);
	return cuarma::matrix<ScalarType, cuarma::column_major>();
}

/************************************
// MethodName:    cuarmaGetImagData
// Description:   从Matlab中获取ScalarType精度虚数矩阵(获取matlab复数矩阵的虚部)
// Author:        YangXian
// Date:          2017/10/07 17:34
// Parameter:     const mxArray * matlabMatrix
// Returns:
**************************************/
template<class ScalarType>
inline cuarma::matrix<ScalarType, cuarma::column_major> cuarmaGetImagData(const mxArray * matlabMatrix)
{
	ScalarType * imag_data = (ScalarType *)mxGetImagData(matlabMatrix);
	if (imag_data != NULL)
	{
#ifdef CUARMA_WITH_CUDA
		mxGPUArray * addr = mxGPUCopyFromMxArray(matlabMatrix);
		mwSize n_dim = mxGPUGetNumberOfDimensions(addr);
		if (n_dim == 2)
		{
			mxGPUArray * imag_addr = mxGPUCopyImag(addr);
			mwSize const * n_size = mxGPUGetDimensions(addr);
			mwSize n_row = n_size[0], n_col = n_size[2];
			mxGPUDestroyGPUArray(addr);

			ScalarType * imag_addr_pointer = (ScalarType *)(mxGPUGetData(imag_addr));
			cuarma::matrix<ScalarType, cuarma::column_major> cuarmaMat = cuarma::matrix<ScalarType, cuarma::column_major>(imag_addr_pointer, cuarma::CUDA_MEMORY, n_row, n_col);
			mxGPUDestroyGPUArray(imag_addr);
			return cuarmaMat;
		}
#else
		mexErrMsgIdAndTxt(errId, GPUDeviceNotAvailableErr);
		return cuarma::matrix<ScalarType, cuarma::column_major>();
#endif
	}
	mexErrMsgIdAndTxt(errId, noDataAvailableErr);
	return cuarma::matrix<ScalarType, cuarma::column_major>();
}

/************************************
// MethodName:    cuarmaGetPr
// Description:   从Matlab中获取double精度实数矩阵(获取matlab复数矩阵的实部)
// Author:        YangXian
// Date:          2017/10/07 17:34
// Parameter:     const mxArray * matlabMatrix
// Returns:
**************************************/
inline cuarma::matrix<double, cuarma::column_major> cuarmaGetPr(const mxArray * matlabMatrix)
{
	return cuarmaGetData<double>(matlabMatrix);
}

/************************************
// MethodName:    cuarmaGetPi
// Description:   从Matlab中获取doublee精度实数矩阵(获取matlab复数矩阵的虚部)
// Author:        YangXian
// Date:          2017/10/07 17:34
// Parameter:     const mxArray * matlabMatrix
// Returns:
**************************************/
inline cuarma::matrix<double, cuarma::column_major> cuarmaGetPi(const mxArray *matlabMatrix)
{
	return cuarmaGetImagData<double>(matlabMatrix);
}

/************************************
// MethodName:    cuarmaGetComplex
// Description:   从Matlab中获取ScalarType类型复数矩阵
// Author:        YangXian
// Date:          2017/10/07 17:34
// Parameter:     const mxArray * matlabMatrix
// Returns:
**************************************/
template<class ScalarType>
inline cuarma::matrix<ScalarType, cuarma::column_major> cuarmaGetComplex(const mxArray * matlabMatrix)
{
	ScalarType * real_addr = (ScalarType *)mxGetData(matlabMatrix);
	ScalarType * imag_addr = (ScalarType *)mxGetImagData(matlabMatrix);

	if ((real_addr == NULL) && (imag_addr == NULL))
	{
		mexErrMsgIdAndTxt(errId, noDataAvailableErr);
		return cuarma::matrix<ScalarType, cuarma::column_major>();
	}

	mwSize n_dim = mxGetNumberOfDimensions(matlabMatrix);
	if (n_dim != 2)
	{
		mexErrMsgIdAndTxt(errId, notMatrixErr);
		return cuarma::matrix<ScalarType, cuarma::column_major>();
	}

	mwSize n_row = mxGetM(matlabMatrix), n_col = mxGetN(matlabMatrix);

	mxArray * combine_data;
	if (cuarma::is_double<ScalarType>::value)
		combine_data = mxCreateNumericMatrix(n_row, 2 * n_col, mxDOUBLE_CLASS, mxREAL);
	else if (cuarma::is_float<ScalarType>::value)
		combine_data = mxCreateNumericMatrix(n_row, 2 * n_col, mxSINGLE_CLASS, mxREAL);
	else if (cuarma::is_integer<ScalarType>::value)
		combine_data = mxCreateNumericMatrix(n_row, 2 * n_col, mxINT32_CLASS, mxREAL);
	else
		return cuarma::matrix<ScalarType, cuarma::column_major>();

	ScalarType * complex_addr = (ScalarType *)mxGetPr(combine_data);

	if (real_addr != NULL && imag_addr != NULL)
	{
		for (mwSize i = 0; i < 2 * n_col; i++)      // col_iter
		{
			if (i % 2 == 0)
			{
				for (mwSize j = 0; j < n_row; j++)  // row_iter
					complex_addr[n_row*i + j] = real_addr[(i / 2)*n_row + j];
			}
			else
			{
				for (mwSize j = 0; j < n_row; j++)
					complex_addr[n_row*i + j] = imag_addr[(i / 2)*n_row + j];
			}
		}
	}
	else if (real_addr != NULL && imag_addr == NULL)
	{
		for (mwSize i = 0; i < 2 * n_col; i++)
		{
			if (i % 2 == 0)
			{
				for (mwSize j = 0; j < n_row; j++)
					complex_addr[n_row*i + j] = real_addr[(i / 2)*n_row + j];
			}
			else
			{
				for (mwSize j = 0; j < n_row; j++)
					complex_addr[n_row*i + j] = ScalarType(0.0);
			}
		}
	}
	else
	{
		for (mwSize i = 0; i < 2 * n_col; i++)
		{
			if (i % 2 == 0)
			{
				for (mwSize j = 0; j < n_row; j++)
					complex_addr[n_row*i + j] = ScalarType(0.0);
			}
			else
			{
				for (mwSize j = 0; j < n_row; j++)
					complex_addr[n_row*i + j] = imag_addr[(i / 2)*n_row + j];
			}
		}
	}

	mxGPUArray * addr = mxGPUCopyFromMxArray(combine_data);
	mxDestroyArray(combine_data);
	ScalarType * com_addr_pt = (ScalarType *)(mxGPUGetData(addr));
	cuarma::matrix<ScalarType, cuarma::column_major> cuarmaMat = cuarma::matrix<ScalarType, cuarma::column_major>(com_addr_pt, cuarma::CUDA_MEMORY, n_row, 2 * n_col);
	mxGPUDestroyGPUArray(addr);

	return cuarmaMat;
}

/************************************
// MethodName:    cuarmaSetData
// Description:   cuarmaMat的值传递给mxArray实部
// Author:        YangXian
// Date:          2017/10/19 13:09
// Parameter:     mxArray * matlabMatrix
// Parameter:     const cuarma::matrix<ScalarType> & cuarmaMat
// Returns:       mxArray *
**************************************/
template<typename ScalarType>
inline mxArray * cuarmaSetData(const cuarma::matrix<ScalarType, cuarma::column_major>& cuarmaMat)
{
	mxArray * matlabMatrix;

	if (cuarma::is_double<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
	else if (cuarma::is_float<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
	else if (cuarma::is_integer<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(0, 0, mxINT32_CLASS, mxREAL);
	else
	{
		return NULL;
	}

	int n_row = cuarmaMat.size1(), n_col = cuarmaMat.size2();
	if (n_row == 0 || n_col == 0)
	{
		mexErrMsgIdAndTxt(errId, noDataAvailableErr);
		return NULL;
	}

	ScalarType * dst_pointer = (ScalarType*)mxCalloc(n_row*n_col, sizeof(ScalarType));
	//cudaMemcpy(dst_pointer, cuarmaMat.handle().cuda_handle().get(), n_row*n_col*sizeof(ScalarType), cudaMemcpyDeviceToHost);
	for (int j = 0; j < n_col; j++)
	for (int i = 0; i < n_row; i++)
		dst_pointer[j*n_row + i] = cuarmaMat(i, j);

	mxSetData(matlabMatrix, dst_pointer);
	mxSetM(matlabMatrix, n_row);
	mxSetN(matlabMatrix, n_col);

	return matlabMatrix;
}

/************************************
// MethodName:    cuarmaSetImag
// Description:    cuarmaMat的值传递给mxArray虚部
// Author:        YangXian
// Date:          2017/10/19 16:01
// Parameter:     const cuarma::matrix<ScalarType> & cuarmaMat
// Returns:       mxArray *
**************************************/
template<typename ScalarType>
inline mxArray * cuarmaSetImagData(const cuarma::matrix<ScalarType, cuarma::column_major>& cuarmaMat)
{
	mxArray * matlabMatrix;

	if (cuarma::is_double<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxCOMPLEX);
	else if (cuarma::is_float<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxCOMPLEX);
	else if (cuarma::is_integer<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(0, 0, mxINT32_CLASS, mxCOMPLEX);
	else
	{
		return NULL;
	}

	int n_row = cuarmaMat.size1(), n_col = cuarmaMat.size2();
	if (n_row == 0 || n_col == 0)
	{
		mexErrMsgIdAndTxt(errId, noDataAvailableErr);
		return NULL;
	}

	ScalarType * real_pointer = (ScalarType*)mxCalloc(n_row*n_col, sizeof(ScalarType));
	ScalarType * imag_pointer = (ScalarType*)mxCalloc(n_row*n_col, sizeof(ScalarType));
	//cudaMemcpy(dst_pointer, cuarmaMat.handle().cuda_handle().get(), n_row*n_col*sizeof(ScalarType), cudaMemcpyDeviceToHost);
	for (int j = 0; j < n_col; j++)
	{
		for (int i = 0; i < n_row; i++)
		{
			real_pointer[j*n_row + i] = cuarmaMat(i, j);
			imag_pointer[j*n_row + i] = ScalarType(0.0);
		}
	}


	mxSetData(matlabMatrix, imag_pointer);
	mxSetImagData(matlabMatrix, real_pointer);

	mxSetM(matlabMatrix, n_row);
	mxSetN(matlabMatrix, n_col);

	return matlabMatrix;
}

/************************************
// MethodName:    cuarmaSetPr
// Description:
// Author:        YangXian
// Date:          2017/10/19 16:07
// Parameter:     const cuarma::matrix<double> & cuarmaMat
// Returns:       mxArray *
**************************************/
inline mxArray * cuarmaSetPr(const cuarma::matrix<double, cuarma::column_major>& cuarmaMat)
{
	return cuarmaSetData<double>(cuarmaMat);
}


/************************************
// MethodName:    cuarmaSetPi
// Description:
// Author:        YangXian
// Date:          2017/10/19 16:07
// Parameter:     const cuarma::matrix<double> & cuarmaMat
// Returns:       mxArray *
**************************************/
inline mxArray * cuarmaSetPi(const cuarma::matrix<double, cuarma::column_major>& cuarmaMat)
{
	return cuarmaSetImagData<double>(cuarmaMat);
}

/************************************
// MethodName:    cuarmaSetComplex
// Description:   cuarma复数矩阵传递给mxArray
// Author:        YangXian
// Date:          2017/10/19 16:07
// Parameter:     const cuarma::matrix<ScalarType> & cuarmaMat
// Returns:       mxArray *
**************************************/
template<typename ScalarType>
inline mxArray * cuarmaSetComplex(const cuarma::matrix<ScalarType, cuarma::column_major>& cuarmaMat)
{
	int n_row = cuarmaMat.size1(), n_col = cuarmaMat.size2();
	if ((n_row == 0) || (n_col == 0))
	{
		mexErrMsgIdAndTxt(errId, noDataAvailableErr);
		return NULL;
	}

	if ((n_col % 2) != 0)
	{
		mexErrMsgIdAndTxt(errId, "Column number of complex cuarma matrix should be even.");
		return NULL;
	}

	mxArray * matlabMatrix;
	n_col = n_col / 2;

	if (cuarma::is_double<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(n_row, n_col, mxDOUBLE_CLASS, mxCOMPLEX);
	else if (cuarma::is_float<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(n_row, n_col, mxSINGLE_CLASS, mxCOMPLEX);
	else if (cuarma::is_integer<ScalarType>::value)
		matlabMatrix = mxCreateNumericMatrix(n_row, n_col, mxINT32_CLASS, mxCOMPLEX);
	else
	{
		return NULL;
	}

	ScalarType *  real_part = (ScalarType *)mxCalloc(n_row*n_col, sizeof(ScalarType));
	ScalarType *  imag_part = (ScalarType *)mxCalloc(n_row*n_col, sizeof(ScalarType));

	for (int j = 0; j < n_col; j++)
	{
		for (int i = 0; i < n_row; i++)
		{
			real_part[j*n_row + i] = cuarmaMat(i, 2 * j);
			imag_part[j*n_row + i] = cuarmaMat(i, 2 * j + 1);
		}
	}

	mxSetData(matlabMatrix, real_part);
	mxSetImagData(matlabMatrix, imag_part);

	return matlabMatrix;
}

/************************************
// MethodName:    cuarmaWriteDataToMatFile
// Description:   将cuarma实数矩阵写入到mat文件指定变量中
// Author:        YangXian
// Date:          2017/10/20 13:42
// Parameter:     const char * filename
// Parameter:     cuarma::matrix<ScalarType
// Parameter:     cuarma::column_major> cuarmaMat
// Parameter:     const char * var
// Returns:       int
**************************************/
template<typename ScalarType>
inline int cuarmaWriteDataToMatFile(const char* filename, cuarma::matrix<ScalarType,cuarma::column_major> cuarmaMat, const char * var)
{
	MATFile *file = matOpen(filename, "wz");
	int result;
	if (NULL==file)
	{
		mexErrMsgIdAndTxt(errId, matFileCantOpen);
		return 0;
	}
	else
	{
		mxArray *temp = cuarmaSetData<ScalarType>(cuarmaMat);
		result = matPutVariable(file, var, temp);
		mxDestroyArray(temp);
	}
	matClose(file);
	return result;
}

/************************************
// MethodName:    cuarmaWritePrToMatFile
// Description:   将cuarma实数矩阵写入到mat文件指定变量中，双精度double型
// Author:        YangXian
// Date:          2017/10/20 14:53
// Parameter:     const char * filename
// Parameter:     cuarma::matrix<double
// Parameter:     cuarma::column_major> cuarmaMat
// Parameter:     const char * var
// Returns:       int
**************************************/
inline int cuarmaWritePrToMatFile(const char* filename, cuarma::matrix<double, cuarma::column_major> cuarmaMat, const char * var)
{
	return cuarmaWriteDataToMatFile<double>(filename, cuarmaMat, var);
}

/************************************
// MethodName:    cuarmaWriteImagDataToMatFile
// Description:   将cuarma纯虚矩阵写入到mat文件指定变量中
// Author:        YangXian
// Date:          2017/10/20 13:42
// Parameter:     const char * filename
// Parameter:     cuarma::matrix<ScalarType
// Parameter:     cuarma::column_major> cuarmaMat
// Parameter:     const char * var
// Returns:       int
**************************************/
template<typename ScalarType>
inline int cuarmaWriteImagDataToMatFile(const char* filename, cuarma::matrix<ScalarType, cuarma::column_major> cuarmaMat, const char * var)
{
	MATFile *file = matOpen(filename, "wz");
	int result;
	if (NULL == file)
	{
		mexErrMsgIdAndTxt(errId, matFileCantOpen);
		return 0;
	}
	else
	{
		mxArray *temp = cuarmaSetImagData<ScalarType>(cuarmaMat);
		result = matPutVariable(file, var, temp);
		mxDestroyArray(temp);
	}
	matClose(file);
	return result;
}

/************************************
// MethodName:    cuarmaWritePiToMatFile
// Description:   将cuarma纯虚矩阵写入到mat文件指定变量中，双精度double型
// Author:        YangXian
// Date:          2017/10/20 14:53
// Parameter:     const char * filename
// Parameter:     cuarma::matrix<double
// Parameter:     cuarma::column_major> cuarmaMat
// Parameter:     const char * var
// Returns:       int
**************************************/
inline int cuarmaWritePiToMatFile(const char* filename, cuarma::matrix<double, cuarma::column_major> cuarmaMat, const char * var)
{
	return cuarmaWriteImagDataToMatFile<double>(filename, cuarmaMat, var);
}

/************************************
// MethodName:    cuarmaWriteComplexToMatFile
// Description:   将cuarma复数矩阵写入到mat文件指定变量中
// Author:        YangXian
// Date:          2017/10/20 13:44
// Parameter:     const char * filename
// Parameter:     cuarma::matrix<ScalarType
// Parameter:     cuarma::column_major> cuarmaMat
// Parameter:     const char * var
// Returns:       int
**************************************/
template<typename ScalarType>
inline int cuarmaWriteComplexToMatFile(const char* filename, cuarma::matrix<ScalarType, cuarma::column_major> cuarmaMat, const char * var)
{
	MATFile *file = matOpen(filename, "wz");
	int result;
	if (NULL == file)
	{
		mexErrMsgIdAndTxt(errId, matFileCantOpen);
		return 0;
	}
	else
	{
		mxArray *temp = cuarmaSetComplex<ScalarType>(cuarmaMat);
		result = matPutVariable(file, var, temp);
		mxDestroyArray(temp);
	}
	matClose(file);
	return result;
}

/************************************
// MethodName:    cuarmaReadDataFromMatFile
// Description:   从mat文件中读取指定变量矩阵的实数部分
// Author:        YangXian
// Date:          2017/10/20 13:59
// Parameter:     const char * filename
// Parameter:     const char * var
// Returns:       
**************************************/
template<typename ScalarType>
inline cuarma::matrix<ScalarType, cuarma::column_major> cuarmaReadDataFromMatFile(const char * filename,const char * var)
{
	MATFile * file = matOpen(filename, "r");
	if (NULL == file)
	{
		mexErrMsgIdAndTxt(errId, matFileCantOpen);
		return cuarma::matrix<ScalarType, cuarma::column_major>();
	}
	cuarma::matrix<ScalarType, cuarma::column_major> cuarmaMat = cuarmaGetData<ScalarType>(matGetVariable(file, var));
	matClose(file);
	return cuarmaMat;
}

/************************************
// MethodName:    cuarmaReadPrFromMatFile
// Description:   从mat文件中读取指定变量矩阵的实数部分-双精度double型
// Author:        YangXian
// Date:          2017/10/20 14:52
// Parameter:     const char * filename
// Parameter:     const char * var
// Returns:       cuarma::matrix<ScalarType, cuarma::column_major>
**************************************/
inline cuarma::matrix<ScalarType, cuarma::column_major> cuarmaReadPrFromMatFile(const char * filename, const char * var)
{
	return cuarmaReadDataFromMatFile<double>(filename, var);
}

/************************************
// MethodName:    cuarmaReadImagDataFromMatFile
// Description:   从mat文件中读取指定变量矩阵的虚数部分
// Author:        YangXian
// Date:          2017/10/20 13:59
// Parameter:     const char * filename
// Parameter:     const char * var
// Returns:       
**************************************/
template<typename ScalarType>
inline cuarma::matrix<ScalarType, cuarma::column_major> cuarmaReadImagDataFromMatFile(const char * filename, const char * var)
{
	MATFile * file = matOpen(filename, "r");
	if (NULL == file)
	{
		mexErrMsgIdAndTxt(errId, matFileCantOpen);
		return cuarma::matrix<ScalarType, cuarma::column_major>();
	}
	cuarma::matrix<ScalarType, cuarma::column_major> cuarmaMat = cuarmaGetImagData<ScalarType>(matGetVariable(file, var));
	matClose(file);
	return cuarmaMat;
}

/************************************
// MethodName:    cuarmaReadPiFromMatFile
// Description:    从mat文件中读取指定变量矩阵的虚数部分-双精度double型
// Author:        YangXian
// Date:          2017/10/20 14:52
// Parameter:     const char * filename
// Parameter:     const char * var
// Returns:       cuarma::matrix<ScalarType, cuarma::column_major>
**************************************/
inline cuarma::matrix<ScalarType, cuarma::column_major> cuarmaReadPiFromMatFile(const char * filename, const char * var)
{
	return cuarmaReadImagDataFromMatFile<double>(filename, var);
}

/************************************
// MethodName:    cuarmaReadComplexFromMatFile
// Description:   从mat文件中读取指定变量对应的复数矩阵
// Author:        YangXian
// Date:          2017/10/20 14:00
// Parameter:     const char * filename
// Parameter:     const char * var
// Returns:       
**************************************/
template<typename ScalarType>
inline cuarma::matrix<ScalarType, cuarma::column_major> cuarmaReadComplexFromMatFile(const char * filename, const char * var)
{
	MATFile * file = matOpen(filename, "r");
	if (NULL == file)
	{
		mexErrMsgIdAndTxt(errId, matFileCantOpen);
		return cuarma::matrix<ScalarType, cuarma::column_major>();
	}
	cuarma::matrix<ScalarType, cuarma::column_major> cuarmaMat = cuarmaGetComplex<ScalarType>(matGetVariable(file, var));
	matClose(file);
	return cuarmaMat;
}