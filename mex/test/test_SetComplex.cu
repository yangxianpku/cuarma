#include <stdio.h>
#include "head_define.h"
#include "cuarmaMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Initialize the MathWorks GPU API. */
    if(mxInitGPU()==MX_GPU_FAILURE)  return;

    cuarma::matrix<double,cuarma::column_major> cuarmaMat(2,4);

    for (int i = 0; i < 2; ++i)
    {
    	for (int j = 0; j < 4; ++j)
    	{
            if((j%2)==0)
    		    cuarmaMat(i,j) = 5.26;
            else
                cuarmaMat(i,j) = 1.23; 
    	}
    }
    int n_row=cuarmaMat.size1(),n_col=cuarmaMat.size2();
    mexPrintf("%d,%d \n",n_row,n_col);

    double a;
    for (int i = 0; i < 2; ++i)
    {
        a= cuarmaMat(i,0); mexPrintf("[Real]:%f ",a);
        a= cuarmaMat(i,1); mexPrintf("[Imag]:%f ",a);
        a= cuarmaMat(i,2); mexPrintf("[Real]:%f ",a);
        a= cuarmaMat(i,3); mexPrintf("[Imag]:%f ",a);
        mexPrintf("\n");
    }

    plhs[0] = cuarmaSetComplex<double>(cuarmaMat);

	return;
}