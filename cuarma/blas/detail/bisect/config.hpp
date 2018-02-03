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



/** @file cuarma/blas/detail//bisect/config.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Global configuration parameters
 *
 *  Implementation based on the sample provided with the CUDA 6.0 SDK, for which
 *  the creation of derivative works is allowed by including the following statement:
 *  "This software contains source code provided by NVIDIA Corporation."            
 */

// should be power of two
#define  CUARMA_BISECT_MAX_THREADS_BLOCK                256
#define  CUARMA_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX   512 // change to 256 if errors occur
#define  CUARMA_BISECT_MAX_SMALL_MATRIX                 512 // change to 256 if errors occur
#define  CUARMA_BISECT_MIN_ABS_INTERVAL                 5.0e-37

