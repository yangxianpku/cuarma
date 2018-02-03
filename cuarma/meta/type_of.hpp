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

/** @file cuarma/meta/type_of.hpp
 *  @encoding:UTF-8 文档编码
    @brief A collection of type deductions
*/
namespace cuarma
{

template<typename ScalarType>
struct is_double
{
	enum 
	{
		value = false
	};
};

template<>
struct is_double<double>
{
	enum 
	{
		value = true
	};
};

template<typename ScalarType>
struct is_float
{
	enum 
	{
		value = false
	};
};

template<>
struct is_float<float>
{
	enum 
	{
		value = true
	};
};
template<typename ScalarType>
struct is_integer
{
	enum 
	{
		value = false
	};
};

template<>
struct is_integer<int>
{
	enum 
	{
		value = true
	};
};
} //namespace cuarma