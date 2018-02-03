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

/** @file cuarma/meta/enable_if.hpp
 *  @encoding:UTF-8 文档编码
    @brief Simple enable-if variant that uses the SFINAE pattern
*/

namespace cuarma
{

/** @brief Simple enable-if variant that uses the SFINAE pattern */
template<bool b, class T = void>
struct enable_if
{
  typedef T   type;
};

/** \cond */
template<class T>
struct enable_if<false, T> {};
/** \endcond */

} //namespace cuarma