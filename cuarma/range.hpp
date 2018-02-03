#pragma once

/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                            -----------------
             CuArma - COE of Peking University, Shaoqiang Tang.
                            -----------------

                  Author Email    yangxianpku@pku.edu.cn

            Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/** @file range.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Implementation of a range object for use with proxy objects
*/

#include <vector>
#include <stddef.h>
#include <assert.h>
#include "cuarma/forwards.h"

namespace cuarma
{

/** @brief A range class that refers to an interval [start, stop), where 'start' is included, and 'stop' is excluded.
 *
 * Similar to the boost::numeric::ublas::basic_range class.
 */
template<typename SizeT /* see forwards.h for default argument*/,typename DistanceT /* see forwards.h for default argument*/>
class basic_range
{
public:
  typedef SizeT             size_type;
  typedef DistanceT         difference_type;
  typedef size_type            value_type;
  typedef value_type           const_reference;
  typedef const_reference      reference;

  basic_range() : start_(0), size_(0) {}
  basic_range(size_type start_index, size_type stop_index) : start_(start_index), size_(stop_index - start_index)
  {
    assert(start_index <= stop_index);
  }


  size_type start() const { return start_; }
  size_type size() const { return size_; }

  const_reference operator()(size_type i) const
  {
    assert(i < size());
    return start_ + i;
  }
  const_reference operator[](size_type i) const { return operator()(i); }

  bool operator==(const basic_range & r) const { return (start_ == r.start_) && (size_ == r.size_); }
  bool operator!=(const basic_range & r) const { return !(*this == r); }

private:
  size_type start_;
  size_type size_;
};


}
