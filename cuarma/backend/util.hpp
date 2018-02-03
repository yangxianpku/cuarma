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

/** @file cuarma/backend/util.hpp
 *  @encoding:UTF-8 文档编码
    @brief Helper functionality for working with different memory domains
*/

#include <vector>
#include <cassert>

#include "cuarma/forwards.h"
#include "cuarma/backend/mem_handle.hpp"




namespace cuarma
{
namespace backend
{
namespace detail
{

  /** @brief Helper struct for converting a type to its OpenCL pendant. */
  template<typename T>
  struct convert_to_opencl
  {
    typedef T    type;
    enum { special = 0 };
  };

} //namespace detail


/** @brief Helper class implementing an array on the host. Default case: No conversion necessary */
template<typename T, bool special = detail::convert_to_opencl<T>::special>
class typesafe_host_array
{
  typedef T                                              cpu_type;
  typedef typename detail::convert_to_opencl<T>::type    target_type;

public:
  explicit typesafe_host_array() : bytes_buffer_(NULL), buffer_size_(0) {}

  explicit typesafe_host_array(mem_handle const & handle, arma_size_t num = 0) : bytes_buffer_(NULL), buffer_size_(sizeof(cpu_type) * num)
  {
    resize(handle, num);
  }

  ~typesafe_host_array() { delete[] bytes_buffer_; }

  //
  // Setter and Getter
  //
  void * get() { return reinterpret_cast<void *>(bytes_buffer_); }
  arma_size_t raw_size() const { return buffer_size_; }
  arma_size_t element_size() const  {  return sizeof(cpu_type); }
  arma_size_t size() const { return buffer_size_ / element_size(); }
  template<typename U>
  void set(arma_size_t index, U value)
  {
    reinterpret_cast<cpu_type *>(bytes_buffer_)[index] = static_cast<cpu_type>(value);
  }

  //
  // Resize functionality
  //

  /** @brief Resize without initializing the new memory */
  void raw_resize(mem_handle const & /*handle*/, arma_size_t num)
  {
    buffer_size_ = sizeof(cpu_type) * num;

    if (num > 0)
    {
      delete[] bytes_buffer_;

      bytes_buffer_ = new char[buffer_size_];
    }
  }

  /** @brief Resize including initialization of new memory (cf. std::vector<>) */
  void resize(mem_handle const & handle, arma_size_t num)
  {
    raw_resize(handle, num);

    if (num > 0)
    {
      for (arma_size_t i=0; i<buffer_size_; ++i)
        bytes_buffer_[i] = 0;
    }
  }

  cpu_type operator[](arma_size_t index) const
  {
    assert(index < size() && bool("index out of bounds"));

    return reinterpret_cast<cpu_type *>(bytes_buffer_)[index];
  }

private:
  char * bytes_buffer_;
  arma_size_t buffer_size_;
};




/** @brief Special host array type for conversion between OpenCL types and pure CPU types */
template<typename T>
class typesafe_host_array<T, true>
{
  typedef T                                              cpu_type;
  typedef typename detail::convert_to_opencl<T>::type    target_type;

public:
  explicit typesafe_host_array() : convert_to_opencl_( (default_memory_type() == OPENCL_MEMORY) ? true : false), bytes_buffer_(NULL), buffer_size_(0) {}

  explicit typesafe_host_array(mem_handle const & handle, arma_size_t num = 0) : convert_to_opencl_(false), bytes_buffer_(NULL), buffer_size_(sizeof(cpu_type) * num)
  {
    resize(handle, num);
  }

  ~typesafe_host_array() { delete[] bytes_buffer_; }

  //
  // Setter and Getter
  //

  template<typename U>
  void set(arma_size_t index, U value)
  {

      reinterpret_cast<cpu_type *>(bytes_buffer_)[index] = static_cast<cpu_type>(value);
  }

  void * get() { return reinterpret_cast<void *>(bytes_buffer_); }
  cpu_type operator[](arma_size_t index) const
  {
    assert(index < size() && bool("index out of bounds"));

    return reinterpret_cast<cpu_type *>(bytes_buffer_)[index];
  }

  arma_size_t raw_size() const { return buffer_size_; }
  arma_size_t element_size() const
  {

    return sizeof(cpu_type);
  }
  arma_size_t size() const { return buffer_size_ / element_size(); }

  //
  // Resize functionality
  //

  /** @brief Resize without initializing the new memory */
  void raw_resize(mem_handle const & handle, arma_size_t num)
  {
    buffer_size_ = sizeof(cpu_type) * num;
    (void)handle; //silence unused variable warning if compiled without OpenCL support



    if (num > 0)
    {
      delete[] bytes_buffer_;

      bytes_buffer_ = new char[buffer_size_];
    }
  }

  /** @brief Resize including initialization of new memory (cf. std::vector<>) */
  void resize(mem_handle const & handle, arma_size_t num)
  {
    raw_resize(handle, num);

    if (num > 0)
    {
      for (arma_size_t i=0; i<buffer_size_; ++i)
        bytes_buffer_[i] = 0;
    }
  }

private:
  bool convert_to_opencl_;
  char * bytes_buffer_;
  arma_size_t buffer_size_;
};

} //backend
} //cuarma
