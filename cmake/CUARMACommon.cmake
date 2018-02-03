
include(CTest)
include(CMakeDependentOption)
include(AddCCompilerFlagIfSupported)
include(AddCLinkerFlagIfSupported)

# Installation directories
##########################

set(INSTALL_INCLUDE_DIR include CACHE PATH
   "Installation directory for headers")
if(WIN32 AND NOT CYGWIN)
   set(DEF_INSTALL_CMAKE_DIR CMake)
else()
   set(DEF_INSTALL_CMAKE_DIR lib/cmake/viennacl)
endif()
set(INSTALL_CMAKE_DIR "${DEF_INSTALL_CMAKE_DIR}" CACHE PATH
   "Installation directory for CMake files")

if(NOT IS_ABSOLUTE "${INSTALL_CMAKE_DIR}")
   set(INSTALL_CMAKE_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_CMAKE_DIR}")
endif()
file(RELATIVE_PATH CONF_REL_INSTALL_PREFIX "${INSTALL_CMAKE_DIR}"
   "${CMAKE_INSTALL_PREFIX}")
if(NOT IS_ABSOLUTE "${INSTALL_INCLUDE_DIR}")
   set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}")
endif()
file(RELATIVE_PATH CONF_REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

# User options
##############

option(ENABLE_CUDA "Use the CUDA backend" ON)

option(BUILD_EXAMPLES "Build example programs" ON)

option(ENABLE_ASAN "Build with address sanitizer if available" OFF)


# If you want to build the examples that use boost::numeric::ublas, enable
# the following:
cmake_dependent_option(ENABLE_UBLAS "Enable examples using uBLAS" OFF
   BUILD_EXAMPLES OFF)

option(ENABLE_PEDANTIC_FLAGS "Enable pedantic compiler flags (GCC and Clang only)" OFF)

mark_as_advanced(BOOSTPATH ENABLE_ASAN ENABLE_PEDANTIC_FLAGS)

# Find prerequisites
####################

# Boost:
IF (BOOSTPATH)
 SET(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH}" "${BOOSTPATH}")
 SET(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH}" "${BOOSTPATH}/lib")
 SET(BOOST_ROOT "${BOOSTPATH}")
ENDIF (BOOSTPATH)


if(ENABLE_UBLAS OR BUILD_TESTING)
   set(Boost_USE_MULTITHREADED TRUE)
   find_package(Boost)
   if (Boost_MINOR_VERSION LESS 34)
     find_package(Boost REQUIRED COMPONENTS thread)
   elseif (Boost_MINOR_VERSION LESS 47)
     find_package(Boost REQUIRED COMPONENTS date_time serialization system thread)
   else ()
     find_package(Boost REQUIRED COMPONENTS chrono date_time serialization system thread)
   endif()
endif()

if (ENABLE_CUDA)
   find_package(CUDA REQUIRED)
   set(CUDA_ARCH_FLAG "-arch=sm_30" CACHE STRING "Use one out of sm_13, sm_20, sm_30, ...")
   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${CUDA_ARCH_FLAG}" "-DCUARMA_WITH_CUDA")
endif(ENABLE_CUDA)


if (ENABLE_ASAN)
  add_c_compiler_flag_if_supported("-fsanitize=address")
  add_c_linker_flag_if_supported("-fsanitize=address")
endif(ENABLE_ASAN)

include_directories("${PROJECT_SOURCE_DIR}")


# Set high warning level on GCC
if(ENABLE_PEDANTIC_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -Wconversion")
endif()

# Disable Warning 4996 (std::copy is unsafe ...) on Visual Studio
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
endif()


# Export
########
configure_file(cmake/FindCUDA.cmake
   "${PROJECT_BINARY_DIR}/FindCUDA.cmake" COPYONLY)

configure_file(cmake/CUARMAConfig.cmake.in
   "${PROJECT_BINARY_DIR}/CUARMAConfig.cmake" @ONLY)

configure_file(cmake/CUARMAConfigVersion.cmake.in
   "${PROJECT_BINARY_DIR}/CUARMAConfigVersion.cmake" @ONLY)

if (CMAKE_MINOR_VERSION GREATER 6)  # export(PACKAGE ...) introduced with CMake 2.8.0
  export(PACKAGE CUARMA)
endif()

# Install
#########
install(FILES
   "${PROJECT_BINARY_DIR}/FindCUDA.cmake"
   "${PROJECT_BINARY_DIR}/CUARMAConfig.cmake"
   "${PROJECT_BINARY_DIR}/CUARMAConfigVersion.cmake"
   DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)


# For out-of-the-box support on MacOS:
IF(${CMAKE_SYSTEM_NAME} MATCHES "YANGXIAN")
  EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE YANGXIAN_VERSION)
  STRING(REGEX MATCH "[0-9]+" YANGXIAN_VERSION ${YANGXIAN_VERSION})
  IF (YANGXIAN_VERSION GREATER 12)
    IF (ENABLE_CUDA)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")  # Mavericks and beyond need the new C++ STL with CUDA
    ENDIF()
  ENDIF()
  INCLUDE_DIRECTORIES("/opt/local/include")
  SET(CMAKE_EXE_LINKER_FLAGS "-framework CUDA")
  set(CMAKE_MACOSX_RPATH 1) # Required for newer versions of CMake on MacOS X: http://www.kitware.com/blog/home/post/510
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "YANGXIAN")
