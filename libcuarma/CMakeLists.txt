include_directories(${PROJECT_SOURCE_DIR}/libcuarma/include/)

if(ENABLE_CUDA)
  cuda_add_library(cuarma SHARED  src/backend.cu
                                  src/blas1.cu src/blas1_host.cu src/blas1_cuda.cu
                                  src/blas2.cu src/blas2_host.cu src/blas2_cuda.cu
                                  src/blas3.cu src/blas3_host.cu src/blas3_cuda.cu)
  set_target_properties(cuarma PROPERTIES COMPILE_FLAGS "-DCUARMA_WITH_CUDA")
endif(ENABLE_CUDA)