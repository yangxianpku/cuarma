include_directories(${Boost_INCLUDE_DIRS})

#test with CPU backend
foreach(PROG  blas3_solve fft_1d fft_2d
	global_variables
	iterators 
	matrix_col_float matrix_col_double matrix_col_int matrix_convert
	matrix_product_double matrix_product_flaot matrix_row_double matrix_row_float matrix_row_int
	matrix_vector matrix_vector_int
	nmf
	qr_method qr_method_func
	scalar scan
	scheduler_matrix scheduler_matrix_matrix scheduler_matrix_vector
	scheduler_sparse scheduler_vector
	self_assign sparse sparse_prod spmdm tql
	vector_convert vector_float_double vector_int vector_multi_inner_prod vector_uint)
	add_executable(${PROG}-test-cpu src/${PROG}.cpp)
	target_link_libraries(${PROG}-test_cpu ${Boost_LIBRARIES})
	add_test(${PROG}-cpu ${prog}-test-cpu)
endforeach(PROG)

# tests with CUDA backend
if (ENABLE_CUDA)
  foreach(PROG bisect matrix_product_float matrix_product_double blas3_solve fft_1d fft_2d iterators
               global_variables
               matrix_convert
               matrix_vector matrix_vector_int
               matrix_row_float matrix_row_double matrix_row_int
               matrix_col_float matrix_col_double matrix_col_int nmf
               scalar self_assign sparse qr_method qr_method_func scan sparse_prod tql
               vector_convert vector_float_double vector_int vector_uint vector_multi_inner_prod
               spmdm)
     cuda_add_executable(${PROG}-test-cuda src/${PROG}.cu)
     target_link_libraries(${PROG}-test-cuda ${Boost_LIBRARIES})
     add_test(${PROG}-cuda ${PROG}-test-cuda)
  endforeach(PROG)

  include_directories(${PROJECT_SOURCE_DIR}/external)
  cuda_add_executable(external_linkage-cuda
                      src/external_1.cu
                      src/external_2.cu)
  target_link_libraries(external_linkage-cuda ${Boost_LIBRARIES})
endif (ENABLE_CUDA)

include_directories(${PROJECT_SOURCE_DIR}/libcuarma/include/)

if (ENABLE_CUDA)
    cuda_add_executable(libcuarma_blas1-test src/libcuarma_blas1.cu)
    target_link_libraries(libcuarma_blas1-test cuarma)

    cuda_add_executable(libcuarma_blas2-test src/libcuarma_blas2.cu)
    target_link_libraries(libcuarma_blas2-test cuarma)

    cuda_add_executable(libcuarma_blas3-test src/libcuarma_blas3.cu)
    target_link_libraries(libcuarma_blas3-test cuarma)
else(ENABLE_CUDA)
  	message(FATAL_ERROR "CUDA is not avaliable!")
endif (ENABLE_CUDA)

add_test(libcuarma-blas1 libcuarma_blas1-test)
add_test(libcuarma-blas2 libcuarma_blas2-test)
add_test(libcuarma-blas3 libcuarma_blas3-test)