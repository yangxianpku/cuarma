include_directories(${Boost_INCLUDE_DIRS})

foreach(PROG bandwith-reduction warp-host-buffer)
    add_executable(${PROG}-example-cpu ${PROG}.cpp)
    target_link_libraries(${PROG}-example_cpu ${Boost_LIBRARIES})
    add_test(${PROG}-cpu ${prog}-example-cpu)
endforeach(PROG)


# tests with CUDA backend
if (ENABLE_CUDA)
  foreach(PROG  bisect blas1 blas2 blas3 blas3range custom-cuda dense_blas direct_solve fft
                iterative-custom iterative least-squares libcuarma matrix-free matrix-range nmf 
                power-iter qr qr_method scheduler sparse solver sparse tql2 vector-io vector-range 
                wrap-cuda-buffer)
     cuda_add_executable(${PROG}-example-cuda ${PROG}.cu)
     target_link_libraries(${PROG}-example-cuda ${Boost_LIBRARIES})
     add_test(${PROG}-cuda ${PROG}-example-cuda)
  endforeach(PROG)
endif (ENABLE_CUDA)
