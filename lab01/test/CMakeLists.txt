

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
include_directories(
        ${CMAKE_SOURCE_DIR}/include)
add_executable( lab1_gemm_test
        gemm_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        )
target_link_libraries( lab1_gemm_test
        PRIVATE
        GTest::gtest_main
        )

# TODO: add three tests for each of GEMM, GEMV, and NN
add_executable( lab1_gemv_test
        gemv_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        )
target_link_libraries( lab1_gemv_test
        PRIVATE
        GTest::gtest_main
        )  


add_executable( denseNN_test
        denseNN_test.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/src/dense_nn.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
)
target_link_libraries( denseNN_test
        PRIVATE
        GTest::gtest_main
        )    

include(GoogleTest)
gtest_discover_tests(lab1_gemm_test)
gtest_discover_tests(lab1_gemv_test)
gtest_discover_tests(denseNN_test)
