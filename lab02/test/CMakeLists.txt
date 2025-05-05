

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Configuring for Debug build.")
        # Set the build type to Debug
        set(CMAKE_BUILD_TYPE Debug)

        # Add extra debug flags for Linux (GCC or Clang)
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# else()
#         message(STATUS "Configuring for Release build.")
#         # Set normal build flags for non-Debug builds
#         set(CMAKE_CXX_FLAGS_RELEASE "-O3") # Example: Optimization flag for Release
endif()

add_executable( lab2_spmm_test
        spmm_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemm.cpp
        ${CMAKE_SOURCE_DIR}/src/spmm.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
        )
target_link_libraries( lab2_spmm_test
        PRIVATE
        GTest::gtest_main
        )

add_executable( lab2_spmv_test
        spmv_test.cpp
        ${CMAKE_SOURCE_DIR}/src/gemv.cpp
        ${CMAKE_SOURCE_DIR}/src/spmv.cpp
)
target_link_libraries( lab2_spmv_test
        PRIVATE
        GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(lab2_spmm_test)
gtest_discover_tests(lab2_spmv_test)
