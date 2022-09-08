cmake_minimum_required(VERSION 3.13)

project(isogeny LANGUAGES CUDA C ASM)

set(CMAKE_VERBOSE_MAKEFILE OFF)

find_package(OpenMP REQUIRED)

# ccache is good but optional
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  message(STATUS "ccache:yes")
else()
  message(STATUS "ccache:no")
endif()

# define default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)
endif()

IF (OPENMP_FOUND)
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_C_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS} -no-pie")
ENDIF()

## set CFLAGS depending on build type
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")

message(STATUS "CMAKE_BUILD_TYPE:${CMAKE_BUILD_TYPE}")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR:${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR:${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR:${PROJECT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# define default architecture type if not specified
if(NOT ARCH)
  set(ARCH "64")
endif()

# define default cuda architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# architecture
message(STATUS "ARCH:${ARCH}")
# Montgomery model is by default 
message(STATUS "MODEL:Montgomery")
message(STATUS "CUDA ARCHITECTURE:${CMAKE_CUDA_ARCHITECTURES}")

# generate include/config.h
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/include/config.h
  BYPRODUCTS ${PROJECT_SOURCE_DIR}/include/config.h
  DEPENDS ${PROJECT_SOURCE_DIR}/setup.py
  COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_SOURCE_DIR}/ env python3 setup.py --arch=${ARCH} -c > ${PROJECT_SOURCE_DIR}/include/config.h
  COMMENT "Generated config.h")

# generate src/constants.cu
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/src/constants.cu
  BYPRODUCTS ${PROJECT_SOURCE_DIR}/src/constants.h
  DEPENDS ${PROJECT_SOURCE_DIR}/setup.py
  COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_SOURCE_DIR}/ env python3 setup.py --arch=${ARCH} -s > ${PROJECT_SOURCE_DIR}/src/constants.cu
  COMMENT "Generated constants.cu")

# generate src/rng.cu
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/src/rng.cu
  BYPRODUCTS ${PROJECT_SOURCE_DIR}/src/rng.cu
  DEPENDS ${PROJECT_SOURCE_DIR}/setup.py
  COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_SOURCE_DIR}/ env python3 setup.py --arch=${ARCH} -r > ${PROJECT_SOURCE_DIR}/src/rng.cu
  COMMENT "Generated rng.cu")

# generate include/vowgcs_setup.h
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/include/vowgcs_setup.h
  BYPRODUCTS ${PROJECT_SOURCE_DIR}/include/vowgcs_setup.h
  DEPENDS ${PROJECT_SOURCE_DIR}/setup.py
  COMMAND ${CMAKE_COMMAND} -E chdir ${PROJECT_SOURCE_DIR}/ env python3 setup.py --arch=${ARCH} -v > ${PROJECT_SOURCE_DIR}/include/vowgcs_setup.h
  COMMENT "Generated vowgcs_setup.h")

add_custom_target(config
  DEPENDS ${PROJECT_SOURCE_DIR}/src/rng.cu ${PROJECT_SOURCE_DIR}/include/config.h
  ${PROJECT_SOURCE_DIR}/src/constants.cu ${PROJECT_SOURCE_DIR}/include/vowgcs_setup.h
  COMMENT "Files generated")

add_custom_target(vowgcs_setup
  DEPENDS ${PROJECT_SOURCE_DIR}/include/vowgcs_setup.h
  COMMENT "vOW gcs setup generated")

set(COMMON_SOURCES ${PROJECT_SOURCE_DIR}/src/rng.cu
  ${PROJECT_SOURCE_DIR}/src/fp.cu 
  ${PROJECT_SOURCE_DIR}/src/fpx.cu
  ${PROJECT_SOURCE_DIR}/src/mont/curvemodel.cu 
  ${PROJECT_SOURCE_DIR}/src/mont/utils.cu
  ${PROJECT_SOURCE_DIR}/src/constants.cu
  ${PROJECT_SOURCE_DIR}/src/instance.cu)

add_library(gpu STATIC ${COMMON_SOURCES})
target_compile_features(gpu PUBLIC)
set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# BEGIN  PRECOMP

set(COMMON_SRC ${PROJECT_SOURCE_DIR}/external/curvemodel.c
  ${PROJECT_SOURCE_DIR}/external/fpx.c
  ${PROJECT_SOURCE_DIR}/external/vowgcs.c
  ${PROJECT_SOURCE_DIR}/external/fp_asm_.s)

add_library(precomp STATIC ${COMMON_SRC})
target_compile_features(precomp PRIVATE)
set_target_properties(precomp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(precomp PRIVATE ${PROJECT_SOURCE_DIR}/external)

if(NOT ARCH)
    add_definitions(-D_x64_)
endif()

# END  PRECOMP

# arith test
add_executable(arith_tests ${PROJECT_SOURCE_DIR}/src/arith_tests.cu
    ${PROJECT_SOURCE_DIR}/src/arith_tests_mont.cu
    ${PROJECT_SOURCE_DIR}/src/arith_benchs_mont.cu)
set_target_properties(arith_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(arith_tests PRIVATE gpu)

add_definitions(-D_mont_)

if(BFUNC)
  message(STATUS "BENCHMARK FUNCTION (BFUNC): ON")
  add_definitions(-DNFNC)
else()
  message(STATUS "BENCHMARK FUNCTION (BFUNC): OFF")
endif()

if(NOT MITM)
  message(STATUS "ALGORITHM: vOW")
  add_definitions(-D_vowgcs_)
  set(ALGORITHM "vowgcs")
else()
  message(STATUS "ALGORITHM: MITM")
  add_definitions(-D_mitm_)
  set(ALGORITHM "mitm")
endif()

set(FILENAME "${ALGORITHM}_mont.cu")

add_executable(${ALGORITHM} ${PROJECT_SOURCE_DIR}/src/${FILENAME} 
  ${PROJECT_SOURCE_DIR}/src/${ALGORITHM}.cu)
set_target_properties(${ALGORITHM} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${ALGORITHM} gpu precomp)

# testing mont PHONY target
add_custom_target(check_arith
  COMMAND arith_tests -t
  DEPENDS arith_tests)

# benchmarking PHONY target
add_custom_target(bench
  COMMAND arith_tests -b
  DEPENDS arith_tests)

