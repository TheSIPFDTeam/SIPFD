
set(COMMON_SRC ${PROJECT_SOURCE_DIR}/external/curvemodel.c
  ${PROJECT_SOURCE_DIR}/external/fpx.c
  ${PROJECT_SOURCE_DIR}/external/vowgcs.c
  ${PROJECT_SOURCE_DIR}/external/fp_asm_.s)

add_library(precomp STATIC ${COMMON_SRC})
target_compile_features(precomp PUBLIC)

if(NOT ARCH)
    add_definitions(-D_x64_)
endif()

