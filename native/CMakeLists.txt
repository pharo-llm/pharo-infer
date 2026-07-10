cmake_minimum_required(VERSION 3.20)

project(PharoInferNativeShim C)

set(LLAMA_DIR "" CACHE PATH "Path to the llama.cpp source checkout")
set(LLAMA_BUILD_DIR "" CACHE PATH "Path to the llama.cpp build directory")
set(AI_LLAMA_RPATH "" CACHE STRING "Runtime library search path for the shim")

if(NOT LLAMA_DIR)
  message(FATAL_ERROR "Set LLAMA_DIR to a llama.cpp source checkout")
endif()

if(NOT EXISTS "${LLAMA_DIR}/include/llama.h")
  message(FATAL_ERROR "Could not find llama.h under ${LLAMA_DIR}/include")
endif()

if(NOT LLAMA_BUILD_DIR)
  message(FATAL_ERROR "Set LLAMA_BUILD_DIR to the llama.cpp build directory")
endif()

set(_llama_library_search_paths
  "${LLAMA_BUILD_DIR}"
  "${LLAMA_BUILD_DIR}/bin"
  "${LLAMA_BUILD_DIR}/bin/Release"
  "${LLAMA_BUILD_DIR}/bin/RelWithDebInfo"
  "${LLAMA_BUILD_DIR}/lib"
  "${LLAMA_BUILD_DIR}/lib/Release"
  "${LLAMA_BUILD_DIR}/lib/RelWithDebInfo"
  "${LLAMA_BUILD_DIR}/src"
  "${LLAMA_BUILD_DIR}/src/Release"
  "${LLAMA_BUILD_DIR}/src/RelWithDebInfo"
)

find_library(LLAMA_LIBRARY
  NAMES llama libllama
  PATHS ${_llama_library_search_paths}
  NO_DEFAULT_PATH
)

if(NOT LLAMA_LIBRARY)
  message(FATAL_ERROR "Could not find the llama shared/import library under ${LLAMA_BUILD_DIR}")
endif()

add_library(ai_llama SHARED ai_llama_shim.c)

target_include_directories(ai_llama
  PRIVATE
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
)

target_link_libraries(ai_llama PRIVATE "${LLAMA_LIBRARY}")

set_target_properties(ai_llama PROPERTIES OUTPUT_NAME "ai_llama")

if(AI_LLAMA_RPATH)
  set_target_properties(ai_llama
    PROPERTIES
      BUILD_RPATH "${AI_LLAMA_RPATH}"
      INSTALL_RPATH "${AI_LLAMA_RPATH}"
  )
endif()

if(WIN32)
  target_compile_definitions(ai_llama PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
