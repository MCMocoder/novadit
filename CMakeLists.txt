cmake_minimum_required(VERSION 3.26)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(
    CMAKE_CUDA_ARCHITECTURES
    "80;86;89;90;100"
    CACHE STRING
    "CUDA architectures"
  )
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc")

project(NovaDiTCpp LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NOVADIT_CUDA_ROOT "/usr/local/cuda-12.9" CACHE PATH "Preferred CUDA toolkit root")
set(NOVADIT_CUDA_TARGET_ROOT "${NOVADIT_CUDA_ROOT}/targets/x86_64-linux")
set(NOVADIT_CUDA_INCLUDE_DIR "${NOVADIT_CUDA_TARGET_ROOT}/include")
set(NOVADIT_CUDA_LIB_DIR "${NOVADIT_CUDA_TARGET_ROOT}/lib")
set(NOVADIT_CUDA_NVCC "${NOVADIT_CUDA_ROOT}/bin/nvcc")
set(NOVADIT_CUDA_CUDART "${NOVADIT_CUDA_LIB_DIR}/libcudart.so")
set(NOVADIT_CUDA_HOST_COMPILER "/usr/bin/clang++" CACHE FILEPATH "Preferred CUDA host compiler")
set(NOVADIT_CLANGD_COMPDB_DIR "${CMAKE_BINARY_DIR}/clangd" CACHE PATH "Directory for clangd-friendly compile commands")

function(novadit_append_prefixed out_var prefix)
  set(result "")
  foreach(item IN LISTS ARGN)
    string(APPEND result " ${prefix}${item}")
  endforeach()
  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

function(novadit_write_clangd_compdb)
  file(MAKE_DIRECTORY "${NOVADIT_CLANGD_COMPDB_DIR}")

  set(novadit_clangd_project_includes "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
  if(NOVADIT_HAS_CUTLASS)
    list(APPEND novadit_clangd_project_includes
      "${NOVADIT_CUTLASS_INCLUDE}"
      "${NOVADIT_CUTLASS_UTIL_INCLUDE}")
  endif()

  set(novadit_clangd_system_includes
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    "${NOVADIT_CUDA_INCLUDE_DIR}")

  set(novadit_clangd_defines
    "TORCH_EXTENSION_NAME=novadit_cuda"
    "TORCH_API_INCLUDE_EXTENSION_H"
    "_GLIBCXX_USE_CXX11_ABI=${NOVADIT_TORCH_CXX11_ABI}")

  if(NOVADIT_HAS_CUTLASS)
    list(APPEND novadit_clangd_defines
      "NOVADIT_HAS_CUTLASS=1"
      "CUTE_USE_PACKED_TUPLE=1")
  endif()

  novadit_append_prefixed(novadit_clangd_include_flags "-I" ${novadit_clangd_project_includes})
  novadit_append_prefixed(novadit_clangd_system_include_flags "-isystem " ${novadit_clangd_system_includes})
  novadit_append_prefixed(novadit_clangd_define_flags "-D" ${novadit_clangd_defines})

  set(novadit_clangd_cuda_arch "90")
  if(CMAKE_CUDA_ARCHITECTURES)
    list(GET CMAKE_CUDA_ARCHITECTURES -1 novadit_clangd_cuda_arch)
  endif()
  if(novadit_clangd_cuda_arch STREQUAL "OFF")
    set(novadit_clangd_cuda_arch "90")
  endif()
  set(novadit_clangd_db "[\n")
  set(novadit_clangd_first TRUE)

  foreach(source IN LISTS NOVADIT_SOURCES)
    get_filename_component(source_abs "${source}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

    if(source IN_LIST NOVADIT_CUDA_SOURCES)
      set(command
        "${NOVADIT_CUDA_HOST_COMPILER} --cuda-host-only -xcuda --cuda-path=${NOVADIT_CUDA_ROOT} --cuda-gpu-arch=sm_${novadit_clangd_cuda_arch} --no-cuda-version-check -std=c++17 -fPIC -O3${novadit_clangd_define_flags}${novadit_clangd_include_flags}${novadit_clangd_system_include_flags} -c ${source_abs}")
    else()
      set(command
        "${CMAKE_CXX_COMPILER} -std=c++17 -fPIC -O3${novadit_clangd_define_flags}${novadit_clangd_include_flags}${novadit_clangd_system_include_flags} -c ${source_abs}")
    endif()

    string(REPLACE "\\" "\\\\" command "${command}")
    string(REPLACE "\"" "\\\"" command "${command}")

    if(NOT novadit_clangd_first)
      string(APPEND novadit_clangd_db ",\n")
    endif()

    string(APPEND novadit_clangd_db
      "  {\n"
      "    \"directory\": \"${CMAKE_BINARY_DIR}\",\n"
      "    \"file\": \"${source_abs}\",\n"
      "    \"command\": \"${command}\"\n"
      "  }")
    set(novadit_clangd_first FALSE)
  endforeach()

  string(APPEND novadit_clangd_db "\n]\n")
  file(WRITE "${NOVADIT_CLANGD_COMPDB_DIR}/compile_commands.json" "${novadit_clangd_db}")
endfunction()

if(EXISTS "${NOVADIT_CUDA_NVCC}")
  set(CMAKE_CUDA_COMPILER "${NOVADIT_CUDA_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
  set(CMAKE_CUDA_HOST_COMPILER "${NOVADIT_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler" FORCE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=${NOVADIT_CUDA_HOST_COMPILER}")
  set(CUDAToolkit_ROOT "${NOVADIT_CUDA_ROOT}" CACHE PATH "CUDA toolkit root" FORCE)
  set(CUDA_TOOLKIT_ROOT_DIR "${NOVADIT_CUDA_ROOT}" CACHE PATH "Legacy CUDA toolkit root" FORCE)
  set(CUDA_BIN_PATH "${NOVADIT_CUDA_ROOT}" CACHE PATH "Legacy CUDA bin path" FORCE)
  set(CUDA_NVCC_EXECUTABLE "${NOVADIT_CUDA_NVCC}" CACHE FILEPATH "Legacy CUDA nvcc" FORCE)
  set(CUDA_INCLUDE_DIRS "${NOVADIT_CUDA_INCLUDE_DIR}" CACHE PATH "Legacy CUDA include dirs" FORCE)
  set(CUDA_CUDART_LIBRARY "${NOVADIT_CUDA_CUDART}" CACHE FILEPATH "Legacy CUDA cudart" FORCE)
  set(ENV{CUDACXX} "${NOVADIT_CUDA_NVCC}")
  set(ENV{CUDA_PATH} "${NOVADIT_CUDA_ROOT}")
  set(ENV{CUDA_HOME} "${NOVADIT_CUDA_ROOT}")
  set(ENV{CUDA_BIN_PATH} "${NOVADIT_CUDA_ROOT}")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${NOVADIT_CUDA_ROOT}")
list(PREPEND CMAKE_LIBRARY_PATH "${NOVADIT_CUDA_LIB_DIR}")
list(PREPEND CMAKE_INCLUDE_PATH "${NOVADIT_CUDA_INCLUDE_DIR}")

set(NOVADIT_VENV_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
if(NOT DEFINED Python3_EXECUTABLE AND EXISTS "${NOVADIT_VENV_PYTHON}")
  set(Python3_EXECUTABLE "${NOVADIT_VENV_PYTHON}" CACHE FILEPATH "Preferred Python interpreter" FORCE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
  set(
    TORCH_CUDA_ARCH_LIST
    "8.0;8.6;8.9;9.0;10.0;10.3;11.0;12.0;12.1+PTX"
    CACHE STRING
    "CUDA arch list for Torch-dependent builds"
  )
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sys, torch; sys.stdout.write(torch.utils.cmake_prefix_path)"
  RESULT_VARIABLE NOVADIT_TORCH_PREFIX_RESULT
  OUTPUT_VARIABLE NOVADIT_TORCH_CMAKE_PREFIX
  ERROR_VARIABLE NOVADIT_TORCH_PREFIX_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT NOVADIT_TORCH_PREFIX_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Failed to import torch with ${Python3_EXECUTABLE}.\n"
    "Select a Python environment that already has PyTorch installed before running CMake.\n"
    "Original error:\n${NOVADIT_TORCH_PREFIX_ERROR}")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${NOVADIT_TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED CONFIG)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sys, torch; sys.stdout.write(str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))"
  RESULT_VARIABLE NOVADIT_TORCH_ABI_RESULT
  OUTPUT_VARIABLE NOVADIT_TORCH_CXX11_ABI
  ERROR_VARIABLE NOVADIT_TORCH_ABI_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT NOVADIT_TORCH_ABI_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Failed to query torch C++ ABI setting from ${Python3_EXECUTABLE}.\n"
    "Original error:\n${NOVADIT_TORCH_ABI_ERROR}")
endif()

set(NOVADIT_CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
set(NOVADIT_CUTLASS_INCLUDE "${NOVADIT_CUTLASS_ROOT}/include")
set(NOVADIT_CUTLASS_UTIL_INCLUDE "${NOVADIT_CUTLASS_ROOT}/tools/util/include")
set(NOVADIT_HAS_CUTLASS OFF)

if(EXISTS "${NOVADIT_CUTLASS_INCLUDE}/cutlass" AND EXISTS "${NOVADIT_CUTLASS_INCLUDE}/cute")
  set(NOVADIT_HAS_CUTLASS ON)
endif()

set(NOVADIT_CUDA_SOURCES
  csrc/ops/adaln_kernel.cu
  csrc/ops/adarmsnorm_kernel.cu
  csrc/ops/fc1_bias_act_kernel.cu
)

set(NOVADIT_HOST_SOURCES
  csrc/bindings.cpp
  csrc/ops/qkv_pack_kernel.cu
  csrc/ops/residual_kernel.cu
  csrc/ops/gate_act_kernel.cu
)

set(NOVADIT_SOURCES
  ${NOVADIT_HOST_SOURCES}
  ${NOVADIT_CUDA_SOURCES}
)

set_source_files_properties(
  csrc/ops/qkv_pack_kernel.cu
  csrc/ops/residual_kernel.cu
  csrc/ops/gate_act_kernel.cu
  PROPERTIES
    LANGUAGE CXX
)

add_library(novadit_cuda_objects OBJECT ${NOVADIT_SOURCES})
target_compile_features(novadit_cuda_objects PUBLIC cxx_std_17)
set(NOVADIT_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  ${Python3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS})
target_include_directories(
  novadit_cuda_objects
  PRIVATE
    ${NOVADIT_INCLUDE_DIRS}
)
target_link_libraries(
  novadit_cuda_objects
  PRIVATE
    ${TORCH_LIBRARIES}
    Python3::Module
    CUDA::cudart
)
target_compile_definitions(
  novadit_cuda_objects
  PRIVATE
    TORCH_EXTENSION_NAME=novadit_cuda
    TORCH_API_INCLUDE_EXTENSION_H
    _GLIBCXX_USE_CXX11_ABI=${NOVADIT_TORCH_CXX11_ABI}
)
target_compile_options(
  novadit_cuda_objects
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
)

if(NOVADIT_HAS_CUTLASS)
  list(APPEND NOVADIT_INCLUDE_DIRS
    "${NOVADIT_CUTLASS_INCLUDE}"
    "${NOVADIT_CUTLASS_UTIL_INCLUDE}")
  target_include_directories(
    novadit_cuda_objects
    PRIVATE
      "${NOVADIT_CUTLASS_INCLUDE}"
      "${NOVADIT_CUTLASS_UTIL_INCLUDE}"
  )
  target_compile_definitions(
    novadit_cuda_objects
    PRIVATE
      NOVADIT_HAS_CUTLASS=1
      CUTE_USE_PACKED_TUPLE=1
  )
endif()

novadit_write_clangd_compdb()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES not set; using compiler default")
endif()

message(STATUS "Using CUDA root: ${NOVADIT_CUDA_ROOT}")
message(STATUS "Using CUDA nvcc: ${NOVADIT_CUDA_NVCC}")
message(STATUS "Using CUDA include dir: ${NOVADIT_CUDA_INCLUDE_DIR}")
message(STATUS "Using CUDA cudart: ${NOVADIT_CUDA_CUDART}")
message(STATUS "Using Python interpreter: ${Python3_EXECUTABLE}")
message(STATUS "Using Torch CMake prefix: ${NOVADIT_TORCH_CMAKE_PREFIX}")
message(STATUS "Using Torch ABI setting: ${NOVADIT_TORCH_CXX11_ABI}")
message(STATUS "CUTLASS enabled: ${NOVADIT_HAS_CUTLASS}")
message(STATUS "Generated clangd compile database: ${NOVADIT_CLANGD_COMPDB_DIR}/compile_commands.json")
