cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)


set(TORCHTEXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../torchtext")

add_library(torchtext SHARED IMPORTED GLOBAL)
set_target_properties(
  torchtext
  PROPERTIES
  IMPORTED_LOCATION "${TORCHTEXT_DIR}/_torchtext.so"
)

add_library(libpython SHARED IMPORTED GLOBAL)
set_target_properties(
  libpython
  PROPERTIES
  IMPORTED_LOCATION
  /home/moto/conda/envs/PY3.8-cuda101/lib/libpython3.so
)

message("${TORCH_LIBRARIES}")
add_executable(benchmark-jit_pipeline benchmark_jit_pipeline.cpp)
target_include_directories(benchmark-jit_pipeline PUBLIC "${TORCHTEXT_DIR}/csrc")
target_link_libraries(benchmark-jit_pipeline torchtext)
target_link_libraries(benchmark-jit_pipeline "${TORCH_LIBRARIES}")
target_link_libraries(benchmark-jit_pipeline libpython)
