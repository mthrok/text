cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)


add_library(torchtext SHARED IMPORTED GLOBAL)
set_target_properties(torchtext PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../../torchtext/_torchtext.so")

add_library(libpython SHARED IMPORTED GLOBAL)
set_target_properties(libpython PROPERTIES IMPORTED_LOCATION /home/moto/conda/envs/PY3.8-cuda101/lib/libpython3.so)

message("${TORCH_LIBRARIES}")
add_executable(benchmark-jit_pipeline benchmark_jit_pipeline.cpp)
target_link_libraries(benchmark-jit_pipeline -Wl,--no-as-needed torchtext)
target_link_libraries(benchmark-jit_pipeline libpython)
target_link_libraries(benchmark-jit_pipeline "${TORCH_LIBRARIES}")
