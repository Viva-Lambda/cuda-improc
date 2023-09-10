cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("CUDA-imgproc" LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# single stream gradient
add_executable(gengrad.out
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gengrad/gengrad.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgio.cpp"
    )

target_include_directories(gengrad.out PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(gengrad.out PRIVATE CUDA::cudart)
target_link_libraries(gengrad.out PRIVATE CUDA::cuda_driver)
target_link_libraries(gengrad.out PRIVATE CUDA::curand)

set_target_properties(gengrad.out PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(gengrad.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# multi stream gradient
add_executable(asyncgengrad.out 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gengrad/asyncgengrad.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgio.cpp"
    )

target_include_directories(asyncgengrad.out PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(asyncgengrad.out PRIVATE CUDA::cudart)
target_link_libraries(asyncgengrad.out PRIVATE CUDA::cuda_driver)
target_link_libraries(asyncgengrad.out PRIVATE CUDA::curand)

set_target_properties(asyncgengrad.out PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(asyncgengrad.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#
# sobel
set(ImDir "${CMAKE_CURRENT_SOURCE_DIR}/images")

add_executable(sobelx.out 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/edged/sobelx.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgio.cpp"
    )

target_include_directories(sobelx.out PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(sobelx.out PRIVATE CUDA::cudart)
target_link_libraries(sobelx.out PRIVATE CUDA::cuda_driver)
target_link_libraries(sobelx.out PRIVATE CUDA::curand)

set_target_properties(sobelx.out PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(sobelx.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(
    sobelx.out 
    PRIVATE
    IMAGE_DIR="${ImDir}"
    )

add_executable(sobely.out 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/edged/sobely.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgio.cpp"
    )

target_include_directories(sobely.out PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(sobely.out PRIVATE CUDA::cudart)
target_link_libraries(sobely.out PRIVATE CUDA::cuda_driver)
target_link_libraries(sobely.out PRIVATE CUDA::curand)

set_target_properties(sobely.out PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(sobely.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(
    sobely.out 
    PRIVATE
    IMAGE_DIR="${ImDir}"
    )


add_executable(sobelxy.out 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/edged/sobelxy.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imgio.cpp"
    )

target_include_directories(sobelxy.out PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(sobelxy.out PRIVATE CUDA::cudart)
target_link_libraries(sobelxy.out PRIVATE CUDA::cuda_driver)
target_link_libraries(sobelxy.out PRIVATE CUDA::curand)

set_target_properties(sobelxy.out PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(sobelxy.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(
    sobelxy.out 
    PRIVATE
    IMAGE_DIR="${ImDir}"
    )
