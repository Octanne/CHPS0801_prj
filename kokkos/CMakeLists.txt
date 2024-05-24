cmake_minimum_required(VERSION 3.16)
project(KokkosStencilJacobi)
include(../../common.cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Define source files including gaussianNoise.cpp
set(SOURCES jacobi.cpp gaussianNoise.cpp)

#force kokkos to use OpenMP
#SET(Kokkos_ENABLE_SERIAL OFF CACHE BOOL "" FORCE)
#SET(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
#SET(Kokkos_ARCH_PASCAL60 ON CACHE BOOL "" FORCE)

# Add executable with source files
add_executable(kokkos_jacobi ${SOURCES})

# Add compilation flags
target_compile_options(kokkos_jacobi PRIVATE
    -g                 # Generate debug information
)

# Link Kokkos library
target_link_libraries(kokkos_jacobi Kokkos::kokkos)

# Link OpenCV libraries
target_link_libraries(kokkos_jacobi ${OpenCV_LIBS})
