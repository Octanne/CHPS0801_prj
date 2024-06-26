# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /apps/spack/0.16.0/opt/spack/linux-rhel7-skylake_avx512/gcc-10.2.0/cmake-3.18.4-kw3wpvkxke7lo3atixjcvt4roclfopip/bin/cmake

# The command to remove a file.
RM = /apps/spack/0.16.0/opt/spack/linux-rhel7-skylake_avx512/gcc-10.2.0/cmake-3.18.4-kw3wpvkxke7lo3atixjcvt4roclfopip/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/build-gpu

# Include any dependencies generated for this target.
include CMakeFiles/kokkos_jacobi.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/kokkos_jacobi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kokkos_jacobi.dir/flags.make

CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.o: CMakeFiles/kokkos_jacobi.dir/flags.make
CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.o: ../jacobi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/build-gpu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.o"
	/apps/2021/gcc/10.2/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.o -c /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/jacobi.cpp

CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.i"
	/apps/2021/gcc/10.2/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/jacobi.cpp > CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.i

CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.s"
	/apps/2021/gcc/10.2/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/jacobi.cpp -o CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.s

CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.o: CMakeFiles/kokkos_jacobi.dir/flags.make
CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.o: ../gaussianNoise.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/build-gpu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.o"
	/apps/2021/gcc/10.2/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.o -c /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/gaussianNoise.cpp

CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.i"
	/apps/2021/gcc/10.2/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/gaussianNoise.cpp > CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.i

CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.s"
	/apps/2021/gcc/10.2/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/gaussianNoise.cpp -o CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.s

# Object files for target kokkos_jacobi
kokkos_jacobi_OBJECTS = \
"CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.o" \
"CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.o"

# External object files for target kokkos_jacobi
kokkos_jacobi_EXTERNAL_OBJECTS =

kokkos_jacobi: CMakeFiles/kokkos_jacobi.dir/jacobi.cpp.o
kokkos_jacobi: CMakeFiles/kokkos_jacobi.dir/gaussianNoise.cpp.o
kokkos_jacobi: CMakeFiles/kokkos_jacobi.dir/build.make
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_gapi.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_highgui.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_ml.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_objdetect.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_photo.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_stitching.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_video.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_videoio.so.4.5.1
kokkos_jacobi: /apps/2021/spack/0.17.0/opt/spack/linux-rhel7-skylake_avx512/gcc-10.2.0/kokkos-3.4.01-bngy6344ddjox34ktqilmykznuufydug/lib64/libkokkoscontainers.so.3.4.01
kokkos_jacobi: /apps/2021/spack/0.17.0/opt/spack/linux-rhel7-skylake_avx512/gcc-10.2.0/kokkos-3.4.01-bngy6344ddjox34ktqilmykznuufydug/lib64/libkokkoscore.so.3.4.01
kokkos_jacobi: /usr/lib64/libdl.so
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_dnn.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_imgcodecs.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_calib3d.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_features2d.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_flann.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_imgproc.so.4.5.1
kokkos_jacobi: /apps/2021/opencv/4.5.1/lib64/libopencv_core.so.4.5.1
kokkos_jacobi: CMakeFiles/kokkos_jacobi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/build-gpu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable kokkos_jacobi"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kokkos_jacobi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kokkos_jacobi.dir/build: kokkos_jacobi

.PHONY : CMakeFiles/kokkos_jacobi.dir/build

CMakeFiles/kokkos_jacobi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kokkos_jacobi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kokkos_jacobi.dir/clean

CMakeFiles/kokkos_jacobi.dir/depend:
	cd /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/build-gpu && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/build-gpu /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/build-gpu /home/groubahiefissa/chps/chps0801_projet/kokkos/parallel/test/build-gpu/CMakeFiles/kokkos_jacobi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kokkos_jacobi.dir/depend

