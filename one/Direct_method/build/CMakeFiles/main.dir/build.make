# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

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
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.21.3_1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.21.3_1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/main.cpp.o -MF CMakeFiles/main.dir/src/main.cpp.o.d -o CMakeFiles/main.dir/src/main.cpp.o -c /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/src/main.cpp

CMakeFiles/main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/src/main.cpp > CMakeFiles/main.dir/src/main.cpp.i

CMakeFiles/main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/src/main.cpp -o CMakeFiles/main.dir/src/main.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src/main.cpp.o
main: CMakeFiles/main.dir/build.make
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_stitching.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_superres.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_videostab.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_aruco.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_bgsegm.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_bioinspired.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_ccalib.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_dnn_objdetect.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_dpm.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_face.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_freetype.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_fuzzy.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_hfs.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_img_hash.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_line_descriptor.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_optflow.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_reg.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_rgbd.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_saliency.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_sfm.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_stereo.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_structured_light.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_surface_matching.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_tracking.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_xfeatures2d.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_ximgproc.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_xobjdetect.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_xphoto.3.4.15.dylib
main: libdirect.a
main: libpic_process.a
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_highgui.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_videoio.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_shape.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_phase_unwrapping.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_dnn.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_video.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_datasets.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_ml.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_plot.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_imgcodecs.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_objdetect.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_calib3d.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_features2d.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_flann.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_photo.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_imgproc.3.4.15.dylib
main: /opt/homebrew/Cellar/opencv@3/3.4.15/lib/libopencv_core.3.4.15.dylib
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/build /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/build /Users/pijunyin/Documents/GitHub/Dec_slam/one/Direct_method/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

