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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aldo/Documentos/TCC2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aldo/Documentos/TCC2/src

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/demo.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/demo.cpp.o: demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aldo/Documentos/TCC2/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo.dir/demo.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/demo.cpp.o -c /home/aldo/Documentos/TCC2/src/demo.cpp

CMakeFiles/demo.dir/demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/demo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aldo/Documentos/TCC2/src/demo.cpp > CMakeFiles/demo.dir/demo.cpp.i

CMakeFiles/demo.dir/demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/demo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aldo/Documentos/TCC2/src/demo.cpp -o CMakeFiles/demo.dir/demo.cpp.s

CMakeFiles/demo.dir/ModelLoader.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/ModelLoader.cpp.o: ModelLoader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aldo/Documentos/TCC2/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/demo.dir/ModelLoader.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/ModelLoader.cpp.o -c /home/aldo/Documentos/TCC2/src/ModelLoader.cpp

CMakeFiles/demo.dir/ModelLoader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/ModelLoader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aldo/Documentos/TCC2/src/ModelLoader.cpp > CMakeFiles/demo.dir/ModelLoader.cpp.i

CMakeFiles/demo.dir/ModelLoader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/ModelLoader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aldo/Documentos/TCC2/src/ModelLoader.cpp -o CMakeFiles/demo.dir/ModelLoader.cpp.s

CMakeFiles/demo.dir/DetectionPostProcess.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/DetectionPostProcess.cpp.o: DetectionPostProcess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aldo/Documentos/TCC2/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/demo.dir/DetectionPostProcess.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/DetectionPostProcess.cpp.o -c /home/aldo/Documentos/TCC2/src/DetectionPostProcess.cpp

CMakeFiles/demo.dir/DetectionPostProcess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/DetectionPostProcess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aldo/Documentos/TCC2/src/DetectionPostProcess.cpp > CMakeFiles/demo.dir/DetectionPostProcess.cpp.i

CMakeFiles/demo.dir/DetectionPostProcess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/DetectionPostProcess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aldo/Documentos/TCC2/src/DetectionPostProcess.cpp -o CMakeFiles/demo.dir/DetectionPostProcess.cpp.s

CMakeFiles/demo.dir/FaceDetection.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/FaceDetection.cpp.o: FaceDetection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aldo/Documentos/TCC2/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/demo.dir/FaceDetection.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/FaceDetection.cpp.o -c /home/aldo/Documentos/TCC2/src/FaceDetection.cpp

CMakeFiles/demo.dir/FaceDetection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/FaceDetection.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aldo/Documentos/TCC2/src/FaceDetection.cpp > CMakeFiles/demo.dir/FaceDetection.cpp.i

CMakeFiles/demo.dir/FaceDetection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/FaceDetection.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aldo/Documentos/TCC2/src/FaceDetection.cpp -o CMakeFiles/demo.dir/FaceDetection.cpp.s

CMakeFiles/demo.dir/evm.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/evm.cpp.o: evm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aldo/Documentos/TCC2/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/demo.dir/evm.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/evm.cpp.o -c /home/aldo/Documentos/TCC2/src/evm.cpp

CMakeFiles/demo.dir/evm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/evm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aldo/Documentos/TCC2/src/evm.cpp > CMakeFiles/demo.dir/evm.cpp.i

CMakeFiles/demo.dir/evm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/evm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aldo/Documentos/TCC2/src/evm.cpp -o CMakeFiles/demo.dir/evm.cpp.s

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/demo.cpp.o" \
"CMakeFiles/demo.dir/ModelLoader.cpp.o" \
"CMakeFiles/demo.dir/DetectionPostProcess.cpp.o" \
"CMakeFiles/demo.dir/FaceDetection.cpp.o" \
"CMakeFiles/demo.dir/evm.cpp.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

demo: CMakeFiles/demo.dir/demo.cpp.o
demo: CMakeFiles/demo.dir/ModelLoader.cpp.o
demo: CMakeFiles/demo.dir/DetectionPostProcess.cpp.o
demo: CMakeFiles/demo.dir/FaceDetection.cpp.o
demo: CMakeFiles/demo.dir/evm.cpp.o
demo: CMakeFiles/demo.dir/build.make
demo: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
demo: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aldo/Documentos/TCC2/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: demo

.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /home/aldo/Documentos/TCC2/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aldo/Documentos/TCC2 /home/aldo/Documentos/TCC2 /home/aldo/Documentos/TCC2/src /home/aldo/Documentos/TCC2/src /home/aldo/Documentos/TCC2/src/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo.dir/depend

