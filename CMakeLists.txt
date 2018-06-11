cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -DNDEBUG " )

set(OPENCV  ON)
set(GPU     ON)

find_package(OpenCL)
find_package(OpenCV)

if (OpenCL_FOUND AND GPU)
	ADD_DEFINITIONS(-DGPU=1)
	include_directories(${OpenCL_INCLUDE_DIR})
	link_libraries(${OpenCL_LIBRARY})
	link_libraries(${PROJECT_SOURCE_DIR}/darknet_cl/clBLAS/libclBLAS.so)
endif()

if (OpenCV_FOUND AND OPENCV)
	ADD_DEFINITIONS(-DOPENCV)
	include_directories(${OpenCV_INCLUDE_DIR})
	link_libraries(${OpenCV_LIBRARIES})
endif()

aux_source_directory(./darknet_cl/src DIR_LIB_SRC)
aux_source_directory(./darknet_cl/example DIR_EXAMPLE_SRC)

include_directories(./darknet_cl/include)
include_directories(./darknet_cl/clBLAS)

link_libraries(libpthread.so)

add_executable(darknet ${DIR_LIB_SRC} ${DIR_EXAMPLE_SRC})
