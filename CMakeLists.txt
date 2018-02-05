cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -DNDEBUG " )
find_package(OpenCL)

aux_source_directory(./darknet_cl/src DIR_LIB_SRC)
aux_source_directory(./darknet_cl/example DIR_EXAMPLE_SRC)

include_directories(./darknet_cl/include)
include_directories(./darknet_cl/clBLAS)
include_directories(${OpenCL_INCLUDE_DIR})

link_libraries(libpthread.so)
link_libraries(${OpenCL_LIBRARY})
link_libraries(${PROJECT_SOURCE_DIR}/darknet_cl/clBLAS/libclBLAS.so)

ADD_DEFINITIONS(-DGPU=1)
add_executable(darknet_cl ${DIR_LIB_SRC} ${DIR_EXAMPLE_SRC})
