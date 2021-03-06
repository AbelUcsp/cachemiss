cmake_minimum_required(VERSION 2.8)

project(ballpivoting)


SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

option(USE_CLANG "build application with clang" OFF) # OFF is the default


if(USE_CLANG)
    SET (CMAKE_CXX_COMPILER "/usr/bin/clang++")
    add_definitions(-DUSE_CLANG)
else(USE_CLANG)
    SET (CMAKE_CXX_COMPILER "/usr/bin/g++")
    SET(CMAKE_CXX_FLAGS "-ansi -fopenmp")
endif(USE_CLANG)

MESSAGE(${CMAKE_CXX_COMPILER})
MESSAGE(${CMAKE_CXX_FLAGS})


add_executable(ballpivoting main.cpp
			src/Facet.cpp
			src/Edge.cpp
			src/Vertex.cpp
			src/Point.cpp
			src/Mesher.cpp
			src/FileIO.cpp)

