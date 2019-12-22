
#find_package(Flatbuffers REQUIRED) # fails to set variables: https://github.com/google/flatbuffers/issues/5576
#MESSAGE(STATUS "FLATBUFFERS_FOUND=${FLATBUFFERS_FOUND}")
#MESSAGE(STATUS "YURI FLATBUFFERS_INCLUDE_DIRS=${FLATBUFFERS_INCLUDE_DIRS}")
#include_directories(${FLATBUFFERS_INCLUDE_DIRS})

include_directories(${CMAKE_INSTALL_PREFIX}/include)

add_library(tf-lite-plugin SHARED
  tf-lite.cpp
)
