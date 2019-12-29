cmake_minimum_required(VERSION 3.1.0)

project(nn-insight)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

##
## C++ standard that we use is the highest possible
##
set(CMAKE_CXX_FLAGS "-std=c++2a -Wall")

##
## Build type
##
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

##
## Some required locations
##
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

##
## Options
##
option(USE_PERFTOOLS "Use Google perftools to monitor/improve memory use" ON)

##
## Find the required dependencies
##
find_package(Qt5 COMPONENTS Core Gui Widgets Svg REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(png++ REQUIRED)
find_package(PNG REQUIRED)
find_program(DOT_PROGRAM NAMES dot)
if (NOT DOT_PROGRAM)
	message(FATAL_ERROR "Failed to find dot")
else()
	message(STATUS "Found the dot program: ${DOT_PROGRAM}")
endif()
if (USE_PERFTOOLS)
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(libtcmalloc REQUIRED IMPORTED_TARGET libtcmalloc)
add_definitions(-DUSE_PERFTOOLS)
endif()

message(STATUS "PNG++ found: includes ${png++_INCLUDE_DIRS}, libs ${png++_LIBRARIES}")

include_directories(${png++_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/3rdparty) # for avir/avir.h to be found

##
## Targets
##
add_executable(nn-insight
	main.cpp main-window.cpp nn-model-viewer.cpp
	plugin-manager.cpp
	plugin-interface.cpp
	util.cpp
	fonts.cpp
	nn-types.cpp
	model-functions.cpp
	render-model-to-coordinates.cpp
	svg-graphics-generator.cpp
	zoomable-svg-widget.cpp
	DataTable2D.cpp
	scale-image-widget.cpp
	image.cpp
	compute.cpp
	3rdparty/tensorflow/tflite-reference-implementation.cpp
	resources.qrc
)
target_link_libraries(nn-insight
	Qt5::Widgets Qt5::Svg
	nlohmann_json::nlohmann_json
	${png++_LIBRARIES}
	${CMAKE_DL_LIBS}
)
if (USE_PERFTOOLS)
target_link_libraries(nn-insight
	PkgConfig::libtcmalloc
)
endif()

if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Release")
	add_definitions(-DDEBUG) # -DNDEBUG is turned on by cmake, but only for some compilers (?)
endif()

##
## Subdirectories
##
subdirs(plugins)

##
## Install targets
##

install(TARGETS nn-insight DESTINATION bin)
