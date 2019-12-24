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
set(CMAKE_BUILD_TYPE Release)

##
## Options
##
option(USE_PERFTOOLS "Use Google perftools to monitor/improve memory use" ON)

##
## Find the required dependencies
##
find_package(Qt5 COMPONENTS Core Gui Widgets Svg REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
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
	ZoomableSvgWidget.cpp
	DataTable2D.cpp
)
target_link_libraries(nn-insight
	Qt5::Widgets Qt5::Svg
	nlohmann_json::nlohmann_json
	${CMAKE_DL_LIBS}
)
if (USE_PERFTOOLS)
target_link_libraries(nn-insight
	PkgConfig::libtcmalloc
)
endif()

##
## Subdirectories
##
subdirs(plugins)
