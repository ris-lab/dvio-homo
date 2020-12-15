cmake_minimum_required(VERSION 2.8.12)
project(dvio-homo)

# set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_definitions(-DEUROC)
add_definitions(-DMS_CONVERTER)
# message("Build type: " ${CMAKE_BUILD_TYPE})
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -msse2 -msse4.1 -Wno-deprecated-declarations -Wno-unused-variable -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -msse2 -msse4.1 -Wno-deprecated-declarations -Wno-unused-variable -std=c++11 -O3")

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
	find_package(OpenCV 2.4.3 QUIET)
	if(NOT OpenCV_FOUND)
		message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
	endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system filesystem)
find_package(GFlags REQUIRED)
find_package(kindr)

include_directories(
	include
    ${kindr_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${Boost_INCLUDE_DIR}
	${GFLAGS_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
        src/setting_config.cpp
		src/Utilities.cpp
		src/State.cpp
		src/ImageUtil.cpp
        )

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${GLOG_LIBRARIES}
    ${Boost_LIBRARIES}
    )

add_executable(main test/main.cc)
target_link_libraries(main
	${OpenCV_LIBS}
	${GLOG_LIBRARIES}
    ${PROJECT_NAME}
	${Boost_LIBRARIES}
	${GFLAGS_LIBRARIES}
	)
