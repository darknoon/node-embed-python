cmake_minimum_required(VERSION 3.10)
project (node-embed-python)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
if (NOT MSVC)
  set(CMAKE_C_STANDARD 11)
endif()

# Takes care of finding the Python libs we need to compile
find_package(Python3 COMPONENTS Development NumPy)

# Allow VSCode to find our headers (writes build/compile_commands.json)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include anything cmake-js reccomends
include_directories(${CMAKE_JS_INC})

# Our source files are in src/
file(GLOB SOURCE_FILES "src/*.h" "src/*.cpp")

# Compile anything cmake-js recommends
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})

# Set our extension to .node so we can be loaded
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Allow using napi.h N-API C++ wrapper
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/node_modules/node-addon-api")
target_include_directories(${PROJECT_NAME} PRIVATE "${Python3_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PRIVATE "${Python3_NumPy_INCLUDE_DIRS}")

target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})

message(STATUS "Include for NumPy" ${Python3_NumPy_INCLUDE_DIRS})
message(STATUS "Linking Python 3" ${Python3_LIBRARY_DIRS})
#link_directories(${Python3_LIBRARY_DIRS})
