cmake_minimum_required(VERSION 2.8)

project(Circle_in_square)
SET("OpenCV_DIR" "/usr/local/include/")
find_package( OpenCV REQUIRED )

MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
