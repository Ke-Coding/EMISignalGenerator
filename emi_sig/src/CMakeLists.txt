cmake_minimum_required(VERSION 3.6)

project(brownrv)
add_executable(brownrv brownrv.c)
target_link_libraries(brownrv m)

project(exponentialrv)
add_executable(exponentialrv exponentialrv.c)
target_link_libraries(exponentialrv m)

project(laplace)
add_executable(laplace laplace.c)
target_link_libraries(laplace m)

project(pinkgen)
add_executable(pinkgen pinkgen.c)
target_link_libraries(pinkgen m)

project(uniformrv)
add_executable(uniformrv uniformrv.c)
target_link_libraries(uniformrv m)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/../dist")
set(CMAKE_CXX_FLAGS "-O3 -std=c11")
