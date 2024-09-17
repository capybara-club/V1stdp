cmake_minimum_required(VERSION 3.30)

project(stdp)

add_executable(
	${PROJECT_NAME}
	stdp.cpp
)

target_include_directories(
	${PROJECT_NAME}
	PRIVATE
	thirdparty/eigen
	.
)

target_compile_definitions(${PROJECT_NAME} PRIVATE "CMAKE_SOURCE_PATH=${CMAKE_CURRENT_SOURCE_DIR}")
