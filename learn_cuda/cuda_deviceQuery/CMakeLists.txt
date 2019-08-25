# add cuda header and lib
set(LEARN_DEVICEQUERY_INCLUDE_DIR include)
include_directories(${CUDA_INLUDE_DIRS}
                    ${CUDA_COMMON_INCLUDE_DIR}
                    ${COMMON_INCLUDE_DIR}
                    ${LEARN_DEVICEQUERY_INCLUDE_DIR})

link_libraries(${CUDA_LIBRARIES})
# add project source file
file(GLOB CURRENT_CUDA_SOURCES src/*.cu)
file(GLOB CURRENT_CXX_SOURCES src/*.cpp)
# build deviceQuery lib
cuda_add_library(devicequery STATIC ${CURRENT_CUDA_SOURCES})
# build learn_deviceQuery target
link_libraries(devicequery)
add_executable(learn_deviceQuery ${CURRENT_CXX_SOURCES})