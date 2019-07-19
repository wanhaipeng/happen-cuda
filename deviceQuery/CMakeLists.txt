# add cuda header and lib
include_directories(${CUDA_INLUDE_DIRS}
                    ${CUDA_COMMON_INCLUDE_DIR}
                    ${COMMON_INCLUDE_DIR})
link_libraries(${CUDA_LIBRARIES})
# add project source file
set(SRC src/deviceQuery.cu)
# build deviceQuery
cuda_add_executable(deviceQuery ${SRC})