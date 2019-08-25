# add cuda header and lib
set(LEARN_PERFORMANCEANALYSIS_INCLUDE_DIR include)
include_directories(${CUDA_INLUDE_DIRS}
                    ${COMMON_INCLUDE_DIR}
                    ${LEARN_PERFORMANCEANALYSIS_INCLUDE_DIR}
                    ${PROTOBUF_INCLUDE_DIRS})

link_libraries(${CUDA_LIBRARIES}
               ${PROTOBUF_LIBRARIES})
# add project source file
file(GLOB CURRENT_CUDA_SOURCES src/*.cu)
file(GLOB CURRENT_CXX_SOURCES src/*.cpp)
# build performanceanalysis lib
cuda_add_library(performanceanalysis SHARED ${CURRENT_CUDA_SOURCES} ${COMMON_SRCS})
# build learn_performanceanalysis target
link_libraries(performanceanalysis)
add_executable(learn_performanceanalysis ${CURRENT_CXX_SOURCES})