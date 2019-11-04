# add cuda header and lib
set(LEARN_DEVICEQUERY_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS}
                    ${CUDA_COMMON_INCLUDE_DIR}
                    ${COMMON_INCLUDE_DIR}
                    ${LEARN_DEVICEQUERY_INCLUDE_DIR})

link_libraries(${CUDA_LIBRARIES})
# add project source file
file(GLOB CURRENT_CUDA_SOURCES src/*.cu)
file(GLOB CURRENT_CXX_SOURCES src/*.cpp)
# build basicCalculations lib
cuda_add_library(basicCalculation STATIC ${CURRENT_CUDA_SOURCES})
# build learn_basicCalculatuon target
link_libraries(basicCalculation)
add_executable(learn_basicCalculatuon ${CURRENT_CXX_SOURCES})
