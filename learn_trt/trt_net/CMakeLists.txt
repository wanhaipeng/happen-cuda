include_directories(${CUDA_INCLUDE_DIRS}
                    ${TENSORRT_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/trtbasenet)
# link cuda and tensorRT lib
link_libraries(${CUDA_LIBRARIES}
               ${TENSORRT_LIBRARIES})
# add project source file
file(GLOB TRTNET_SOURCES src/*.cu src/*.cpp)
# build trtnet lib
cuda_add_library(trtnet STATIC ${TRTNET_SOURCES})
