# add include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${TRTNET_COMMON_INCLUDE}
                    ${CUDA_INCLUDE_DIRS}
                    ${TENSORRT_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS})
#build run_yolov3
file(GLOB CURRENT_SRC src/*.cpp)
add_executable(run_yolov3 ${CURRENT_SRC})
target_link_libraries(run_yolov3 trtnet ${OpenCV_LIBRARIES})
install(TARGETS run_yolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)