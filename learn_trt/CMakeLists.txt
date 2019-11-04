# set trt bin and lib output path
set(TRT_LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_PATH}/learn_trt)
set(TRT_EXECUTABLE_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH}/learn_trt)
set(TRTNET_COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/trt_net/include
                          ${CMAKE_CURRENT_SOURCE_DIR}/trt_net/include/trtbasenet)

SUBDIRLIST(TRT_SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})
foreach(subdir ${TRT_SUBDIRS})
  message(STATUS "TRT_SUBDIR: ${subdir}")
  set(LIBRARY_OUTPUT_PATH ${TRT_LIBRARY_OUTPUT_PATH}/${subdir})
  set(EXECUTABLE_OUTPUT_PATH ${TRT_EXECUTABLE_OUTPUT_PATH}/${subdir})
  add_subdirectory(${subdir})
endforeach(subdir ${TRT_SUBDIRS})
