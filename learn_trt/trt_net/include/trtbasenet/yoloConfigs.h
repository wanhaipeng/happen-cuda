#ifndef YOLO_CONFIGS_H_
#define YOLO_CONFIGS_H_

namespace Yolo {
  static constexpr int CHECK_COUNT = 3;
  static constexpr float IGNORE_THRESH = 0.5f;
  static constexpr int CLASS_NUM = 80;

  struct YoloKernel
  {
      int width;
      int height;
      float anchors[CHECK_COUNT*2];
  };
  extern YoloKernel yolo1;
  extern YoloKernel yolo2;
  extern YoloKernel yolo3;
}

#endif
