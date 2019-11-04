#ifndef YOLO_LAYER_H_
#define YOLO_LAYER_H_

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include "NvInfer.h"
#include "utils.h"
#include "yoloConfigs.h"

namespace Yolo {
  struct YoloKernel;
  static constexpr int LOCATIONS = 4; // 常量表达式允许程序利用编译时的计算能力，只在编译的时候做一次计算，避免在运行时候做计算，提升效率
  struct alignas(float) Detection{ // alignas 内存对齐，结构体size不到4字节占用4字节
    // x y w h
    float bbox[LOCATIONS];
    // float objectness;
    int classId;
    float prob;
  };
}


namespace nvinfer1 {
  /**
   * @brief The trt plugin of yolo detection layer
   */
  class YoloLayerPlugin: public IPluginExt {
   public:
    /**
     * @brief Create yolo detection plugin with cudaThread
     *
     * @param cudaThread cuda runtime thread number
     */
    explicit YoloLayerPlugin(const int cudaThread = 512);
    
    /**
     * @brief Create the plugin at runtime from a byte stream
     *
     * @param data input runtime byte stream
     * @param length byte stream size
     */
    YoloLayerPlugin(const void* data, size_t length);

    /**
     * YoloLayer destructor
     */
    ~YoloLayerPlugin();

    /**
     * @brief Get network output tensor number
     */
    int getNbOutputs() const override {
      return 1;
    }

    /**
     * @brief Get output tensor dims
     * 
     * @param index unknown
     * @param inputs input tensor dims
     * @param nbInputDims unknown
     */
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    /**
     * @brief Decide whether to support current datatype && plugin format.
     */
    bool supportsFormat(DataType type, PluginFormat format) const override { 
      return type == DataType::kFLOAT && format == PluginFormat::kNCHW; 
    }

    /**
     * @brief Configure layer with input datatype && plugintype.
     * 
     * @param inputDims unknown
     * @param nbInputs unknown
     * @param outputDims unknown
     * @param nbOutputs unknown
     * @param type calculated data type
     * @param format layer plugin format
     * @param maxBatchSize layer inference max batch size
     */
    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                             int nbOutputs, DataType type, PluginFormat format,
                             int maxBatchSize) override {};

    /**
     * @brief Yolo layer init
     */
    int initialize() override;

    virtual void terminate() override {};

    /**
     * @brief Get layer workspace mem size
     */
    virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

    /**
     * @brief Do layer inference, user defined execution.
     * 
     * @param batchsize inference batchsize
     * @param inputs the ptr of input data ptr
     * @param outputs layer output ptr
     * @param workspace unknown
     * @param stream cuda runtime stream
     */
    virtual int enqueue(int batchSize, const void*const * inputs,
                        void** outputs, void* workspace,
                        cudaStream_t stream) override;

    /**
     * @brief Get the serialization data size
     */
    virtual size_t getSerializationSize() override;

    /**
     * @brief Serialize the param of layer into cache.
     */
    virtual void serialize(void* buffer) override;

    /**
     * @brief Yolo layer gpu forward
     * 
     * @param inputs layer input datas
     * @param output layer output data
     * @param stream cuda runtime stream
     * @param batchSize forward batchsize
     */
    void forwardGpu(const float *const * inputs, float * output,
                    cudaStream_t stream, int batchSize = 1);

    /**
     * @brief Yolo layer cpu forward
     * 
     * @param inputs layer input datas
     * @param output layer output data
     * @param stream cuda runtime stream
     * @param batchSize forward batchsize
     */
    void forwardCpu(const float *const * inputs, float * output,
                    cudaStream_t stream, int batchSize = 1);

   private:
    int mClassCount; /// yolo detection classes number
    int mKernelCount; /// yolo detection level count
    std::vector<Yolo::YoloKernel> mYoloKernel; /// yolo layer kernel config
    int mThreadCount; /// cuda runtime thread number
    void* mInputBuffer  {nullptr}; /// layer input data ptr
    void* mOutputBuffer {nullptr}; /// layer output data ptr
  };
};

#endif 
