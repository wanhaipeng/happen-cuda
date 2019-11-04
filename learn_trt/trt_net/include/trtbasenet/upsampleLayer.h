#ifndef UPSAMPLE_LAYER_H_
#define UPSAMPLE_LAYER_H_

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "utils.h"
#include <iostream>

namespace nvinfer1 {
  /**
   * @brief The trt plugin of upsample layer
   */
  class UpsampleLayerPlugin : public IPluginExt {
   public:
    /**
     * @brief Create upsample plugin with param: scale，and cudaThread num
     * 
     * @param scale upsmaple layer scale value.
     * @param cudaThread cuda runtime thread num.
     */
    explicit UpsampleLayerPlugin(const float scale, const int cudaThread = 512);
    
    /**
     * @brief create the plugin at runtime from a byte stream
     * 
     * @param data input runtime byte stream
     * @param length byte stream size
     */
    UpsampleLayerPlugin(const void* data, size_t length);
    
    /**
     * @brief UpsampleLayerPlugin destructor
     */
    ~UpsampleLayerPlugin();

    /**
     * @brief Get network output tensor number.
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
      // std::cout << "supportsFormat === type:"  << int(type) << "format" << int(format) << std::endl;
      return (type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8 ) 
        && format == PluginFormat::kNCHW; 
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
                             int maxBatchSize) override;

    /**
     * @brief Upsample layer init.
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
    virtual size_t getSerializationSize() override {
      return sizeof(nvinfer1::Dims) + sizeof(mDataType) + sizeof(mScale)
        + sizeof(mOutputWidth) + sizeof(mOutputHeight) + sizeof(mThreadCount);
    }

    /**
     * @brief Serialize the param of layer into cache.
     */
    virtual void serialize(void* buffer) override;

    /**
     * @brief Upsample layer forward template implementation.
     * 
     * @param input layer input data
     * @param output layer output data
     * @param NCHW layer input shape
     */
    template <typename Dtype>
      void forwardGpu(const Dtype* input, Dtype * output ,int N,int C,int H ,int W);

   private:
    nvinfer1::Dims mCHW; /// layer input shape
    DataType mDataType{DataType::kFLOAT}; /// layer input data type
    float mScale; /// upsample layer scale param
    int mOutputWidth; /// output width
    int mOutputHeight; /// output height
    int mThreadCount; /// cuda thread count
    void* mInputBuffer  {nullptr}; /// input data ptr
    void* mOutputBuffer {nullptr}; /// output data ptr
  };
};

#endif // UPSAMPLE_LAYER_H_
