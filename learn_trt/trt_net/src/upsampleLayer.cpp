#include "upsampleLayer.h"

namespace nvinfer1 {
  UpsampleLayerPlugin::UpsampleLayerPlugin(const float scale, const int cudaThread /*= 512*/)
   : mScale(scale),mThreadCount(cudaThread) {
  }
  
  UpsampleLayerPlugin::~UpsampleLayerPlugin() {
  }
  
  // create the plugin at runtime from a byte stream
  UpsampleLayerPlugin::UpsampleLayerPlugin(const void* data, size_t length) {
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    TrtNet::read(d, mCHW);
    TrtNet::read(d, mDataType);
    TrtNet::read(d, mScale);
    TrtNet::read(d, mOutputWidth);
    TrtNet::read(d, mOutputHeight);
    TrtNet::read(d, mThreadCount);
    // std::cout << "read:" << a << " " << mOutputWidth<< " " <<mOutputHeight<<std::endl;
    assert(d == a + length);
  }
  
  void UpsampleLayerPlugin::serialize(void* buffer) {
    char* d = static_cast<char*>(buffer), *a = d;
    TrtNet::write(d, mCHW);
    TrtNet::write(d, mDataType);
    TrtNet::write(d, mScale);
    TrtNet::write(d, mOutputWidth);
    TrtNet::write(d, mOutputHeight);
    TrtNet::write(d, mThreadCount);
    // std::cout << "write:" << a << " " << mOutputHeight<< " " <<mOutputWidth<<std::endl;
    assert(d == a + getSerializationSize());
  }
  
  int UpsampleLayerPlugin::initialize() {
    int inputHeight = mCHW.d[1];
    int inputWidth = mCHW.d[2];
    // upsample output tensor shape
    mOutputHeight = inputHeight * mScale;
    mOutputWidth = inputWidth * mScale;
    return 0;
  }
  
  void UpsampleLayerPlugin::configureWithFormat(
      const Dims* inputDims, int nbInputs,
      const Dims* outputDims, int nbOutputs,
      DataType type, PluginFormat format, int maxBatchSize) {
    // std::cout << "type " << int(type) << "format " << (int)format <<std::endl;
    assert((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8) && format == PluginFormat::kNCHW);
    mDataType = type;
    //std::cout << "configureWithFormat:" <<inputDims[0].d[0]<< " " <<inputDims[0].d[1] << " "<<inputDims[0].d[2] <<std::endl;
  }
  
  // it is called prior to any call to initialize().
  Dims UpsampleLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    // std::cout <<"Input:" << inputs[0].d[0] << " "<<inputs[0].d[1]<< " "<<inputs[0].d[2]<<std::endl;
    // std::cout <<"nbInputDims : "<<nbInputDims<< " input:" << inputs[0].nbDims << std::endl;
    mCHW = inputs[0];
    mOutputHeight = inputs[0].d[1] * mScale;
    mOutputWidth = inputs[0].d[2] * mScale;
    // std::cout << "ouputDims:" << mCHW.d[0] << " " << mOutputHeight << " " << mOutputWidth << std::endl;
    return Dims3(mCHW.d[0], mOutputHeight, mOutputWidth);
  }
}
