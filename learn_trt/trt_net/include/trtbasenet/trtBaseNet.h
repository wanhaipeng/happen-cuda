#ifndef TRT_BASENET_H_
#define TRT_BASENET_H_

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>
#include "utils.h"
#include "pluginFactory.h"

/**
 * @brief happen TrtNet namespace.
 * construct basic tensorRT network
 */
namespace TrtNet {
  /**
   * @brief tensorRT inference precision mode
   */
  enum class RUN_MODE {
    FLOAT32 = 0,
    FLOAT16 = 1,
    INT8 = 2
  };
  /**
   * @brief A base network which can do general trt inference
   */
  class TrtBaseNet {
   public:
    /**
     * @brief Create trtbasenet from caffe model.
     *
     * @param prototxt caffe prototxt path
     * @param caffemodel caffe model path
     * @param outputnodesname network output nodes name array
     * @param calibratordata int8 model calibration dataset
     * @param mode network run mode(default=FP32)
     * @param maxbatchsize network max input batch(default=1)
     */
    TrtBaseNet(const std::string& prototxt, const std::string& caffemodel,
               const std::vector<std::string>& outputnodesname, const std::vector<std::vector<float>>& calibratordata,
               RUN_MODE mode = RUN_MODE::FLOAT32, int maxbatchsize = 1);
    
    /**
     * @brief Create trtbasenet from trt engine
     * 
     * @param trtenginefile input trt engine file path
     */
    explicit TrtBaseNet(const std::string& trtenginefile);
    
    /**
     * @brief TrtBaseNet destructor
     */
    ~TrtBaseNet();
    
    /**
     * @brief Save trt engine
     * 
     * @param filename output engine filename
     */
    void saveEngine(std::string filename);

    /**
     * @brief Do trt engine inference.
     * 
     * @param inputdata network input data
     * @param outputdata network output data
     * @param batchsize inference input batchsize(default=1)
     */
    void doInference(const void* inputdata, void* outputdata,
                     int batchsize = 1);

    /**
     * @brief Get network input data size.
     * 
     * @return network input data size
     */
    size_t getInputSize();

    /**
     * @brief Get network output data size.
     * 
     * @return network output data size
     */
    size_t getOutputSize();

    /**
     * @brief Print const time.
     */
    void printTime();

    /**
     * @brief Get network runtime batchsize.
     * 
     * @return runtime batchsize
     */
    size_t getBatchSize();
   private:
    /**
     * @brief Create trt engine
     */
    nvinfer1::ICudaEngine* loadModelAndCreateEngine(
        const char* deployfile,
        const char* modelfile,
        int maxbatchsize,
        nvcaffeparser1::ICaffeParser* parser,
        nvcaffeparser1::IPluginFactory* pluginFactory,
        nvinfer1::IInt8Calibrator* calibrator,
        nvinfer1::IHostMemory*& trtModelStream,
        const std::vector<std::string>& outputnodesname);
    
    /**
     * @brief Init trt engine
     */
    void InitEngine();
    nvinfer1::IExecutionContext* mTrtContext; /// trt context
    nvinfer1::ICudaEngine* mTrtEngine; /// trt model engine
    nvinfer1::IRuntime* mTrtRunTime; /// trt runtime
    PluginFactory mTrtPluginFactory; /// trt layer plugin
    cudaStream_t mTrtCudaStream; /// cuda stream
    Profiler mTrtProfiler; /// trt profiler
    RUN_MODE mTrtRunMode; /// trt runmode
    std::vector<void*> mTrtCudaBuffer; /// trt cuda buffer
    std::vector<int64_t> mTrtBindBufferSize; /// trt bind buffer size
    int mTrtInputCount; /// input number
    int mTrtIterationTime; /// run iteration number
    int mTrtBatchSize; /// run batch size
  };
}
#endif // TRT_BASENET_H_
