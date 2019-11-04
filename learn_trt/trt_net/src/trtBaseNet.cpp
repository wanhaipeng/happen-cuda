#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include "trtBaseNet.h"
#include "entroyCalibrator.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static TrtNet::Logger gLogger;

#define RETURN_AND_LOG(ret, severity, message)                                 \
    do                                                                         \
    {                                                                          \
      std::string error_message = "ssd_error_log: " + std::string(message);    \
      gLogger.log(ILogger::Severity::k##severity, error_message.c_str());      \
      return (ret);                                                            \
    } while (0)

inline void* safeCudaMalloc(size_t memSize) { // malloc gpu mem.
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline int64_t volume(const nvinfer1::Dims& d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

namespace TrtNet {
  TrtBaseNet::TrtBaseNet(const std::string& prototxt,
                         const std::string& caffemodel,
                         const std::vector<std::string>& outputNodesName,
                         const std::vector<std::vector<float>>& calibratorData,
                         RUN_MODE mode /*= RUN_MODE::FLOAT32*/,
                         int maxBatchSize /*= 1*/)
    : mTrtContext(nullptr), mTrtEngine(nullptr),
      mTrtRunTime(nullptr), mTrtRunMode(mode),
      mTrtInputCount(0),mTrtIterationTime(0),
      mTrtBatchSize(maxBatchSize) {
    std::cout << "init plugin proto: " << prototxt
              << " caffemodel: " << caffemodel << std::endl;
    auto parser = createCaffeParser(); // create caffe parser
    IHostMemory* trtModelStream = nullptr;
    Int8EntropyCalibrator2 * calibrator = nullptr;
    if (calibratorData.size() > 0 ){
      auto endPos= prototxt.find_last_of(".");
	    auto beginPos= prototxt.find_last_of('/') + 1;
      std::string calibratorName = prototxt.substr(beginPos,endPos - beginPos);
      std::cout << "create calibrator,Named:" << calibratorName << std::endl;
      calibrator = new Int8EntropyCalibrator2(maxBatchSize, calibratorData, calibratorName);
    }
    PluginFactory pluginFactorySerialize;
    ICudaEngine* tmpEngine =
      loadModelAndCreateEngine(prototxt.c_str(),caffemodel.c_str(),
                               maxBatchSize, parser, &pluginFactorySerialize,
                               calibrator, trtModelStream, outputNodesName);
    assert(tmpEngine != nullptr);
    assert(trtModelStream != nullptr);
    if(calibrator){
      delete calibrator;
      calibrator = nullptr;
    }
    tmpEngine->destroy();
    pluginFactorySerialize.destroyPlugin();
    mTrtRunTime = createInferRuntime(gLogger);     
    assert(mTrtRunTime != nullptr);
    mTrtEngine= mTrtRunTime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &mTrtPluginFactory);
    assert(mTrtEngine != nullptr);
    // Deserialize the engine.
    trtModelStream->destroy();
    InitEngine();
  }

  TrtBaseNet::TrtBaseNet(const std::string& engineFile)
    : mTrtContext(nullptr), mTrtEngine(nullptr),
      mTrtRunTime(nullptr), mTrtRunMode(RUN_MODE::FLOAT32),
      mTrtInputCount(0),mTrtIterationTime(0) {
    std::fstream file;
    file.open(engineFile, std::ios::binary | std::ios::in);
    if(!file.is_open()) {
      std::cout << "read engine file" << engineFile <<" failed" << std::endl;
      return;
    }
    file.seekg(0, std::ios::end);
    int length = file.tellg(); // input file stream length.
    file.seekg(0, std::ios::beg); 
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();
    std::cout << "deserializing" << std::endl;
    mTrtRunTime = createInferRuntime(gLogger);
    assert(mTrtRunTime != nullptr);
    mTrtEngine = mTrtRunTime->deserializeCudaEngine(data.get(), length, &mTrtPluginFactory);
    assert(mTrtEngine != nullptr);
    InitEngine();
  }

  TrtBaseNet::~TrtBaseNet() {
    cudaStreamSynchronize(mTrtCudaStream);
    cudaStreamDestroy(mTrtCudaStream);
    for (auto& item : mTrtCudaBuffer) {
      cudaFree(item);
    }
    mTrtPluginFactory.destroyPlugin();
    if(!mTrtRunTime) {
      mTrtRunTime->destroy();
    }
    if(!mTrtContext) {
      mTrtContext->destroy();
    }
    if(!mTrtEngine) {
      mTrtEngine->destroy();
    }
  }

  size_t TrtBaseNet::getBatchSize() {
    return mTrtBatchSize;
  }

  size_t TrtBaseNet::getInputSize() {
    return std::accumulate(mTrtBindBufferSize.begin(), mTrtBindBufferSize.begin() + mTrtInputCount,0);
  }

  size_t TrtBaseNet::getOutputSize() {
    return std::accumulate(mTrtBindBufferSize.begin() + mTrtInputCount, mTrtBindBufferSize.end(),0);
  }

  void TrtBaseNet::printTime() {
    mTrtProfiler.printLayerTimes(mTrtIterationTime);
  }

  void TrtBaseNet::InitEngine() {
    mTrtBatchSize = mTrtEngine->getMaxBatchSize();
    mTrtContext = mTrtEngine->createExecutionContext();
    assert(mTrtContext != nullptr);
    mTrtContext->setProfiler(&mTrtProfiler);
    // Input and output buffer pointers 
    // that we pass to the engine - the engine requires exactly IEngine::getNbBindings()
    int nbBindings = mTrtEngine->getNbBindings();
    mTrtCudaBuffer.resize(nbBindings);
    mTrtBindBufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i) {
      Dims dims = mTrtEngine->getBindingDimensions(i);
      DataType dtype = mTrtEngine->getBindingDataType(i);
      // 计算当前输入输出节点i的 data byte size
      int64_t totalSize = volume(dims) * mTrtBatchSize * getElementSize(dtype);
      mTrtBindBufferSize[i] = totalSize;
      mTrtCudaBuffer[i] = safeCudaMalloc(totalSize);
      if(mTrtEngine->bindingIsInput(i)) {
        mTrtInputCount++;
      }
    }
    CUDA_CHECK(cudaStreamCreate(&mTrtCudaStream));
  }

  nvinfer1::ICudaEngine* TrtBaseNet::loadModelAndCreateEngine(
      const char* deployFile, const char* modelFile, int maxBatchSize,
      ICaffeParser* parser, nvcaffeparser1::IPluginFactory* pluginFactory,
      IInt8Calibrator* calibrator, IHostMemory*& trtModelStream,
      const std::vector<std::string>& outputNodesName) {
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    // Parse the model to populate the network, then set the outputs.
    INetworkDefinition* network = builder->createNetwork();
    parser->setPluginFactory(pluginFactory);
    std::cout << "Begin parsing model..." << std::endl;
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile,modelFile, *network, nvinfer1::DataType::kFLOAT);
    if (!blobNameToTensor) {
      RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    }
    std::cout << "End parsing model..." << std::endl;
    // specify which tensors are outputs
    for (auto& name : outputNodesName) {
      auto output = blobNameToTensor->find(name.c_str());
      assert(output!=nullptr);
      if (output == nullptr) {
        std::cout << "can not find output named " << name << std::endl;
      }
      // 把指定tensor标记为网络的输出
      network->markOutput(*output);
    }
    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30); // 设置trt工作空间size: 1G
    if (mTrtRunMode == RUN_MODE::INT8) {
      std::cout <<"setInt8Mode"<<std::endl;
      if (!builder->platformHasFastInt8()) {
        std::cout << "Notice: the platform do not has fast for int8" << std::endl;
      }
      builder->setInt8Mode(true);
      builder->setInt8Calibrator(calibrator);
    } else if (mTrtRunMode == RUN_MODE::FLOAT16) {
      std::cout <<"setFp16Mode"<<std::endl;
      if (!builder->platformHasFastFp16()) {
        std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
      }
      builder->setFp16Mode(true);
    }
    std::cout << "Begin building engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine) {
      RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    }
    std::cout << "End building engine..." << std::endl;
    // We don't need the network any more, and we can destroy the parser.
    network->destroy();
    parser->destroy();
    // Serialize the engine, then close everything down.
    trtModelStream = engine->serialize();
    // std::string cache_path = "./yolov3_416.engine";
    // std::ofstream serialize_output_stream(cache_path, std::ios::out | std::ios::binary);
    // serialize_output_stream.write((char*)(trtModelStream->data()), trtModelStream->size());
    // serialize_output_stream.close();
    builder->destroy();
    shutdownProtobufLibrary();
    return engine;
  }

  void TrtBaseNet::doInference(const void* inputData, void* outputData, int batchSize /*= 1*/) {
    //static const int batchSize = 1;
    assert(mTrtInputCount == 1);
    assert(batchSize <= mTrtBatchSize);
    // DMA the input to the GPU, execute the batch asynchronously, and DMA it back:
    int inputIndex = 0;
    CUDA_CHECK(cudaMemcpyAsync(mTrtCudaBuffer[inputIndex], inputData, mTrtBindBufferSize[inputIndex], cudaMemcpyHostToDevice, mTrtCudaStream));
    // do inference and timing with chrono.
    auto t_start = std::chrono::high_resolution_clock::now();
    mTrtContext->execute(batchSize, &mTrtCudaBuffer[inputIndex]);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Time taken for inference is " << total << " ms." << std::endl;
    for (size_t bindingIdx = mTrtInputCount; bindingIdx < mTrtBindBufferSize.size(); ++bindingIdx) {
      auto size = mTrtBindBufferSize[bindingIdx];
      CUDA_CHECK(cudaMemcpyAsync(outputData, mTrtCudaBuffer[bindingIdx], size, cudaMemcpyDeviceToHost, mTrtCudaStream));
      outputData = (char*)outputData + size;
    }
    //cudaStreamSynchronize(mTrtCudaStream);
    mTrtIterationTime++ ;
  }

  void TrtBaseNet::saveEngine(std::string fileName) {
    if(mTrtEngine) {
      nvinfer1::IHostMemory* data = mTrtEngine->serialize();
      std::ofstream file;
      file.open(fileName,std::ios::binary | std::ios::out);
      if(!file.is_open()) {
          std::cout << "read create engine file" << fileName <<" failed" << std::endl;
          return;
      }
      file.write((const char*)data->data(), data->size());
      file.close();
    }
  }
}
