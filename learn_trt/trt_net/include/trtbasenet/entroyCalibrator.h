#ifndef ENTROY_CALIBRATOR_H_
#define ENTROY_CALIBRATOR_H_

#include <cudnn.h>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "utils.h"

namespace nvinfer1 {

/**
 * @brief The trt int8 calibrator
 */
class Int8EntropyCalibrator2 : public IInt8EntropyCalibrator2 {
 public:
  /**
   * @brief Create trt int8 calibrator
   * 
   * @param BatchSize model input batchsize
   * @param data calibration data
   * @param CalibDataName calibration dataname
   * @param readCache read calibration data from cache or not
   */
	Int8EntropyCalibrator2(int BatchSize, const std::vector<std::vector<float>>& data,
                         const std::string& CalibDataName = "",
                         bool readCache = true);

  /**
   * @brief Int8 calibrator destructor
   */
	virtual ~Int8EntropyCalibrator2();

  /**
   * @brief Get netework mbatchszie
   */
	int getBatchSize() const override { 
    return mBatchSize;
  }

  /**
   * @brief to be added
   */
	bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

  /**
   * @brief Read calibrationdata into cache.
   * 
   * @param length calibrationcache data size
   */
  const void* readCalibrationCache(size_t& length) override;

  /**
   * @brief write calibrationdata into file from cache.
   * 
   * @param cache calibrationdata in cache
   * @param length cache length
   */
  void writeCalibrationCache(const void* cache, size_t length) override;

 private:
  std::string mCalibDataName; /// calibration data name
  std::vector<std::vector<float>> mDatas; /// calibration data
  int mBatchSize; /// input batch size
  int mCurBatchIdx; /// current batch idx
  float* mCurBatchData{ nullptr }; /// current batch data ptr
  size_t mInputCount; /// input data count
  bool mReadCache; /// readcache flag
  void* mDeviceInput{ nullptr }; /// input dev ptr
  std::vector<char> mCalibrationCache; /// calibration data cache
};

} //namespace

#endif // ENTROY_CALIBRATOR_H_
