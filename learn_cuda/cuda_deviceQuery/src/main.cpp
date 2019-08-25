#include "deviceQuery.hpp"

int main(int argc, char* argv[]) {
  spdlog::set_pattern("[%^%l%$] %v");
  // get device number
  int n = get_deviceNum();
  spdlog::info("GPU nums: {}", n);

  // get device info
  for (int i = 0; i < n; ++i) {
    get_deviceInfo(i);
  }
  return 0;
}