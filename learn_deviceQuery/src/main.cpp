#include "deviceQuery.hpp"

int main(int argc, char* argv[]) {
  // get device number
  int n = get_deviceNum();
  spdlog::info("GPU nums: {}", n);
  return 0;
}