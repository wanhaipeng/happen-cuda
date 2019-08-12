#include <fstream>
#include "cuda_common.hpp"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h> 

void ReadProtoFromTextFile(const char* filename, google::protobuf::Message& proto) {
  // std::fstream input(filename, std::ios::in | std::ios::binary);
  int fd = open(filename, O_RDONLY);
  if (fd < 0) {
    spdlog::error("file Descriptor not valid!");
    exit(-1);
  }
  google::protobuf::io::FileInputStream* input =
      new google::protobuf::io::FileInputStream(fd);
  input->SetCloseOnDelete(true);
  if (!google::protobuf::TextFormat::Parse(input, &proto)) {
    spdlog::error("parse protobuf text failed!");
    exit(-1);
  }
  delete input;
  close(fd);
}