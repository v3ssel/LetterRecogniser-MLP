#ifndef _EMNISTDATASET_READER_H_
#define _EMNISTDATASET_READER_H_

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

#include "EmnistData.h"

namespace s21 {
class EMNISTDatasetReader {
   public:
    void open(const std::string& filename);
    EMNISTData readLine();
    size_t getNumberOfLines();
    void close();

    bool is_open() const;

   private:
    std::ifstream file_;
};
}  // namespace s21

#endif  // _EMNISTDATASET_READER_H_