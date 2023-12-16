#ifndef _EMNISTDATASET_READER_H_
#define _EMNISTDATASET_READER_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "EMNISTData.h"

namespace s21 {
    class EMNISTDatasetReader {
       public:
        void open(const std::string& filename);
        EMNISTData readLine();
        void close();

        bool is_open() const;

       private:
        std::ifstream _file;
    };
}

#endif // _EMNISTDATASET_READER_H_