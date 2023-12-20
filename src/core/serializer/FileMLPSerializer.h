#ifndef _FILEMLPSERIALIZER_H_
#define _FILEMLPSERIALIZER_H_

#include <fstream>
#include <sstream>

#include "../MLPSerializer.h"

namespace s21 {
    class FileMLPSerializer : public MLPSerializer {
       public:
        void serialize(std::unique_ptr<MLPModel>& model, const std::string& filename) override;
        void deserialize(std::unique_ptr<MLPModel>& model, const std::string& filename) override;

       private:
        std::vector<double> fillImportVector(std::ifstream &s, const std::string& type, const size_t elements);
        std::pair<size_t, size_t> getImportSize(std::unique_ptr<MLPModel> &model, const std::string &line);
    };
}


#endif // _FILEMLPSERIALIZER_H_
