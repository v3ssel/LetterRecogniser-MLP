#ifndef _MLPSERIALIZER_H_
#define _MLPSERIALIZER_H_

#include <memory>
#include <string>

#include "MLPModel.h"

namespace s21 {
class MLPSerializer {
   public:
    virtual ~MLPSerializer() = default;

    virtual void serialize(std::unique_ptr<MLPModel>& model,
                           const std::string& filename) = 0;
    virtual void deserialize(std::unique_ptr<MLPModel>& model,
                             const std::string& filename) = 0;
};
}  // namespace s21

#endif  // _MLPSERIALIZER_H_
