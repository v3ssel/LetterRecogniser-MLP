#ifndef _GRAPHLAYER_H_
#define _GRAPHLAYER_H_

#include <memory>
#include <random>

#include "GraphNode.h"

namespace s21 {
class GraphLayer {
   public:
    GraphLayer(size_t size);

    void randomize();

    std::vector<double> getWeights();
    std::vector<double> getBiases();

    void setWeights(std::vector<double>::const_iterator& begin);
    void setBiases(std::vector<double>::const_iterator& begin);

    size_t getSize();

    std::shared_ptr<GraphLayer>& getInputLayer();
    std::shared_ptr<GraphLayer>& getOutputLayer();

    void setInputLayer(std::shared_ptr<GraphLayer>& input);
    void setOutputLayer(std::shared_ptr<GraphLayer>& output);

    std::shared_ptr<GraphLayer> _input_layer;
    std::shared_ptr<GraphLayer> _output_layer;

    std::vector<GraphNode> nodes_;
    size_t size_;
};
}  // namespace s21

#endif  // _GRAPHLAYER_H_
