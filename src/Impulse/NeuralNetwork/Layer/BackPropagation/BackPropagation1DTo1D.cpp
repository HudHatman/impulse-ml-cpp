#include "../../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            namespace BackPropagation {
                BackPropagation1DTo1D::BackPropagation1DTo1D
                (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) : Abstract(layer, previousLayer) {
                }

                Math::Matrix BackPropagation1DTo1D::propagate(const Math::Matrix &input,
                                                                 T_Size numberOfExamples,
                                                                 double regularization,
                                                                 const Math::Matrix &sigma) {
                    Math::Matrix dZ = sigma;

                    if (this->layer->isLast() && this->layer->getType() != LayerType::Softmax) {
                        dZ = sigma * this->layer->derivative(this->layer->getComputation()->getVariable("Z"));
                    }
                    Math::Matrix previousActivations =
                            this->previousLayer == nullptr
                                ? input
                                : this->previousLayer->getComputation()->getVariable(
                                    "A");

                    Math::Matrix W = this->layer->getComputation()->getVariable("W");
                    Math::Matrix b = this->layer->getComputation()->getVariable("W");

                    Math::Matrix gW_temp = (1.0 / numberOfExamples) * (dZ * previousActivations.transpose());
                    gW_temp += (regularization / numberOfExamples) * W;
                    Math::Matrix gb_temp = dZ.rowwise().sum() / numberOfExamples;
                    Math::Matrix dA_prev_temp = W.transpose() * dZ;

                    this->layer->getComputation()->setVariable("gW", gW_temp);
                    this->layer->getComputation()->setVariable("gB", gb_temp);

                    if (this->previousLayer != nullptr) {
                        return dA_prev_temp;
                    }
                    return Math::Matrix(); // return empty - this is first layer
                }
            }
        }
    }
}