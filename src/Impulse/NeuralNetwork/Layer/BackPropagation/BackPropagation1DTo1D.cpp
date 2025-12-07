#include "../../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            namespace BackPropagation {
                BackPropagation1DTo1D::BackPropagation1DTo1D
                (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) : Abstract(layer, previousLayer) {
                }

                Eigen::MatrixXd BackPropagation1DTo1D::propagate(const Eigen::MatrixXd &input,
                                                                 T_Size numberOfExamples,
                                                                 double regularization,
                                                                 const Eigen::MatrixXd &sigma) {
                    Eigen::MatrixXd dZ = sigma;

                    if (this->layer->isLast() && this->layer->getType() != LayerType::Softmax) {
                        dZ = sigma * this->layer->derivative(this->layer->getComputation()->getVariable("Z"));
                    }
                    Eigen::MatrixXd previousActivations =
                            this->previousLayer == nullptr
                                ? input
                                : this->previousLayer->getComputation()->getVariable(
                                    "A");

                    Eigen::MatrixXd W = this->layer->getComputation()->getVariable("W");
                    Eigen::MatrixXd b = this->layer->getComputation()->getVariable("W");

                    Eigen::MatrixXd gW_temp = (1.0 / numberOfExamples) * (dZ * previousActivations.transpose());
                    gW_temp += (regularization / numberOfExamples) * W;
                    Eigen::MatrixXd gb_temp = dZ.rowwise().sum() / numberOfExamples;
                    Eigen::MatrixXd dA_prev_temp = W.transpose() * dZ;

                    this->layer->getComputation()->setVariable("gW", gW_temp);
                    this->layer->getComputation()->setVariable("gB", gb_temp);

                    if (this->previousLayer != nullptr) {
                        return dA_prev_temp;
                    }
                    return Eigen::MatrixXd(); // return empty - this is first layer
                }
            }
        }
    }
}