#pragma once

#include "Abstract.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Trainer {
            namespace Cost {
                class CrossEntropy final : public Abstract {
                public:
                    explicit CrossEntropy(Network::Network &network) : Abstract(network) {};

                    double loss(Math::Matrix output, Math::Matrix predictions) {
                        double miniBatchSize = output.cols();
                        Math::Matrix logPredictions = (predictions.array().cwiseMax(1e-8)).log();
                        double cost = (output.array() * logPredictions.array()).sum();

                        return -cost / miniBatchSize;
                    }
                    double accuracy(Math::Matrix output, Math::Matrix predictions) {
                        double acc = 0.0;
                        for (int i = 0; i < output.cols(); ++i) {
                            T_Size max_j = 0;
                            output.col(i).maxCoeff(&max_j);

                            T_Size max_k = 0;
                            predictions.col(i).maxCoeff(&max_k);

                            if (max_j == max_k) {
                                acc += 1.0;
                            }
                        }
                        return acc / (double) output.cols();
                    }

                    Math::Matrix derivative(Math::Matrix output, Math::Matrix predictions,
                                               Math::Matrix activationDerivative) {
                        if (this->network.getLayer(this->network.getSize() - 1)->getType() == Layer::LayerType::Softmax) {
                            return predictions - output;
                        }

                        Math::Matrix term1 = output.array() / (predictions.array() + 1e-08);
                        Math::Matrix oneMinusY = 1.0 - output.array();
                        Math::Matrix oneMinusA = 1.0 - predictions.array();
                        Math::Matrix term2 = oneMinusY.array() / (oneMinusA.array() + 1e-08);

                        return term2 - term1;
                    }
                };
            }
        }
    }
}