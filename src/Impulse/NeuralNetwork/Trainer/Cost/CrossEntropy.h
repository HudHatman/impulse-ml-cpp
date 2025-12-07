#ifndef IMPULSE_ML_NEURAL_NETWORK_CROSSENTROPY_H
#define IMPULSE_ML_NEURAL_NETWORK_CROSSENTROPY_H

#include "../../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Trainer {
            namespace Cost {
                class CrossEntropy final : public Abstract {
                public:
                    explicit CrossEntropy(Network::Network &network) : Abstract(network) {
                    };

                    double loss(Eigen::MatrixXd output, Eigen::MatrixXd predictions) {
                        double miniBatchSize = output.cols();
                        Eigen::MatrixXd logPredictions = (predictions.array().cwiseMax(1e-8)).log();
                        double cost = (output.array() * logPredictions.array()).sum();

                        return -cost / miniBatchSize;
                    }

                    double accuracy(Eigen::MatrixXd output, Eigen::MatrixXd predictions) {
                        return 0.0;
                    }

                    Eigen::MatrixXd derivative(Eigen::MatrixXd output, Eigen::MatrixXd predictions,
                                               Eigen::MatrixXd activationDerivative) {
                        if (this->network.getLayer(this->network.getSize() - 1)->getType() ==
                            Layer::LayerType::Softmax) {
                            return predictions - output;
                        }

                        Eigen::MatrixXd term1 = output.array() / (predictions.array() + 1e-08);
                        Eigen::MatrixXd oneMinusY = 1.0 - output.array();
                        Eigen::MatrixXd oneMinusA = 1.0 - predictions.array();
                        Eigen::MatrixXd term2 = oneMinusY.array() / (oneMinusA.array() + 1e-08);

                        return term2 - term1;
                    }
                };
            }
        }
    }
}
#endif //IMPULSE_ML_NEURAL_NETWORK_CROSSENTROPY_H
