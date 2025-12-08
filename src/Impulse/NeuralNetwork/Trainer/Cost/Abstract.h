#pragma once

#include "../../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Trainer {
            namespace Cost {
                class Abstract {
                protected:
                    Network::Network &network;
                public:
                    explicit Abstract(Network::Network &network) : network(network) {} ;
                    virtual ~Abstract() = default;
                    virtual double loss(Eigen::MatrixXd output, Eigen::MatrixXd predictions) = 0;
                    virtual double accuracy(Eigen::MatrixXd output, Eigen::MatrixXd predictions) = 0;
                    virtual Eigen::MatrixXd derivative(Eigen::MatrixXd output, Eigen::MatrixXd predictions,
                                                       Eigen::MatrixXd activationDerivative) = 0;
                };
            }
        }
    }
}