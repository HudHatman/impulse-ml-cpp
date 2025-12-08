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
                    virtual double loss(Math::Matrix output, Math::Matrix predictions) = 0;
                    virtual double accuracy(Math::Matrix output, Math::Matrix predictions) = 0;
                    virtual Math::Matrix derivative(Math::Matrix output, Math::Matrix predictions,
                                                       Math::Matrix activationDerivative) = 0;
                };
            }
        }
    }
}