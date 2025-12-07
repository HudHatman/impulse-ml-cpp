#ifndef IMPULSE_ML_NEURAL_NETWORK_TYPES_H
#define IMPULSE_ML_NEURAL_NETWORK_TYPES_H

#include "../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            enum class LayerType {
                Conv,
                FullyConnected,
                Logistic,
                Purelin,
                Relu,
                Softmax,
                Softplus,
                Tanh,
                MaxPool,
                None,
            };
        }
    }
}


#endif //IMPULSE_ML_NEURAL_NETWORK_TYPES_H