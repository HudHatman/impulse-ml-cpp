#pragma once

#include "../../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            namespace BackPropagation {
                class BackPropagationToConv : public Abstract {
                public:
                    BackPropagationToConv(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);

                    Math::Matrix propagate(const Math::Matrix &input,
                                              T_Size numberOfExamples,
                                              double regularization,
                                              const Math::Matrix &sigma) override;
                };
            }
        }
    }
}