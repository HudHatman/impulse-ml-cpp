#pragma once

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            namespace BackPropagation {
                class BackPropagation1DTo1D : public Abstract {
                public:
                    BackPropagation1DTo1D(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);

                    Math::Matrix propagate(const Math::Matrix &input,
                                              T_Size numberOfExamples,
                                              double regularization,
                                              const Math::Matrix &sigma);
                };
            }
        }
    }
}