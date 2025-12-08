#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            const T_String TYPE_SOFTMAX = "softmax";

            class Softmax : public Abstract1D {
            protected:
            public:
                Softmax();

                Math::Matrix activation() override;

                Math::Matrix derivative(Math::Matrix &a) override;

                LayerType getType() override;

                double loss(Math::Matrix &output, Math::Matrix &predictions) override;

                double error(T_Size m) override;
            };
        }
    }
}