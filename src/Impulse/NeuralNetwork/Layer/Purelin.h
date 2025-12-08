#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            const T_String TYPE_PURELIN = "purelin";

            class Purelin : public Abstract1D {
            protected:
            public:
                Purelin();

                Math::Matrix activation() override;

                Math::Matrix derivative(Math::Matrix &a) override;

                LayerType getType() override;

                double loss(Math::Matrix &output, Math::Matrix &predictions) override;

                double error(T_Size m) override;
            };
        }
    }
}