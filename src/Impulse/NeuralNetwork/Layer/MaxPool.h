#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            const T_String TYPE_MAXPOOL = "maxpool";

            class MaxPool : public Abstract3D {
            protected:
                T_Size filterSize = 2;
                T_Size stride = 2;

            public:
                MaxPool();

                void configure() override;

                void setFilterSize(T_Size value);

                T_Size getFilterSize();

                void setStride(T_Size value);

                T_Size getStride();

                Math::Matrix forward(const Math::Matrix &input) override;

                Math::Matrix activation() override;

                Math::Matrix derivative(Math::Matrix &a) override;

                LayerType getType() override;

                double loss(Math::Matrix &output, Math::Matrix &predictions) override;

                double error(T_Size m) override;

                T_Size getOutputHeight() override;

                T_Size getOutputWidth() override;

                T_Size getOutputDepth() override;
            };
        }
    }
}