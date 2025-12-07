#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        namespace Builder {
            class Builder : public Abstract<Network::Network> {
            protected:
            public:
                explicit Builder(T_Dimension dims);

                void firstLayerTransition(Layer::LayerPointer layer) override;

                static Builder fromJSON(T_String path);
            };
        }
    }
}