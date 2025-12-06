#pragma once

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        class Serializer {
        protected:
            Impulse::NeuralNetwork::Network::Network network;
        public:
            explicit Serializer(Impulse::NeuralNetwork::Network::Network &net);

            void toJSON(T_String path);
        };
    }
}
