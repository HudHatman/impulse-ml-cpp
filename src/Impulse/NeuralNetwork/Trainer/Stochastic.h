#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        namespace Trainer {
            template<class OPTIMIZER_TYPE, class COST_TYPE>
            class Stochastic : public AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE> {
            public:
                explicit Stochastic(Network::Network &net);

                void train(Impulse::Dataset::SlicedDataset &dataSet) override;
            };
        }
    }
}