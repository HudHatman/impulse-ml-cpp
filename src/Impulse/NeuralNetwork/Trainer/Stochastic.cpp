#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        namespace Trainer {
            template<class OPTIMIZER_TYPE, class COST_TYPE>
            Stochastic<OPTIMIZER_TYPE,
                COST_TYPE>::Stochastic(Network::Network &net) : AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>(net) {
            }

            template
            Stochastic<Optimizer::GradientDescent, Cost::CrossEntropy>::Stochastic(Network::Network & net);

            template
            Stochastic<Optimizer::Adam, Cost::CrossEntropy>::Stochastic(Network::Network & net);

            template
            Stochastic<Optimizer::Adagrad, Cost::CrossEntropy>::Stochastic(Network::Network & net);

            template
            Stochastic<Optimizer::Momentum, Cost::CrossEntropy>::Stochastic(Network::Network & net);

            template
            Stochastic<Optimizer::Nesterov, Cost::CrossEntropy>::Stochastic(Network::Network & net);

            template
            Stochastic<Optimizer::Rmsprop, Cost::CrossEntropy>::Stochastic(Network::Network & net);

            template
            Stochastic<Optimizer::Adadelta, Cost::CrossEntropy>::Stochastic(Network::Network & net);

            template<class OPTIMIZER_TYPE, class COST_TYPE>
            void Stochastic<OPTIMIZER_TYPE, COST_TYPE>::train(Impulse::Dataset::SlicedDataset &dataSet) {
                T_Size t = 0;

                for (T_Size i = 0; i < this->learningIterations; i++) {
                    high_resolution_clock::time_point begin = high_resolution_clock::now();

                    Eigen::MatrixXd input = dataSet.getInput();
                    Eigen::MatrixXd output = dataSet.getOutput();
                    Eigen::MatrixXd forward = this->network.forward(input);

                    this->network.backward(input, output, forward, this->regularization);

                    for (T_Size j = 0; j < this->network.getSize(); j++) {
                        Layer::LayerPointer layer = this->network.getLayer(j);
                        if (layer->getType() == Layer::LayerType::MaxPool) {
                            continue;
                        }
                        this->optimizer->setT(++t);
                        this->optimizer->optimize(layer.get());
                    }

                    if (this->verbose) {
                        if ((i + 1) % this->verboseStep == 0) {
                            high_resolution_clock::time_point end = high_resolution_clock::now();
                            auto duration = duration_cast<milliseconds>(end - begin).count();
                            std::cout << "Iteration: " << (i + 1)
                                    << " | Cost: " << this->cost->loss(dataSet.getInput(), dataSet.getOutput())
                                    << " | Accuracy: " << this->cost->accuracy(dataSet.getInput(), dataSet.getOutput())
                                    << "% | Time: " << duration
                                    << std::endl;
                        }
                    }

                    if (this->stepCallbackSet) {
                        this->stepCallback();
                    }
                }
            }
        }
    }
}