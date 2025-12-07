#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            MiniBatch<OPTIMIZER_TYPE, COST_TYPE>::MiniBatch(Network::Network &net) : AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>(net) {}

            template
            MiniBatch<Optimizer::GradientDescent, Cost::CrossEntropy>::MiniBatch(Network::Network &net);

            template
            MiniBatch<Optimizer::Adam, Cost::CrossEntropy>::MiniBatch(Network::Network &net);

            template
            MiniBatch<Optimizer::Adagrad, Cost::CrossEntropy>::MiniBatch(Network::Network &net);

            template
            MiniBatch<Optimizer::Momentum, Cost::CrossEntropy>::MiniBatch(Network::Network &net);

            template
            MiniBatch<Optimizer::Nesterov, Cost::CrossEntropy>::MiniBatch(Network::Network &net);

            template
            MiniBatch<Optimizer::Rmsprop, Cost::CrossEntropy>::MiniBatch(Network::Network &net);

            template
            MiniBatch<Optimizer::Adadelta, Cost::CrossEntropy>::MiniBatch(Network::Network &net);

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            void MiniBatch<OPTIMIZER_TYPE, COST_TYPE>::setBatchSize(T_Size value) {
                this->batchSize = value;
                this->optimizer->setBatchSize(value);
            }

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            void MiniBatch<OPTIMIZER_TYPE, COST_TYPE>::train(Impulse::Dataset::SlicedDataset &dataSet) {
                auto numberOfExamples = (T_Size) dataSet.getInput().cols();
                high_resolution_clock::time_point beginTrain = high_resolution_clock::now();

                T_Size t = 0;

                for (T_Size i = 0; i < this->learningIterations; i++) {
                    high_resolution_clock::time_point beginIteration = high_resolution_clock::now();

                    for (T_Size batch = 0, offset = 0; batch < numberOfExamples; batch += batchSize, offset++) {
                        high_resolution_clock::time_point beginIterationBatch = high_resolution_clock::now();

                        Eigen::MatrixXd input = dataSet.getInput(offset, batchSize);
                        Eigen::MatrixXd output = dataSet.getOutput(offset, batchSize);
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
                            high_resolution_clock::time_point endIterationBatch = high_resolution_clock::now();
                            auto durationBatch = duration_cast<milliseconds>(
                                    endIterationBatch - beginIterationBatch).count();
                            std::cout << "Batch: " << (offset + 1) << "/" << ceil((double) numberOfExamples / batchSize)
                                      << " | Time: " << durationBatch << "ms"
                                      << "\r";
                        }
                    }

                    if (this->verbose) {
                        if ((i + 1) % this->verboseStep == 0) {
                            high_resolution_clock::time_point endIteration = high_resolution_clock::now();
                            auto duration = duration_cast<milliseconds>(endIteration - beginIteration).count();
                            std::cout << "Iteration: " << (i + 1)
                                      << " | Cost: " << this->cost->loss(this->network.forward(dataSet.getInput()), dataSet.getOutput())
                                      << " | Accuracy: " << this->cost->accuracy(dataSet.getInput(), dataSet.getOutput())
                                      << "% | Time: " << duration << "ms"
                                      << std::endl;
                        }
                    }

                    if (this->stepCallbackSet) {
                        this->stepCallback();
                    }
                }

                if (this->verbose) {
                    high_resolution_clock::time_point endTrain = high_resolution_clock::now();
                    auto duration = duration_cast<seconds>(endTrain - beginTrain).count();
                    std::cout << "Training end. " << duration << "s" << std::endl;
                }
            }
        }
    }
}
