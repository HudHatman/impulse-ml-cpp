#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            template<typename OPTIMIZER_TYPE>
            AbstractTrainer<OPTIMIZER_TYPE>::AbstractTrainer(Network::Abstract &net) : network(net) {
                this->optimizer = new OPTIMIZER_TYPE;
            }

            template
            AbstractTrainer<Optimizer::Adam>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Adadelta>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Adagrad>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::GradientDescent>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Momentum>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Nesterov>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Rmsprop>::AbstractTrainer(Network::Abstract &net);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setRegularization(double value) {
                this->regularization = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setRegularization(double value);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setLearningIterations(T_Size value) {
                this->learningIterations = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setLearningIterations(T_Size value);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setLearningRate(double value) {
                this->learningRate = value;
                this->optimizer->setLearningRate(value);
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setLearningRate(double value);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setVerbose(bool value) {
                this->verbose = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setVerbose(bool value);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setVerboseStep(int value) {
                this->verboseStep = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setVerboseStep(int value);

            template<typename OPTIMIZER_TYPE>
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<OPTIMIZER_TYPE>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient) {
                Eigen::MatrixXd predictedOutput = this->network.forward(dataSet.getInput());
                Eigen::MatrixXd correctOutput = dataSet.getOutput();
                double epsilon = 1e-8;
                double size = correctOutput.cols();
                Eigen::MatrixXd logPredictions = (predictedOutput.array().cwiseMax(epsilon)).log();
                double cost = ((correctOutput.array() * logPredictions.array()).sum()) / size;

                Impulse::NeuralNetwork::Trainer::CostGradientResult result;
                result.cost = -cost;
                result.accuracy = 0;
                if (rollGradient) {
                    result.gradient = this->network.getRolledGradient();
                }

                return result;
            }

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Adam>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Adadelta>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Adagrad>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::GradientDescent>::cost(Impulse::Dataset::SlicedDataset &dataSet,
                                                              bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Momentum>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Nesterov>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Rmsprop>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setStepCallback(std::function<void()> callback) {
                this->stepCallback = callback;
                this->stepCallbackSet = true;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setStepCallback(std::function<void()>);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setStepCallback(std::function<void()>);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setStepCallback(std::function<void()>);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setStepCallback(std::function<void()>);

            template
            void AbstractTrainer<Optimizer::Momentum>::setStepCallback(std::function<void()>);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setStepCallback(std::function<void()>);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setStepCallback(std::function<void()>);
        }
    }
}