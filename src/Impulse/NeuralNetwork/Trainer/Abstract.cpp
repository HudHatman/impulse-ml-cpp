#include "../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Trainer {
            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>::AbstractTrainer(Network::Network &net) : network(net) {
                this->optimizer = new OPTIMIZER_TYPE;
                this->cost = new COST_TYPE(net);
            }

            template
            AbstractTrainer<Optimizer::Adam, Cost::CrossEntropy>::AbstractTrainer(Network::Network & net);

            template
            AbstractTrainer<Optimizer::Adadelta, Cost::CrossEntropy>::AbstractTrainer(Network::Network & net);

            template
            AbstractTrainer<Optimizer::Adagrad, Cost::CrossEntropy>::AbstractTrainer(Network::Network & net);

            template
            AbstractTrainer<Optimizer::GradientDescent, Cost::CrossEntropy>::AbstractTrainer(Network::Network & net);

            template
            AbstractTrainer<Optimizer::Momentum, Cost::CrossEntropy>::AbstractTrainer(Network::Network & net);

            template
            AbstractTrainer<Optimizer::Nesterov, Cost::CrossEntropy>::AbstractTrainer(Network::Network & net);

            template
            AbstractTrainer<Optimizer::Rmsprop, Cost::CrossEntropy>::AbstractTrainer(Network::Network & net);

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>::setRegularization(double value) {
                this->regularization = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam, Cost::CrossEntropy>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Adadelta, Cost::CrossEntropy>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Adagrad, Cost::CrossEntropy>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::GradientDescent, Cost::CrossEntropy>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Momentum, Cost::CrossEntropy>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Nesterov, Cost::CrossEntropy>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Rmsprop, Cost::CrossEntropy>::setRegularization(double value);

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>::setLearningIterations(T_Size value) {
                this->learningIterations = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam, Cost::CrossEntropy>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Adadelta, Cost::CrossEntropy>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Adagrad, Cost::CrossEntropy>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::GradientDescent, Cost::CrossEntropy>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Momentum, Cost::CrossEntropy>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Nesterov, Cost::CrossEntropy>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Rmsprop, Cost::CrossEntropy>::setLearningIterations(T_Size value);

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>::setLearningRate(double value) {
                this->learningRate = value;
                this->optimizer->setLearningRate(value);
            }

            template
            void AbstractTrainer<Optimizer::Adam, Cost::CrossEntropy>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Adadelta, Cost::CrossEntropy>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Adagrad, Cost::CrossEntropy>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::GradientDescent, Cost::CrossEntropy>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Momentum, Cost::CrossEntropy>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Nesterov, Cost::CrossEntropy>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Rmsprop, Cost::CrossEntropy>::setLearningRate(double value);

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>::setVerbose(bool value) {
                this->verbose = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam, Cost::CrossEntropy>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Adadelta, Cost::CrossEntropy>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Adagrad, Cost::CrossEntropy>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::GradientDescent, Cost::CrossEntropy>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Momentum, Cost::CrossEntropy>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Nesterov, Cost::CrossEntropy>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Rmsprop, Cost::CrossEntropy>::setVerbose(bool value);

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>::setVerboseStep(int value) {
                this->verboseStep = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam, Cost::CrossEntropy>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Adadelta, Cost::CrossEntropy>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Adagrad, Cost::CrossEntropy>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::GradientDescent, Cost::CrossEntropy>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Momentum, Cost::CrossEntropy>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Nesterov, Cost::CrossEntropy>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Rmsprop, Cost::CrossEntropy>::setVerboseStep(int value);

            template<typename OPTIMIZER_TYPE, typename COST_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE, COST_TYPE>::setStepCallback(std::function<void()> callback) {
                this->stepCallback = callback;
                this->stepCallbackSet = true;
            }

            template
            void AbstractTrainer<Optimizer::Adam, Cost::CrossEntropy>::setStepCallback(std::function < void() >);

            template
            void AbstractTrainer<Optimizer::Nesterov, Cost::CrossEntropy>::setStepCallback(std::function < void() >);

            template
            void AbstractTrainer<Optimizer::Adadelta, Cost::CrossEntropy>::setStepCallback(std::function < void() >);

            template
            void AbstractTrainer<Optimizer::Rmsprop, Cost::CrossEntropy>::setStepCallback(std::function < void() >);

            template
            void AbstractTrainer<Optimizer::Momentum, Cost::CrossEntropy>::setStepCallback(std::function < void() >);

            template
            void AbstractTrainer<Optimizer::GradientDescent,
                Cost::CrossEntropy>::setStepCallback(std::function < void() >);

            template
            void AbstractTrainer<Optimizer::Adagrad, Cost::CrossEntropy>::setStepCallback(std::function < void() >);
        }
    }
}