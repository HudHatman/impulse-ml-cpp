#include "include.h"

namespace Impulse {
    namespace NeuralNetwork {
        Computation::Computation() : AbstractComputation() {
            this->computation = new ComputationCpu();
        }

        Math::Matrix &Computation::getVariable(T_String name) {
            return this->computation->getVariable(name);
        }

        void Computation::setVariable(T_String name, const Math::Matrix &variable) {
            this->computation->setVariable(name, variable);
        }

        void Computation::initialize(T_String name) {
            this->computation->initialize(name);
        }

        Math::Matrix
        Computation::forward(const Math::Matrix &input) {
            return this->computation->forward(input);
        }

        void Computation::resize(T_String name, T_Size width, T_Size height) {
            this->computation->resize(name, width, height);
        }

        void Computation::setZero(T_String name) {
            this->computation->setZero(name);
        }

        void Computation::randomInit(T_String name, double parameter) {
            this->computation->randomInit(name, parameter);
        }

        void Computation::reluActivation() {
            return this->computation->reluActivation();
        }

        Math::Matrix Computation::reluDerivative(Math::Matrix &m) {
            return this->computation->reluDerivative(m);
        }

        void Computation::logisticActivation() {
            return this->computation->logisticActivation();
        }

        Math::Matrix Computation::logisticDerivative(Math::Matrix &m) {
            return this->computation->logisticDerivative(m);
        }

        void Computation::softmaxActivation() {
            return this->computation->softmaxActivation();
        }

        Math::Matrix Computation::softmaxDerivative(Math::Matrix &m) {
            return this->computation->softmaxDerivative(m);
        }

        void Computation::softplusActivation() {
            return this->computation->softplusActivation();
        }

        Math::Matrix Computation::softplusDerivative(Math::Matrix &m) {
            return this->computation->softplusDerivative(m);
        }

        void Computation::tanhActivation() {
            return this->computation->tanhActivation();
        }

        Math::Matrix Computation::tanhDerivative(Math::Matrix &m) {
            return this->computation->tanhDerivative(m);
        }

        double Computation::logisticLoss(Math::Matrix &output, Math::Matrix &predictions) {
            return this->computation->logisticLoss(output, predictions);
        }

        double Computation::purelinLoss(Math::Matrix &output, Math::Matrix &predictions) {
            return this->computation->purelinLoss(output, predictions);
        }

        double Computation::softmaxLoss(Math::Matrix &output, Math::Matrix &predictions) {
            return this->computation->softmaxLoss(output, predictions);
        }

        double Computation::penalty() {
            return this->computation->penalty();
        }

        void Computation::gradientDescent(double learningRate) {
            this->computation->gradientDescent(learningRate);
        }

        void Computation::gradientAdam(double learningRate, T_Size t) {
            this->computation->gradientAdam(learningRate, t);
        }

        void Computation::gradientRmsProp(double learningRate, T_Size batchSize) {
            this->computation->gradientRmsProp(learningRate, batchSize);
        }

        void Computation::gradientAdagrad(double learningRate, T_Size batchSize) {
            this->computation->gradientAdagrad(learningRate, batchSize);
        }

        void Computation::gradientNesterov(double learningRate, T_Size batchSize) {
            this->computation->gradientNesterov(learningRate, batchSize);
        }

        void Computation::gradientMomentum(double learningRate, T_Size batchSize) {
            this->computation->gradientMomentum(learningRate, batchSize);
        }

        void Computation::gradientAdadelta(double learningRate, T_Size batchSize) {
            this->computation->gradientAdadelta(learningRate, batchSize);
        }
    }
}