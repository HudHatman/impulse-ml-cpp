#pragma once

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        class ComputationCpu : public AbstractComputation {
        private:
            std::map<T_String, Math::Matrix> variables;

        public:
            explicit ComputationCpu();

            Math::Matrix &getVariable(T_String);

            void setVariable(T_String, const Math::Matrix &);

            void initialize(T_String);

            void resize(T_String, T_Size, T_Size);

            void setZero(T_String);

            void randomInit(T_String, double);

            Math::Matrix forward(const Math::Matrix &);

            void reluActivation();

            Math::Matrix reluDerivative(Math::Matrix &);

            void logisticActivation();

            Math::Matrix logisticDerivative(Math::Matrix &);

            void softmaxActivation();

            Math::Matrix softmaxDerivative(Math::Matrix &);

            void softplusActivation();

            Math::Matrix softplusDerivative(Math::Matrix &);

            void tanhActivation();

            Math::Matrix tanhDerivative(Math::Matrix &);

            double logisticLoss(Math::Matrix &, Math::Matrix &);

            double purelinLoss(Math::Matrix &, Math::Matrix &);

            double softmaxLoss(Math::Matrix &, Math::Matrix &);

            double penalty();

            void gradientDescent(double);

            void gradientAdam(double, T_Size);

            void gradientRmsProp(double, T_Size);

            void gradientAdagrad(double, T_Size);

            void gradientNesterov(double, T_Size);

            void gradientMomentum(double, T_Size);

            void gradientAdadelta(double, T_Size);
        };
    }
}