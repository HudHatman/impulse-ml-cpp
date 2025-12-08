#pragma once

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        class Computation : public AbstractComputation {
        private:
            ComputationCpu *computation;

        public:
            explicit Computation();

            Math::Matrix &getVariable(T_String);

            void initialize(T_String);

            void resize(T_String, T_Size, T_Size);

            void setZero(T_String);

            void randomInit(T_String, double);

            void setVariable(T_String, const Math::Matrix &);

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