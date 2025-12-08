#include "../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            Logistic::Logistic() : Abstract1D() {
            };

            Math::Matrix Logistic::activation() {
                this->computation->logisticActivation();
                return this->computation->getVariable("A");
            }

            Math::Matrix Logistic::derivative(Math::Matrix &a) {
                return this->computation->logisticDerivative(a);
            }

            LayerType Logistic::getType() {
                return LayerType::Logistic;
            }

            double Logistic::loss(Math::Matrix &output, Math::Matrix &predictions) {
                return this->computation->logisticLoss(output, predictions);
            }

            double Logistic::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}