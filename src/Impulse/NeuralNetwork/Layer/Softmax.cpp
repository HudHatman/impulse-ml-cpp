#include "../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            Softmax::Softmax() : Abstract1D() {
            }

            Math::Matrix Softmax::activation() {
                this->computation->softmaxActivation();
                return this->computation->getVariable("A");
            }

            Math::Matrix Softmax::derivative(Math::Matrix &a) {
                return this->computation->softmaxDerivative(a);
            }

            LayerType Softmax::getType() {
                return LayerType::Softmax;
            }

            double Softmax::loss(Math::Matrix &output, Math::Matrix &predictions) {
                return this->computation->softmaxLoss(output, predictions);
            }

            double Softmax::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}