#include "../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            Tanh::Tanh() : Abstract1D() {
            };

            Math::Matrix Tanh::activation() {
                this->computation->tanhActivation();
                return this->computation->getVariable("A");
            }

            Math::Matrix Tanh::derivative(Math::Matrix &a) {
                return this->computation->tanhDerivative(a);
            }

            LayerType Tanh::getType() {
                return LayerType::Tanh;
            }

            double Tanh::loss(Math::Matrix &output, Math::Matrix &predictions) {
                return 0.0; // TODO
            }

            double Tanh::error(T_Size m) {
                return 0.0; // TODO
            }
        }
    }
}