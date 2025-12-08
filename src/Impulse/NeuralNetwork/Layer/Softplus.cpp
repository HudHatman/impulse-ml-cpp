#include "../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            Softplus::Softplus() : Abstract1D() {
            };

            Math::Matrix Softplus::activation() {
                this->computation->softplusActivation();
                return this->computation->getVariable("A");
            }

            Math::Matrix Softplus::derivative(Math::Matrix &a) {
                return this->computation->softplusDerivative(a);
            }

            LayerType Softplus::getType() {
                return LayerType::Softplus;
            }

            double Softplus::loss(Math::Matrix &output, Math::Matrix &predictions) {
                return 0.0; // TODO
            }

            double Softplus::error(T_Size m) {
                return 0.0; // TODO
            }
        }
    }
}