#include "../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            Relu::Relu() : Abstract1D() {
            };

            Math::Matrix Relu::activation() {
                this->computation->reluActivation();
                return this->computation->getVariable("A");
            }

            Math::Matrix Relu::derivative(Math::Matrix &a) {
                return this->computation->reluDerivative(a);
            }

            LayerType Relu::getType() {
                return LayerType::Relu;
            }

            double Relu::loss(Math::Matrix &output, Math::Matrix &predictions) {
                // TODO
                return 0.0;
            }

            double Relu::error(T_Size m) {
                // TODO
                return 0.0;
            }
        }
    }
}