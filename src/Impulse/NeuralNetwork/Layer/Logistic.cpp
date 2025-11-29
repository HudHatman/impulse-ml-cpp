#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Logistic::Logistic() : Abstract1D() {};

            Eigen::MatrixXd Logistic::activation() {
                this->computation->logisticActivation();
                return this->computation->getVariable("A");
            }

            Eigen::MatrixXd Logistic::derivative(Eigen::MatrixXd &a) {
                return this->computation->logisticDerivative(a);
            }

            LayerType Logistic::getType() {
                return LayerType::Logistic;
            }

            double Logistic::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                return this->computation->logisticLoss(output, predictions);
            }

            double Logistic::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
