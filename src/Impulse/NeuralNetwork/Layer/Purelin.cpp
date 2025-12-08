#include "../include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Layer {
            Purelin::Purelin() : Abstract1D() {
            };

            Math::Matrix Purelin::activation() {
                return this->computation->getVariable("Z");
            }

            Math::Matrix Purelin::derivative(Math::Matrix &) {
                Math::Matrix d(this->computation->getVariable("Z").rows(),
                                  this->computation->getVariable("Z").cols());
                d.setOnes();
                return d;
            }

            LayerType Purelin::getType() {
                return LayerType::Purelin;
            }

            double Purelin::loss(Math::Matrix &output, Math::Matrix &predictions) {
                return this->computation->purelinLoss(output, predictions);
            }

            double Purelin::error(T_Size m) {
                return (1.0 / (2.0 * (double) m));
            }
        }
    }
}