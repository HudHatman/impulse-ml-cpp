#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        namespace Builder {
            template<class NETWORK_TYPE>
            Abstract<NETWORK_TYPE>::Abstract(T_Dimension dims) : network(NETWORK_TYPE(dims)) {
                this->dimension = dims;
            }

            template<class NETWORK_TYPE>
            template<typename LAYER_TYPE>
            void Abstract<NETWORK_TYPE>::createLayer(std::function<void(LAYER_TYPE *)> callback) {
                auto *layer = new LAYER_TYPE();
                Layer::LayerPointer pointer(layer);

                callback(layer);

                if (this->previousLayer == nullptr) {
                    this->firstLayerTransition(pointer);
                } else {
                    pointer->transition(this->previousLayer);
                }

                pointer->backpropagation = Layer::BackPropagation::Factory::create(this->previousLayer, pointer);
                pointer->configure();

                this->network.addLayer(pointer);
                this->previousLayer = pointer;
            };

            template
            class Abstract<Network::Network>;

            template void
            Abstract<Network::Network>::createLayer<Layer::Logistic>(
                std::function < void(Layer::Logistic *) > callback);

            template void
            Abstract<Network::Network>::createLayer<Layer::Purelin>(
                std::function < void(Layer::Purelin *) > callback);

            template void
            Abstract<Network::Network>::createLayer<Layer::Relu>(
                std::function < void(Layer::Relu *) > callback);

            template void
            Abstract<Network::Network>::createLayer<Layer::Tanh>(
                std::function < void(Layer::Tanh *) > callback);

            template void
            Abstract<Network::Network>::createLayer<Layer::Softmax>(
                std::function < void(Layer::Softmax *) > callback);

            template void
            Abstract<Network::Network>::createLayer<Layer::Softplus>(
                std::function < void(Layer::Softplus *) > callback);

            template<class NETWORK_TYPE>
            NETWORK_TYPE &Abstract<NETWORK_TYPE>::getNetwork() {
                return this->network;
            }

            template<class NETWORK_TYPE>
            NETWORK_TYPE &Abstract<NETWORK_TYPE>::build() {
                this->network.layers.at(this->network.layers.size() - 1)->setIsLast(true);
                return this->network;
            }
        }
    }
}