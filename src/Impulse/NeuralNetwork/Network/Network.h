#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {
    namespace NeuralNetwork {
        namespace Network {
            typedef std::vector<Layer::LayerPointer> LayersContainer;

            class Network {
            protected:
                T_Size size = 0;
                T_Dimension dimension;

            public:
                LayersContainer layers;

                explicit Network(T_Dimension dim);

                void addLayer(Layer::LayerPointer layer);

                Math::Matrix forward(const Math::Matrix &input);

                void
                backward(Math::Matrix &X, Math::Matrix &Y, Math::Matrix &predictions, double regularization);

                T_Dimension getDimension();

                T_Size getSize();

                Layer::LayerPointer getLayer(T_Size key);

                Eigen::VectorXd getRolledTheta();

                Eigen::VectorXd getRolledGradient();

                void setRolledTheta(Eigen::VectorXd &theta);

                double loss(Math::Matrix &output, Math::Matrix &predictions);

                double error(T_Size m);

                void debug();
            };
        }
    }
}