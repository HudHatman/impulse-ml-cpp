#include "include.h"

namespace Impulse {
    namespace NeuralNetwork {
        ComputationCpu::ComputationCpu() : AbstractComputation() {
        }

        void ComputationCpu::initialize(T_String name) {
            this->variables[name] = Math::Matrix();
        }

        void ComputationCpu::resize(T_String name, T_Size width, T_Size height) {
            this->variables[name].resize(width, height);
        }

        void ComputationCpu::setVariable(T_String name, const Math::Matrix &variable) {
            this->variables[name] = variable;
        }

        Math::Matrix &ComputationCpu::getVariable(T_String name) {
            return this->variables[name];
        }

        void ComputationCpu::setZero(T_String name) {
            this->variables[name].setZero();
        }

        void ComputationCpu::randomInit(T_String name, double parameter) {
            this->variables[name].setRandom();
            this->variables[name] *= sqrt(2.0 / parameter);
        }

        Math::Matrix
        ComputationCpu::forward(const Math::Matrix &input) {
            this->setVariable("Z", (this->variables["W"] * input) + this->variables["b"].replicate(1, input.cols()));
            return this->getVariable("Z");
        }

        void ComputationCpu::reluActivation() {
            this->variables["A"] = this->variables["Z"].unaryExpr([](const double x) {
                return std::max(0.0, x);
            });
        }

        Math::Matrix ComputationCpu::reluDerivative(Math::Matrix &m) {
            return m.unaryExpr([](const double x) {
                if (x > 0.0) {
                    return 1.0;
                }
                return 0.0;
            });
        }

        void ComputationCpu::logisticActivation() {
            this->variables["A"] = this->variables["Z"].unaryExpr([](const double x) {
                return 1.0 / (1.0 + std::exp(-x));
            });
        }

        Math::Matrix ComputationCpu::logisticDerivative(Math::Matrix &m) {
            return (m.array() * (1.0 - m.array())).matrix();
        }

        void ComputationCpu::softmaxActivation() {
            Math::Matrix t = this->variables["Z"].unaryExpr([](const double x) {
                return exp(x);
            });
            Math::Matrix divider = t.colwise().sum().replicate(t.rows(), 1);
            Math::Matrix result = t.array() / divider.array();

            this->variables["A"] = result;
        }

        Math::Matrix ComputationCpu::softmaxDerivative(Math::Matrix &) {
            return Math::Matrix(); // TODO
        }

        void ComputationCpu::softplusActivation() {
            this->variables["A"] = this->variables["Z"].unaryExpr([](const double x) {
                return std::log(1.0 + std::exp(x));
            });
        }

        Math::Matrix ComputationCpu::softplusDerivative(Math::Matrix &m) {
            Math::Matrix result = m.unaryExpr([](const double x) {
                return (1.0 / (1.0 + std::exp(-x)));
            });
            return result;
        }

        void ComputationCpu::tanhActivation() {
            this->variables["A"] = this->variables["Z"].array().tanh();
        }

        Math::Matrix ComputationCpu::tanhDerivative(Math::Matrix &m) {
            Math::Matrix result = m.unaryExpr([](const double x) {
                return 1.0 - std::tanh(x) * std::tanh(x);
            });
            return result;
        }

        double ComputationCpu::logisticLoss(Math::Matrix &output, Math::Matrix &predictions) {
            Math::Matrix loss =
                    (output.array() * predictions.unaryExpr([](const double x) { return log(x); }).array())
                    +
                    (output.unaryExpr([](const double x) { return 1.0 - x; }).array()
                     *
                     predictions.unaryExpr([](const double x) { return log(1.0 - x); }).array()
                    );
            return loss.sum();
        }

        double ComputationCpu::purelinLoss(Math::Matrix &output, Math::Matrix &predictions) {
            Math::Matrix loss = (predictions.array() - output.array()).unaryExpr([](const double x) {
                return pow(x, 2.0);
            });
            return loss.sum();
        }

        double ComputationCpu::softmaxLoss(Math::Matrix &output, Math::Matrix &predictions) {
            Math::Matrix loss = (output.array() *
                                    predictions.unaryExpr([](const double x) { return log(x); }).array());
            return loss.sum();
        }

        void ComputationCpu::gradientDescent(double learningRate) {
            this->variables["W"] -= learningRate * this->variables["gW"];
            this->variables["b"] -= learningRate * this->variables["gB"];
        }

        double ComputationCpu::penalty() {
            return this->variables["W"].unaryExpr([](const double x) {
                return pow(x, 2.0);
            }).sum();
        }

        void
        ComputationCpu::gradientAdam(double learningRate, T_Size t) {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;

            Math::Matrix vW = (beta1 * this->variables["vW"] + ((1 - beta1) * this->variables["gW"]));
            this->variables["vW"] = vW;
            Math::Matrix wCorrected = this->variables["vW"] / (1 - std::pow(beta1, t));

            Math::Matrix cW = (beta2 * this->variables["cW"].array() +
                                  (1 - beta2) * (this->variables["gW"].array().square()));
            this->variables["cW"] = cW;

            Math::Matrix sCorrected = this->variables["cW"] / (1 - std::pow(beta2, t));
            sCorrected = sCorrected.array().sqrt();

            this->variables["W"].array() -= learningRate * (wCorrected.array() / sCorrected.array());

            //
            Math::Matrix vB = (beta1 * this->variables["vB"] + (1 - beta1) * this->variables["gB"]);
            this->variables["vB"] = vB;
            Eigen::VectorXd wCorrected2 = this->variables["vB"] / (1 - std::pow(beta1, t));

            Math::Matrix cB = (beta2 * this->variables["cB"].array() +
                                  (1 - beta2) * (this->variables["gB"].array().square()));
            this->variables["cB"] = cB;

            Eigen::VectorXd sCorrected2 = this->variables["cB"] / (1 - std::pow(beta2, t));
            sCorrected2 = sCorrected2.array().sqrt();

            this->variables["b"].array() -= learningRate * (wCorrected2.array() / sCorrected2.array());
        }

        void ComputationCpu::gradientRmsProp(double learningRate, T_Size batchSize) {
            double alpha = 1e-3;
            double gamma = 0.9;
            double epsilon = 1e-8;

            this->variables["cW"] = (gamma * this->variables["cW"].array() +
                                     (1.0 - gamma) * (this->variables["gW"].array().square()));
            this->variables["W"].array() -= (this->variables["gW"].array() * alpha /
                                             this->variables["cW"].unaryExpr(
                                                 [epsilon](double x) {
                                                     return std::sqrt(x + epsilon);
                                                 }).array());

            this->variables["cB"] = (gamma * this->variables["cB"].array() +
                                     (1.0 - gamma) * (this->variables["gB"].array().square()));
            this->variables["b"].array() -= (this->variables["gB"].array() * alpha /
                                             this->variables["cB"].unaryExpr(
                                                 [epsilon](double x) {
                                                     return std::sqrt(x + epsilon);
                                                 }).array());
        }

        void ComputationCpu::gradientAdagrad(double learningRate, T_Size batchSize) {
            double epsilon = 1e-8;

            this->variables["cW"].array() += this->variables["gW"].array().square();
            this->variables["W"].array() -= (learningRate * this->variables["gW"].array() /
                                             this->variables["cW"].unaryExpr(
                                                 [epsilon](double x) {
                                                     return std::sqrt(x + epsilon);
                                                 }).array());

            this->variables["cB"].array() += this->variables["gB"].array().square();
            this->variables["b"].array() -= (learningRate * this->variables["gB"].array() /
                                             this->variables["cB"].unaryExpr(
                                                 [epsilon](double x) {
                                                     return std::sqrt(x + epsilon);
                                                 }).array());
        }

        void ComputationCpu::gradientNesterov(double learningRate, T_Size batchSize) {
            double gamma = 0.9;

            Math::Matrix s_prev_w = this->variables["cW"];
            this->variables["cW"] = (gamma * this->variables["cW"].array()) - (
                                        learningRate * this->variables["gW"].array());
            this->variables["W"].array() += this->variables["cW"].array() + (
                gamma * (this->variables["cW"].array() - s_prev_w.array()));

            Eigen::VectorXd s_prev_b = this->variables["cB"];
            this->variables["cB"] = (gamma * this->variables["cB"].array()) - (
                                        learningRate * this->variables["gB"].array());
            this->variables["b"].array() += this->variables["cB"].array() + (
                gamma * (this->variables["cB"].array() - s_prev_b.array()));
        }

        void ComputationCpu::gradientMomentum(double learningRate, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            this->variables["cW"] = ((gamma * this->variables["cW"].array()) + (alpha * this->variables["gW"].array()));
            this->variables["W"] -= this->variables["cW"];

            this->variables["cB"] = ((gamma * this->variables["cB"].array()) + (alpha * this->variables["gB"].array()));
            this->variables["b"] -= this->variables["cB"];
        }

        void ComputationCpu::gradientAdadelta(double learningRate, T_Size batchSize) {
            //double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;
            double epsilon = 1e-6;

            this->variables["cW"] = ((gamma * this->variables["cW"].array()) +
                                     (1.0 - gamma) * (this->variables["gW"].array().square()));
            Math::Matrix deltaParameters = -(this->variables["vW"].unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array() / this->variables["cW"].unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array()).array() * this->variables["gW"].array();
            this->variables["vW"] =
                    ((gamma * this->variables["cW"].array()) + ((1.0 - gamma) * (deltaParameters.array().square()))).
                    eval();
            this->variables["W"] += deltaParameters;

            this->variables["cB"] = ((gamma * this->variables["cB"].array()) +
                                     (1.0 - gamma) * (this->variables["gB"].array().square()));
            Math::Matrix deltaParameters2 = -(this->variables["vB"].unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array() / this->variables["cB"].unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array()) * this->variables["gB"].array();
            this->variables["vB"] = ((gamma * this->variables["cB"].array()) +
                                     ((1.0 - gamma) * (deltaParameters2.array().square()))).eval();
            this->variables["b"] += deltaParameters2;
        }
    }
}
