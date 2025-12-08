#pragma once

#include "../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        struct SlicedDataset {
            Dataset input;
            Dataset output;

            T_Size getNumberOfExamples() {
                return this->input.getSize();
            }

            Matrix getInput(T_Size offset, T_Size batchSize) {
                Matrix input = this->input.exportToEigen();
                return input.block(offset, 0, batchSize, input.cols()).transpose();
            }

            Matrix getOutput(T_Size offset, T_Size batchSize) {
                Matrix output = this->output.exportToEigen();
                return output.block(offset, 0, batchSize, output.cols()).transpose();
            }

            Matrix getInput() {
                return this->input.exportToEigen().transpose();
            }

            Matrix getOutput() {
                return this->output.exportToEigen().transpose();
            }
        };

        namespace DatasetModifier {

            class DatasetSlicer {
            protected:
                Dataset &dataset;

                T_IntVector inputColumns;

                T_IntVector outputColumns;

            public:
                explicit DatasetSlicer(Dataset &dataset);

                void addInputColumn(int columnIndex);

                void addOutputColumn(int columnIndex);

                SlicedDataset slice();
            };
        }
    }
}
