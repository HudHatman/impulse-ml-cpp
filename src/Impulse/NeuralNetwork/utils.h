#pragma once

#include "include.h"

namespace Impulse {
    namespace NeuralNetwork {
        namespace Utils {
            Math::Matrix im2col(const Math::Matrix &input, int channels,
                                   int height, int width,
                                   int kernel_h, int kernel_w,
                                   int pad_h, int pad_w,
                                   int stride_h, int stride_w);

            Math::Matrix maxpool(const Math::Matrix &input, int channels,
                                    int height, int width,
                                    int kernel_h, int kernel_w,
                                    int stride_h, int stride_w);
        }
    }
}