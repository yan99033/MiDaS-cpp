/*
This file is part of midas-cpp

MIT License

Copyright (c) 2020 Shing Yan Loo (lsyan@ualberta.ca)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef MIDAS_H_
#define MIDAS_H_

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/numpy/dtype.hpp>

#include <torch/script.h>

#include <opencv2/opencv.hpp>

typedef unsigned char uchar;
using namespace boost::python;
namespace p = boost::python;
namespace np = boost::python::numpy;  

namespace midas
{

/// Network input size (predefined)
const int input_width_ = 384;
const int input_height_ = 288;

class MiDas
{
public:

    /// Constructor
    MiDas(const int img_width,
         const int img_height,
         const char* model);

    ~MiDas();

    /// Infer depth from image (implementation)
    np::ndarray inference(const np::ndarray &image);
    cv::Mat inference(const cv::Mat &image);

private:
    /// Convert image to input tensor, estimate depthmap, and convert output tensor to depthmap
    void preprocessImage(uchar* image); //const np::ndarray &image);


    cv::Size original_size_;                       /// Keep the original shape of the image (resize the image if necessary)
    int image_height_;                             /// Input image height
    int image_width_;                              /// Input image width
    torch::jit::script::Module module_;            /// Torch model
    cv::Mat input_cv_;                             /// Network input (cv::Mat)
    torch::Tensor input_tensor_;                   /// Network input (torch::Tensor)
    torch::Tensor output_tensor_;                  /// Network output (torch::Tensor)
    np::ndarray output_np_;                        /// Network output (np::ndarray)
    cv::Mat output_cv_;                            /// Network output (cv::Mat)
};


} // namespace midas

#endif