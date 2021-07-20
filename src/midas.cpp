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

#include <midas/midas.h>

typedef unsigned char uchar;
using namespace boost::python;
namespace p = boost::python;
namespace np = boost::python::numpy;  

namespace midas
{

// Constructor of the interface
MiDas::MiDas(const int image_width,
             const int image_height,
             const char* model_str) :
             output_np_(np::empty(p::make_tuple(input_height_, input_width_), np::dtype::get_builtin<float>())),
             image_height_(image_height), image_width_(image_width)
{
    // Store the original image size
    original_size_ = cv::Size(image_width, image_height);

    // Load model
    module_ = torch::jit::load(model_str);
    module_.to(at::kCUDA);

    // Initialize cv::Mat
    input_cv_ = cv::Mat(input_height_, input_width_, CV_32FC3);
    output_cv_ = cv::Mat(image_height_, image_width_, CV_32FC1);

    printf("Initialized MiDaS-cpp \n");

}

// Destructor (interface)
MiDas::~MiDas() = default;

np::ndarray MiDas::inference(const np::ndarray &image)
{
    // Resize and convert to Torch Tensor
    preprocessImage(reinterpret_cast<uchar*>(image.get_data()));

    // Run network inference
    output_tensor_ = module_.forward({input_tensor_})
                            .toTensor()
                            .squeeze()
                            .detach()
                            .cpu();
    
    // Resize the depth map
    cv::Mat output_mat = cv::Mat(input_height_, input_width_, CV_32FC1, output_tensor_.data_ptr<float>());
    cv::resize(output_mat, output_cv_, original_size_);

    // Convert to np::ndarray
    output_np_ = np::from_data((float*)output_cv_.data,  
                               np::dtype::get_builtin<float>(), 
                               p::make_tuple(image_height_*image_width_),  
                               p::make_tuple(sizeof(float)), 
                               p::object());

    
    // Return the depth map
    return output_np_.copy().reshape(p::make_tuple(image_height_, image_width_));
}

cv::Mat MiDas::inference(const cv::Mat &image)
{
    // Resize and convert to Torch Tensor
    preprocessImage((uchar*)image.data);

    // Run network inference
    output_tensor_ = module_.forward({input_tensor_})
                            .toTensor()
                            .squeeze()
                            .detach()
                            .cpu();

    // Resize the depth map
    cv::Mat output_mat = cv::Mat(input_height_, input_width_, CV_32FC1, output_tensor_.data_ptr<float>());
    cv::resize(output_mat, output_cv_, original_size_);

    // Return the depth map (cv::Mat has a built-in smart pointer to handle the memory)
    return output_cv_.clone();
}

void MiDas::preprocessImage(uchar* image_p) // const np::ndarray &image
{
    // Convert to cv::Mat
    cv::Mat image_cv = cv::Mat(image_height_, image_width_, CV_8UC3, image_p); 

    // Resize the image
    cv::Mat input_cv;

    cv::cvtColor(image_cv, input_cv, CV_BGR2RGB);
    cv::resize(input_cv, input_cv, cv::Size(input_width_, input_height_));
    input_cv.convertTo(input_cv_, CV_32FC3, 1.0f / 255.0f);

    // Convert to (normalized) torch Tensor
    input_tensor_ = torch::from_blob(input_cv_.data, {1, input_height_, input_width_, 3});
    input_tensor_ = input_tensor_.permute({0, 3, 1, 2});
    input_tensor_[0][0] = input_tensor_[0][0].sub_(0.485).div_(0.229);
    input_tensor_[0][1] = input_tensor_[0][1].sub_(0.456).div_(0.224);
    input_tensor_[0][2] = input_tensor_[0][2].sub_(0.406).div_(0.225);
    input_tensor_ = input_tensor_.to(at::kCUDA);
}

} // namespace midas
