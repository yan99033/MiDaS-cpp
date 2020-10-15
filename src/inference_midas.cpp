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
#include <opencv2/opencv.hpp>

int main()
{
    // Initialize Python runtime and the numpy module
    Py_Initialize();
    np::initialize();

    // Load an image
    cv::Mat image = cv::imread("../sample_image/bicycle.jpg");
    int height = image.rows;
    int width = image.cols;

    // Load model
    std::string model_path = "../traced_model.pt";
    midas::MiDas m(width, height, model_path.c_str());

    // Depth prediction
    cv::Mat depth = m.inference(image);

    // visualize depth
    double min_val, max_val;
    cv::Mat depth_visual;
    cv::minMaxLoc(depth, &min_val, &max_val);
    depth = 255 * (depth - min_val) / (max_val - min_val);
    depth.convertTo(depth_visual, CV_8U);
    cv::applyColorMap(depth_visual, depth_visual, 2); //COLORMAP_JET

    // Stack the image and depth map
    cv::Size img_size = image.size();
    int total_height = img_size.height * 2;
    int total_width = img_size.width;
    cv::Mat full(total_height, total_width, CV_8UC3);
    cv::Mat top(full, cv::Rect(0, 0, img_size.width, img_size.height));
    image.copyTo(top);
    cv::Mat bottom(full, cv::Rect(0, img_size.height, img_size.width, img_size.height));
    depth_visual.copyTo(bottom);

    cv::namedWindow("FULL", CV_WINDOW_AUTOSIZE);
    cv::imshow("FULL", full);
    cv::waitKey(0);

}