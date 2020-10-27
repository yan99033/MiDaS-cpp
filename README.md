# MiDaS-cpp

<p float="left">
  <img src="https://github.com/yan99033/MiDaS-cpp/blob/main/sample_output/midas_icl_nuim.gif" width="300" />
  <img src="https://github.com/yan99033/MiDaS-cpp/blob/main/sample_output/midas_tum_rgbd.gif" width="300" /> 
</p>
<p float="left">
  <img src="https://github.com/yan99033/MiDaS-cpp/blob/main/sample_output/midas_kitti.gif" width="300" />
</p>

PyTorch C++ implementation of MiDaS for single-image relative depth prediction. Relative depth prediction, in general, provides more accurate depth prediction in various scene types by forgoing absolute depth scale, compared to absolute depth prediction (e.g., [monodepth](https://github.com/yan99033/monodepth-cpp)).
For more information, please visit the [original work](https://github.com/intel-isl/MiDaS), which is implemented in Python.

The C++ implementation is especially useful for researchers who are working on robotics problems. Tested on Ubuntu 20.04 with an Intel i7 processor and an Nvidia 1070 graphics card.

## Personal project that uses MiDaS

[(Paper) DeepRelativeFusion: Dense Monocular SLAM using Single-Image Relative Depth Prediction](https://arxiv.org/abs/2006.04047)

[![Watch the video](https://img.youtube.com/vi/Jiz_gzM_4Y4/hqdefault.jpg)](https://www.youtube.com/watch?v=Jiz_gzM_4Y4)

## Tested environment
- Ubuntu 20.04
- Cmake 3.16.3
- Boost 1.71.0
- Python 3.8.5
- OpenCV 3.3.1


## Dependencies
- cmake
- Boost
- Python3
- OpenCV

```bash
apt install cmake libboost-all-dev python3-dev
```

As some of you may have a different OpenCV version, we will let you (build and) install your own OpenCV.


## Prerequisites
1. Clone the original [repo](https://github.com/intel-isl/MiDaS) and download the pre-trained model.
2. Download [PyTorch C++](https://pytorch.org/get-started/locally/) (select LibTorch->C++/Java, download the zip file (Pre-cxx11 ABI), and unzip the file in your Home folder).


## Convert the pre-trained model to Torch Script
Please see the [python](https://github.com/yan99033/MiDaS-cpp/tree/main/python) folder for further instructions.


## Build the project

```bash
mkdir build
cd build
cmake ..
make
```

(Optional) Boost.Python

Uncomment the lines (Line 36 - 42) in CMakelists.txt to build a Boost.Python module. Note that you may have different Boost library version that may result in linking errors.


## Run depth prediction

```bash
cd build
./midas_inference
```

(Optional) Boost.Python
 ```bash
 cd python
 python3 inference.py
 ```

## Use cases
1. **Use MiDaS in a C++ project.** You can import the code to your robotics project (e.g., SLAM, visual navigation, AR, etc.).

2. **Use MiDaS in a Python project.** We also include a Boost.Python module for allowing the model to be used in a Python script.


## Licence

The authors take no credit from [MiDaS](https://github.com/intel-isl/MiDaS), and therefore the licence(s) should remain intact. Please cite their work if you find them helpful. 
